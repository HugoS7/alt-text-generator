import timm
import torch
import torch.nn as nn


class ImageCaptioner(nn.Module):
    """EfficientNet-B0 encoder -> Transformer decoder.

    Key difference from the naive version: the decoder cross-attends over the
    backbone's *spatial* feature map (7x7 = 49 image tokens), not a single pooled
    vector. A pooled vector forces the whole image through one bottleneck; 49
    tokens let each generated word attend to the region it is describing.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 20,
        model_dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 4,
        dropout: float = 0.1,
        pad_idx: int = 0,
        backbone: str = "efficientnet_b0",
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.context_length = context_length

        self.cnn = timm.create_model(backbone, pretrained=True, features_only=True)
        enc_dim = self.cnn.feature_info.channels()[-1]  # 1280 for efficientnet_b0
        for p in self.cnn.parameters():
            p.requires_grad = False

        # NOTE: this projection must stay OUTSIDE torch.no_grad() at forward time,
        # otherwise it never receives a gradient.
        self.img_proj = nn.Sequential(nn.Linear(enc_dim, model_dim), nn.LayerNorm(model_dim))
        self.img_pos = nn.Parameter(torch.zeros(1, 49, model_dim))
        nn.init.trunc_normal_(self.img_pos, std=0.02)

        self.word_emb = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(context_length, model_dim)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_blocks)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size)

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode always.

        requires_grad=False does NOT freeze BatchNorm running statistics. Without
        this override the pretrained BN stats drift toward your batches and
        quietly degrade the features you're paying to reuse.
        """
        super().train(mode)
        self.cnn.eval()
        return self

    def encode_image(self, images):
        with torch.no_grad():
            feats = self.cnn(images)[-1]              # (B, C, 7, 7)
        feats = feats.flatten(2).transpose(1, 2)      # (B, 49, C)
        return self.img_proj(feats) + self.img_pos    # gradient flows through img_proj

    def decode(self, memory, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.word_emb(tokens) + self.pos_emb(pos))

        # bool mask (True = blocked), matching the dtype of tgt_key_padding_mask
        causal = torch.ones(T, T, dtype=torch.bool, device=tokens.device).triu(1)
        pad_mask = tokens == self.pad_idx

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=pad_mask,
        )
        return self.head(self.norm(out))

    def forward(self, images, tokens):
        """tokens: the *input* sequence, i.e. captions[:, :-1]."""
        return self.decode(self.encode_image(images), tokens)

    @torch.no_grad()
    def generate(self, images, vocab, max_len=None, temperature=0.0):
        """Greedy (temperature=0) or sampled autoregressive decoding."""
        self.eval()
        max_len = max_len or self.context_length
        memory = self.encode_image(images)
        B = images.size(0)
        tokens = torch.full((B, 1), vocab.start_idx, dtype=torch.long, device=images.device)
        done = torch.zeros(B, dtype=torch.bool, device=images.device)

        for _ in range(max_len - 1):
            logits = self.decode(memory, tokens)[:, -1]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, vocab.pad_idx), nxt)
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            done |= nxt == vocab.end_idx
            if done.all():
                break

        return [vocab.decode(row) for row in tokens]
