import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import Flickr8kDataset, load_captions, split_by_image
from .model import ImageCaptioner
from .vocab import Vocabulary


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, log_every=100):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0

    for step, (images, captions) in enumerate(loader):
        images, captions = images.to(device, non_blocking=True), captions.to(device)

        # Teacher forcing: predict token t+1 from tokens <= t.
        # This shift is the single most commonly botched line in a captioner.
        inputs, targets = captions[:, :-1], captions[:, 1:]

        with torch.autocast("cuda", enabled=scaler is not None):
            logits = model(images, inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)   # was missing entirely
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if step % log_every == 0:
                print(f"  step {step:5d} | loss {loss.item():.4f}")

        total += loss.item() * images.size(0)
        n += images.size(0)

    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="dir with captions.txt and Images/")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)   # 1e-5 was far too low: the
    p.add_argument("--weight-decay", type=float, default=0.01)  # decoder trains from scratch
    p.add_argument("--context-length", type=int, default=20)
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)  # 0.5 would cripple this model
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_captions(args.data_root)
    train_df, val_df, test_df = split_by_image(df)
    print(f"{len(df)} captions | train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")

    vocab = Vocabulary.build(train_df["caption"].tolist(), min_freq=args.min_freq)
    vocab.save(os.path.join(args.out, "vocab.json"))
    test_df.to_csv(os.path.join(args.out, "test_split.csv"), index=False)
    print(f"vocab size: {len(vocab)}")

    mk = lambda d, s: DataLoader(
        Flickr8kDataset(d, args.data_root, vocab, args.context_length, s),
        batch_size=args.batch_size,
        shuffle=(s == "train"),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=(s == "train"),
    )
    train_loader, val_loader = mk(train_df, "train"), mk(val_df, "val")

    model = ImageCaptioner(
        vocab_size=len(vocab),
        context_length=args.context_length,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        pad_idx=vocab.pad_idx,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1
    )
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"epoch {epoch}/{args.epochs}")
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        with torch.no_grad():
            va = run_epoch(model, val_loader, loss_fn, device)
        sched.step()
        print(f"  train {tr:.4f} | val {va:.4f} | {time.time()-t0:.0f}s")

        if va < best:
            best = va
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "vocab_size": len(vocab)},
                os.path.join(args.out, "best.pt"),
            )
            print(f"  saved (val {va:.4f})")

        # Eyeball a couple of samples -- loss alone hides degenerate outputs.
        images, _ = next(iter(val_loader))
        for c in model.generate(images[:3].to(device), vocab):
            print(f"  > {c}")


if __name__ == "__main__":
    main()
