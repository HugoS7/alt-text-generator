# Alt-Text Generator

Draft alt-text for images that ship without it. EfficientNet-B0 encoder + a Transformer decoder trained from scratch on Flickr8k.

**[▶ Live demo](https://huggingface.co/spaces/YOUR_NAME/alt-text-generator)** · [Training notebook](notebooks/walkthrough.ipynb)

| | |
|---|---|
| *(sample image)* | > "a brown dog runs through the tall grass" |
| *(sample image)* | > "two men in climbing gear scale a rock face" |

---

## Why this exists

Roughly half the images on the public web have no alt-text, which makes them invisible to screen readers. Writing it by hand doesn't scale for a CMS with 100k images. A captioner won't produce publishable alt-text on its own, but it can produce a *draft* that a human edits in five seconds instead of writing from a blank box. That framing (assistive tool, human in the loop) is what this repo builds.

## Architecture

```
image ──> EfficientNet-B0 (frozen) ──> 7×7×1280 feature map
                                              │
                                        flatten to 49 tokens
                                        + linear projection
                                              │
                                              ▼
caption tokens ──> embed + pos ──> Transformer decoder ×4 ──> vocab logits
                                   (causal self-attn +
                                    cross-attn over the 49 image tokens)
```

Two decisions worth calling out:

- **The decoder cross-attends over 49 spatial tokens, not one pooled vector.** Pooling forces the entire image through a single bottleneck; keeping the grid lets each generated word attend to the region it describes.
- **The backbone is frozen *and* held in `eval()` mode.** `requires_grad=False` alone does not freeze BatchNorm running statistics, they keep drifting toward your batches and quietly degrade the pretrained features. See `ImageCaptioner.train()`.

## Results

Corpus BLEU on a held-out test split (5% of images, split by **image** so no image's captions appear in both train and test):

| Metric | Score |
|---|---|
| BLEU-1 | _fill in_ |
| BLEU-2 | _fill in_ |
| BLEU-3 | _fill in_ |
| BLEU-4 | _fill in_ |

## Usage

```bash
pip install -r requirements.txt
# Flickr8k: expects <data-root>/captions.txt and <data-root>/Images/
python -m src.train --data-root data/flickr8k --epochs 20
python -m src.evaluate --data-root data/flickr8k
python app.py
```

## Limitations

Stated plainly, because a captioner that hides these is worse than useless in an accessibility context:

- Trained on 8k images. It fails on anything outside Flickr's distribution (diagrams, screenshots, text-heavy images, product photos).
- It hallucinates plausible-but-wrong details, which for a screen-reader user is worse than no caption. **Output is a draft for human review, not a substitute for it.**
- Flickr8k's captions carry the demographic and geographic biases of 2010-era Flickr users. The model reproduces them.
- BLEU correlates weakly with caption quality. It is reported because it is standard, not because it is sufficient.

## What I'd do next

Swap the frozen CNN for a CLIP ViT encoder, add beam search, and fine-tune the top backbone blocks once the decoder has converged.
