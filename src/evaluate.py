"""Corpus BLEU on the held-out test split.

Reporting a real metric on a leak-free split is the difference between "I trained
a model" and "I evaluated a model." Reviewers notice.
"""

import argparse
import os
from collections import defaultdict

import cv2
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from .data import build_transforms
from .model import ImageCaptioner
from .vocab import Vocabulary, tokenize


def load_model(ckpt_path, vocab, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ImageCaptioner(
        vocab_size=len(vocab),
        context_length=a["context_length"],
        model_dim=a["model_dim"],
        num_heads=a["num_heads"],
        num_blocks=a["num_blocks"],
        dropout=0.0,
        pad_idx=vocab.pad_idx,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--vocab", default="checkpoints/vocab.json")
    p.add_argument("--test-csv", default="checkpoints/test_split.csv")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(args.vocab)
    model = load_model(args.ckpt, vocab, device)

    df = pd.read_csv(args.test_csv)
    refs = defaultdict(list)
    for _, r in df.iterrows():
        refs[r["filename"]].append(tokenize(str(r["caption"])))

    tf = build_transforms("val")
    files = list(refs)
    references, hypotheses = [], []

    for i in range(0, len(files), args.batch_size):
        chunk = files[i : i + args.batch_size]
        batch = []
        for fn in chunk:
            bgr = cv2.imread(os.path.join(args.data_root, "Images", fn))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            batch.append(tf(image=rgb)["image"])
        caps = model.generate(torch.stack(batch).to(device), vocab)
        for fn, cap in zip(chunk, caps):
            references.append(refs[fn])
            hypotheses.append(tokenize(cap))

    sm = SmoothingFunction().method1
    print(f"images evaluated: {len(hypotheses)}")
    for n in range(1, 5):
        w = tuple([1 / n] * n)
        b = corpus_bleu(references, hypotheses, weights=w, smoothing_function=sm)
        print(f"BLEU-{n}: {b*100:.2f}")


if __name__ == "__main__":
    main()
