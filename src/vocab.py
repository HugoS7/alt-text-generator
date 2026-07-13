"""Minimal vocabulary. Replaces torchtext, which is unmaintained and no longer
installs cleanly alongside current PyTorch."""

import json
import re
from collections import Counter

PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"
SPECIALS = [PAD, UNK, START, END]

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation. Deliberately simple and dependency-free."""
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.pad_idx = self.stoi[PAD]
        self.unk_idx = self.stoi[UNK]
        self.start_idx = self.stoi[START]
        self.end_idx = self.stoi[END]

    @classmethod
    def build(cls, captions: list[str], min_freq: int = 5) -> "Vocabulary":
        counter = Counter()
        for cap in captions:
            counter.update(tokenize(cap))
        words = sorted(w for w, c in counter.items() if c >= min_freq)
        return cls(SPECIALS + words)

    def encode(self, text: str, context_length: int) -> list[int]:
        """<start> tokens... <end> <pad>... , always exactly context_length long."""
        ids = [self.stoi.get(w, self.unk_idx) for w in tokenize(text)]
        ids = [self.start_idx] + ids[: context_length - 2] + [self.end_idx]
        ids += [self.pad_idx] * (context_length - len(ids))
        return ids

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            i = int(i)
            if i == self.end_idx:
                break
            if i in (self.pad_idx, self.start_idx):
                continue
            words.append(self.itos[i])
        return " ".join(words)

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))
