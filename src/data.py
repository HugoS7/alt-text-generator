import os

import albumentations as alb
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from .vocab import Vocabulary

IMG_SIZE = 224


def load_captions(root: str) -> pd.DataFrame:
    """Flickr8k captions.txt -> one row per (image, caption) pair.

    Your original code kept one row per *image* with a list of 5 captions, then
    threw 4 of them away each epoch. Exploding to one row per caption means the
    model sees all ~40k captions per epoch instead of ~8k.
    """
    path = os.path.join(root, "captions.txt")
    df = pd.read_csv(path)  # columns: image, caption
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"image": "filename"})

    # Drop rows whose image file is actually missing on disk.
    img_dir = os.path.join(root, "Images")
    present = set(os.listdir(img_dir))
    df = df[df["filename"].isin(present)].reset_index(drop=True)
    df["caption"] = df["caption"].astype(str).str.strip()
    return df


def split_by_image(df: pd.DataFrame, val_frac=0.05, test_frac=0.05, seed=0):
    """Split on *images*, not rows. Splitting on rows would put captions of the
    same image in both train and val -- a subtle but real leak."""
    images = df["filename"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    n = len(images)
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    test, val, train = images[:n_test], images[n_test : n_test + n_val], images[n_test + n_val :]
    pick = lambda keys: df[df["filename"].isin(set(keys))].reset_index(drop=True)
    return pick(train), pick(val), pick(test)


def build_transforms(split: str):
    tfms = [alb.Resize(IMG_SIZE, IMG_SIZE)]
    if split == "train":
        tfms += [
            alb.HorizontalFlip(p=0.5),
            alb.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
        ]
    tfms += [alb.Normalize(), ToTensorV2()]
    return alb.Compose(tfms)


class Flickr8kDataset(Dataset):
    def __init__(self, df, root, vocab: Vocabulary, context_length=20, split="train"):
        self.df = df
        self.img_dir = os.path.join(root, "Images")
        self.vocab = vocab
        self.context_length = context_length
        self.tf = build_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bgr = cv2.imread(os.path.join(self.img_dir, row["filename"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.tf(image=rgb)["image"]
        caption = torch.tensor(
            self.vocab.encode(row["caption"], self.context_length), dtype=torch.long
        )
        return image, caption
