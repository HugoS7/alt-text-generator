"""Gradio demo. Deploy free on Hugging Face Spaces so your README can link to a
live model instead of a screenshot -- this is the highest-leverage 30 minutes in
the whole project."""

import gradio as gr
import numpy as np
import torch

from src.data import build_transforms
from src.evaluate import load_model
from src.vocab import Vocabulary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = Vocabulary.load("checkpoints/vocab.json")
model = load_model("checkpoints/best.pt", vocab, DEVICE)
tf = build_transforms("val")


def caption(image: np.ndarray, temperature: float):
    if image is None:
        return ""
    x = tf(image=image)["image"].unsqueeze(0).to(DEVICE)
    text = model.generate(x, vocab, temperature=temperature)[0]
    return text.capitalize() + "." if text else "(no caption produced)"


demo = gr.Interface(
    fn=caption,
    inputs=[
        gr.Image(type="numpy", label="Image"),
        gr.Slider(0.0, 1.2, value=0.0, step=0.1, label="Temperature (0 = greedy)"),
    ],
    outputs=gr.Textbox(label="Generated alt-text"),
    title="Alt-Text Generator",
    description=(
        "EfficientNet-B0 encoder + Transformer decoder, trained from scratch on "
        "Flickr8k. Generates draft alt-text for images that ship without it."
    ),
)

if __name__ == "__main__":
    demo.launch()
