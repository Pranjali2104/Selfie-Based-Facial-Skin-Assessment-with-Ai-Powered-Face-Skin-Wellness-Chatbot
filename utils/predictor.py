"""
utils/predictor.py
==================
Loads the trained CNN and exposes:
  • predict(pil_image)  → (label, confidence, class_names)
  • explain(pil_image)  → PIL Image with LIME heatmap overlay
"""

import os
import io
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# LIME
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "skin_model.pt")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 224

_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_NORM_MEAN, _NORM_STD),
])

# ── Build model architecture (must match training) ────────────────────────────
def _build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ── Lazy singleton ─────────────────────────────────────────────────────────────
_model       = None
_class_names = None


def _load_model():
    global _model, _class_names
    if _model is not None:
        return
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    _class_names = ckpt["class_names"]          # e.g. ['dehydrated', 'hydrated']
    _model = _build_model(len(_class_names))
    _model.load_state_dict(ckpt["model_state"])
    _model.eval().to(DEVICE)


# ── Public helpers ─────────────────────────────────────────────────────────────

def predict(pil_image: Image.Image):
    """
    Returns
    -------
    label      : str   – 'hydrated' or 'dehydrated'
    confidence : float – probability of predicted class (0-1)
    class_names: list  – full class list
    probs      : np.ndarray – full probability vector
    """
    _load_model()
    tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx        = int(probs.argmax())
    return _class_names[idx], float(probs[idx]), _class_names, probs


def _batch_predict(np_images: np.ndarray) -> np.ndarray:
    """
    LIME callback: np_images shape (N, H, W, 3) uint8
    Returns softmax probs shape (N, num_classes)
    """
    _load_model()
    tensors = []
    for img in np_images:
        pil = Image.fromarray(img.astype(np.uint8))
        tensors.append(preprocess(pil))
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        logits = _model(batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def explain(pil_image: Image.Image,
            num_samples: int = 500,
            num_features: int = 10,
            positive_only: bool = False) -> Image.Image:
    """
    Generates a LIME heatmap overlay.

    Green  regions = contribute toward predicted class (dehydrated areas).
    Red    regions = against the predicted class (healthy/hydrated areas).

    Returns a PIL Image ready for display.
    """
    _load_model()

    # Resize to model input size for consistency
    rgb_img = np.array(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        rgb_img,
        _batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    predicted_label_idx = int(_batch_predict(rgb_img[None])[0].argmax())

    # Get image + mask with positive + negative contributions
    temp_img, mask = explanation.get_image_and_mask(
        predicted_label_idx,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=False,
    )

    # Build a nice colour overlay
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Heatmap: mark boundaries on temp_img
    overlay = mark_boundaries(temp_img / 255.0, mask)
    axes[1].imshow(overlay)
    label_name = _class_names[predicted_label_idx].upper()
    colour     = "#d62728" if label_name == "DEHYDRATED" else "#2ca02c"
    axes[1].set_title(f"LIME Explanation — Predicted: {label_name}",
                      fontsize=13, fontweight="bold", color=colour)
    axes[1].axis("off")

    # Colour legend text
    fig.text(0.5, 0.02,
             "🟢 Green = region supports prediction   🔴 Red = region opposes prediction",
             ha="center", fontsize=10, color="gray")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()