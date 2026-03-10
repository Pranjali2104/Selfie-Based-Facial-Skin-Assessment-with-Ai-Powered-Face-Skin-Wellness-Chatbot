# 🧴 SkinSense AI
### Selfie‑Based Facial Skin Assessment with AI‑Powered Face Skin Wellness Chatbot

---

## 📁 Project Structure

```
skin_assessment/
├── app.py                  # Streamlit web app (main entry point)
├── train_model.py          # CNN training script
├── requirements.txt        # Python dependencies
├── .env.example            # Template for API key
├── models/
│   └── skin_model.pt       # ← generated after training
├── utils/
│   ├── predictor.py        # Model inference + LIME XAI
│   └── chatbot.py          # Groq-powered skin chatbot
└── data/
    └── processed/          # ← your dataset goes here
        ├── train/
        │   ├── hydrated/
        │   └── dehydrated/
        ├── val/
        │   ├── hydrated/
        │   └── dehydrated/
        └── test/
            ├── hydrated/
            └── dehydrated/
```

---

## ⚡ Quick Start

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> GPU users: install the CUDA-compatible PyTorch from https://pytorch.org/get-started/locally/

---

### Step 2 — Organise your dataset

Place your processed images (from the `processed/` folder you have) into:

```
data/processed/train/hydrated/      ← ~70% of hydrated images
data/processed/train/dehydrated/    ← ~70% of dehydrated images
data/processed/val/hydrated/        ← ~15%
data/processed/val/dehydrated/      ← ~15%
data/processed/test/hydrated/       ← ~15%
data/processed/test/dehydrated/     ← ~15%
```

You can use this helper script to auto-split:

```python
# run once to split your images
import os, shutil, random
from pathlib import Path

SRC   = "data/processed"         # your original folder
DST   = "data/processed"
SPLIT = (0.70, 0.15, 0.15)       # train / val / test

for cls in ("hydrated", "dehydrated"):
    imgs = list(Path(SRC, cls).glob("*.*"))
    random.shuffle(imgs)
    n = len(imgs)
    cuts = [int(n*SPLIT[0]), int(n*(SPLIT[0]+SPLIT[1]))]
    for split_name, chunk in zip(("train","val","test"),
                                  [imgs[:cuts[0]], imgs[cuts[0]:cuts[1]], imgs[cuts[1]:]]):
        dest = Path(DST, split_name, cls)
        dest.mkdir(parents=True, exist_ok=True)
        for f in chunk:
            shutil.copy(f, dest / f.name)
```

---

### Step 3 — Train the model

```bash
python train_model.py
```

- Trains for 20 epochs (MobileNetV2 fine-tuned)
- Saves best checkpoint to `models/skin_model.pt`
- Prints per-class accuracy report on the test set
- Training time: ~5 min on GPU, ~30–60 min on CPU

---

### Step 4 — Get a FREE Groq API key (for the chatbot)

1. Go to **https://console.groq.com/**
2. Sign up — **no credit card needed**
3. Create an API key
4. Copy `.env.example` → `.env` and paste your key:

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...
```

---

### Step 5 — Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 🔄 Full Pipeline

```
User uploads selfie
        ↓
MobileNetV2 CNN  →  Hydrated / Dehydrated + Confidence %
        ↓
LIME XAI  →  Heatmap overlay (green = dehydrated zones, red = healthy zones)
        ↓
If DEHYDRATED → Chatbot appears
        ↓
"Your skin is dehydrated. Want personalised recommendations?"
        ↓
User selects symptoms (tight / dry / flaky / oily / etc.)
        ↓
Groq LLaMA 3 API → Personalised skin-care advice
        ↓
Free-text follow-up questions (skin topics only)
```

---

## 🤖 Chatbot Details

- **Model**: LLaMA 3.3‑70B via Groq (free tier)
- **API**: `https://api.groq.com/openai/v1/chat/completions`
- **System prompt**: Hard-coded to only answer skin / hydration questions
- **Off-topic guard**: Bot politely declines non-skin questions

---

## 🧠 Model Details

| Property       | Value                    |
|----------------|--------------------------|
| Architecture   | MobileNetV2 (pretrained) |
| Input size     | 224 × 224                |
| Classes        | hydrated, dehydrated     |
| Augmentations  | Flip, Rotate, ColorJitter|
| Optimizer      | Adam (lr=1e-4)           |
| Scheduler      | StepLR (step=7, γ=0.1)  |
| XAI method     | LIME (image segmentation)|

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: lime` | `pip install lime` |
| `FileNotFoundError: skin_model.pt` | Run `train_model.py` first |
| Chatbot returns API error | Check `GROQ_API_KEY` in `.env` |
| Slow LIME on CPU | Reduce `num_samples=100` in `predictor.py` |
| CUDA out of memory | Reduce `BATCH_SIZE` in `train_model.py` |

---

*Built for academic & wellness purposes only. Not a medical diagnostic tool.*