"""
train_model.py  –  Fixed for Windows
=====================================
Run:  python train_model.py
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "processed")
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SAVE_PATH  = os.path.join(MODEL_DIR, "skin_model.pt")

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 2) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def run_epoch(model, loader, criterion, optimizer, is_train, device):
    model.train(is_train)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ── WINDOWS FIX: must be inside if __name__ == '__main__' ─────────────────────
if __name__ == '__main__':

    print(f"Using device: {DEVICE}")

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_tf)

    # num_workers=0  ← fixes RuntimeError on Windows (no multiprocessing fork)
    # pin_memory=False ← fixes UserWarning when no GPU is present
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    CLASS_NAMES = train_ds.classes
    print(f"Classes : {CLASS_NAMES}")
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

    model     = build_model(len(CLASS_NAMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, True,  DEVICE)
        vl_loss, vl_acc = run_epoch(model, val_dl,   criterion, optimizer, False, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  |  "
              f"Val   Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "class_names": CLASS_NAMES,
                "val_acc":     best_val_acc,
            }, SAVE_PATH)
            print(f"  ✓ Saved best model  (val_acc={best_val_acc:.4f})")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n── Test Set Evaluation ──────────────────────────────")
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"\nDone! Model saved to: {SAVE_PATH}")