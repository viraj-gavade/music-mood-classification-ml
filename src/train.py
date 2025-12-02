import os
import sys
import random
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import MoodDataset
from src.models import HybridModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def collate_fn(batch):
    mels = torch.stack([b[0] for b in batch], dim=0)
    math = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.stack([b[2] for b in batch], dim=0)
    return mels, math, labels

def get_file_list(root):
    return sorted([str(p) for p in Path(root).glob("*.npz")])

def create_datasets(root, val_split=0.2, augment=True):
    files = get_file_list(root)
    n = len(files)
    indices = list(range(n))
    random.shuffle(indices)
    split = int((1.0 - val_split) * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = MoodDataset(root=root, augment=augment)
    val_ds = MoodDataset(root=root, augment=False)

    train_ds.files = [files[i] for i in train_idx]
    val_ds.files = [files[i] for i in val_idx]

    train_ds.math_mean, train_ds.math_std = train_ds.compute_math_stats()
    val_ds.math_mean, val_ds.math_std = val_ds.compute_math_stats()

    return train_ds, val_ds

def train_epoch(model, loader, loss_fn, optimizer, device, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for mel, math_vec, y in tqdm(loader, desc="Train", leave=False):
        mel, math_vec, y = mel.to(device), math_vec.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(mel, math_vec)
        loss = loss_fn(logits, y)
        loss.backward()
        
        # Gradient clipping for stability
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        if device == "cuda":
            torch.cuda.empty_cache()

        running_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for mel, math_vec, y in tqdm(loader, desc="Val", leave=False):
            mel, math_vec, y = mel.to(device), math_vec.to(device), y.to(device)
            logits = model(mel, math_vec)
            loss = loss_fn(logits, y)
            running_loss += loss.item() * y.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    return (
        running_loss / total,
        correct / total,
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds, val_ds = create_datasets(args.data_root, args.val_split, augment=True)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # Use multiple workers
        pin_memory=device == "cuda",  # Pin memory if using GPU
        drop_last=True  # Drop incomplete batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=device == "cuda"
    )

    model = HybridModel(num_classes=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer with better defaults
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    # Improved scheduler with cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device, args.grad_clip)
        print(f"Train loss: {train_loss:.4f}  Train acc: {train_acc:.4f}")

        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}")

        # Update scheduler - cosine annealing doesn't need validation loss
        scheduler.step()

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(best_state, "best_model.pth")
            print(f"Saved best model (val_acc={val_acc:.4f})")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            print(f"No improvement for {epochs_since_improve} epoch(s)")

        if epochs_since_improve >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        print(f"\nLoaded best model from epoch {best_state['epoch']} (val_acc={best_state['val_acc']:.4f})")
    else:
        print("No best model found. Using current model.")

    _, _, val_preds, val_labels = eval_epoch(model, val_loader, loss_fn, device)
    labels = ["happy", "sad", "calm", "energetic"]

    print("\nConfusion Matrix:")
    print(confusion_matrix(val_labels, val_preds))
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=labels, digits=4))

    if best_state is not None:
        torch.save(best_state["model"], "model.pth")
    else:
        torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=40)  # Increased epochs
    parser.add_argument("--batch-size", type=int, default=8)  # Larger batch size
    parser.add_argument("--lr", type=float, default=2e-4)  # Lower learning rate
    parser.add_argument("--weight-decay", type=float, default=1e-4)  # Higher weight decay
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=8)  # More patience
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping threshold")
    args = parser.parse_args()
    main(args)
