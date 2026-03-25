#!/usr/bin/env python3
"""Train a 1D CNN to detect ball-crossing events from signal traces.

Uses a fully-convolutional sequence-to-sequence architecture:
  Input:  Full signal trace (N frames of area values)
  Output: Per-frame event probability with sharp peaks at ball crossings

The model replaces the threshold-based peak detector while keeping the
existing signal extraction pipeline (bg sub + color mask + zone area).

Training uses Gaussian targets: each human mark becomes a narrow Gaussian
peak (sigma=1 frame). This teaches the model to output distinct spikes
even for rapid consecutive events (4-5 frames apart).

Usage:
    uv run python scripts/extract_signals.py configs/clips --config configs/live.json -o data/signals.npz
    uv run python scripts/train_peak_model.py data/signals.npz -o models/peak_detector.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ── Model ─────────────────────────────────────────────────────────────────────

class PeakDetector1D(nn.Module):
    """Fully-convolutional 1D model for detecting ball events in signal traces.

    Architecture: stack of dilated conv blocks → per-frame logit
    Input:  (batch, C, T) — C signal channels, variable length T
    Output: (batch, T) — per-frame event logit (sigmoid → probability)

    Uses dilated convolutions for large receptive field without pooling,
    preserving temporal resolution needed to separate events 4-5 frames apart.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels

        # Dilated conv stack: receptive field = 1 + 2*(3-1)*(1+2+4+8+16) = 125 frames
        channels = 48
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList()
        for dilation in [1, 2, 4, 8, 16]:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
            ))

        self.output_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        h = self.input_conv(x)
        for block in self.blocks:
            h = h + block(h)  # residual connection
        return self.output_conv(h).squeeze(1)  # (B, T)


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_gaussian_targets(labels: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Convert binary frame labels to Gaussian peaks.

    Each mark frame becomes a Gaussian with the given sigma. This produces
    narrow, distinct peaks even for events 4-5 frames apart.
    """
    n = len(labels)
    target = np.zeros(n, dtype=np.float32)
    mark_frames = np.where(labels > 0)[0]
    if len(mark_frames) == 0:
        return target

    # Vectorized Gaussian computation
    x = np.arange(n, dtype=np.float32)
    for mf in mark_frames:
        target += np.exp(-0.5 * ((x - mf) / sigma) ** 2)

    # Clip to [0, 1] — overlapping Gaussians can exceed 1
    return np.clip(target, 0, 1)


class SignalDataset(Dataset):
    """Dataset of full signal traces with Gaussian peak targets.

    Handles both 1D (N_frames,) and multi-channel (N_frames, C) signals.
    """

    def __init__(self, signals: list[np.ndarray], labels: list[np.ndarray],
                 sigma: float = 1.0, signal_norm: np.ndarray | float | None = None):

        # Determine dimensionality
        sample = signals[0]
        self.n_channels = sample.shape[1] if sample.ndim == 2 else 1

        # Compute per-channel normalization from training data
        if signal_norm is None:
            if self.n_channels > 1:
                stacked = np.concatenate(signals, axis=0)  # (total_frames, C)
                self.signal_norm = np.zeros(self.n_channels, dtype=np.float32)
                for c in range(self.n_channels):
                    col = stacked[:, c]
                    pos = col[col > 0]
                    self.signal_norm[c] = float(np.percentile(pos, 99)) if len(pos) > 0 else 1.0
            else:
                all_vals = np.concatenate(signals)
                self.signal_norm = np.array([
                    float(np.percentile(all_vals[all_vals > 0], 99)) if np.any(all_vals > 0) else 1.0
                ], dtype=np.float32)
        else:
            self.signal_norm = np.atleast_1d(np.asarray(signal_norm, dtype=np.float32))

        self.signals = []
        self.targets = []
        self.n_marks = []

        for sig, lab in zip(signals, labels):
            # Ensure 2D: (T, C)
            if sig.ndim == 1:
                sig = sig[:, np.newaxis]
            # Per-channel normalize
            normed = sig / self.signal_norm[np.newaxis, :]
            normed = np.clip(normed, 0, 3.0)
            # Store as (C, T) for Conv1d
            self.signals.append(torch.tensor(normed.T, dtype=torch.float32))

            # Gaussian targets
            target = make_gaussian_targets(lab, sigma=sigma)
            self.targets.append(torch.tensor(target, dtype=torch.float32))
            self.n_marks.append(int(lab.sum()))

        total_marks = sum(self.n_marks)
        total_frames = sum(s.shape[1] for s in self.signals)
        norm_str = ", ".join(f"{v:.0f}" for v in self.signal_norm)
        print(f"  Dataset: {len(self.signals)} clips, {total_frames} frames, "
              f"{total_marks} marks, {self.n_channels}ch, norm=[{norm_str}]")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.targets[idx], self.n_marks[idx]


def collate_fn(batch):
    """Pad variable-length sequences to the same length in a batch."""
    signals, targets, n_marks = zip(*batch)

    # signals are (C, T) — pad along T dimension
    # pad_sequence expects (T, ...) so transpose, pad, transpose back
    signals_t = [s.T for s in signals]  # list of (T, C)
    signals_padded = pad_sequence(signals_t, batch_first=True)  # (B, T_max, C)
    signals_padded = signals_padded.permute(0, 2, 1)  # (B, C, T_max)

    targets_padded = pad_sequence(targets, batch_first=True)  # (B, T_max)

    # Create mask for valid positions
    lengths = torch.tensor([s.shape[1] for s in signals])  # T dimension
    mask = torch.arange(signals_padded.size(2)).unsqueeze(0) < lengths.unsqueeze(1)

    return (signals_padded,    # (B, C, T)
            targets_padded,    # (B, T)
            mask,              # (B, T)
            torch.tensor(n_marks))


# ── Training ──────────────────────────────────────────────────────────────────

def count_peaks(probs: np.ndarray, threshold: float = 0.5, min_distance: int = 3) -> int:
    """Count peaks in a probability trace (for validation metrics)."""
    above = probs > threshold
    count = 0
    i = 0
    while i < len(above):
        if above[i]:
            # Find end of this peak
            j = i + 1
            while j < len(above) and above[j]:
                j += 1
            count += 1
            i = j + min_distance  # skip min_distance frames
        else:
            i += 1
    return count


def find_peak_times(probs: np.ndarray, fps: float,
                    threshold: float = 0.5, min_distance: int = 3) -> list[float]:
    """Find event times from probability trace."""
    events = []
    above = probs > threshold
    i = 0
    while i < len(above):
        if above[i]:
            j = i + 1
            while j < len(above) and above[j]:
                j += 1
            # Peak = argmax in [i, j)
            peak_frame = i + np.argmax(probs[i:j])
            events.append(peak_frame / fps)
            i = j + min_distance
        else:
            i += 1
    return events


def train(
    data_path: Path,
    output_path: Path,
    sigma: float = 1.0,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 25,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data = np.load(data_path, allow_pickle=True)
    signals = list(data["signals"])
    labels = list(data["labels"])
    clip_names = list(data["clip_names"])

    n_channels = int(data["n_channels"]) if "n_channels" in data else 1
    print(f"Loaded {len(signals)} clips, {n_channels} channel(s)")

    # Train/val split by clip
    n = len(signals)
    indices = np.random.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx = set(indices[:n_val].tolist())
    train_idx = set(indices[n_val:].tolist())

    train_sigs = [signals[i] for i in sorted(train_idx)]
    train_labs = [labels[i] for i in sorted(train_idx)]
    val_sigs = [signals[i] for i in sorted(val_idx)]
    val_labs = [labels[i] for i in sorted(val_idx)]

    print(f"Train: {len(train_sigs)} clips, Val: {len(val_sigs)} clips")

    print("\nBuilding training dataset...")
    train_ds = SignalDataset(train_sigs, train_labs, sigma=sigma)
    print("Building validation dataset...")
    val_ds = SignalDataset(val_sigs, val_labs, sigma=sigma,
                           signal_norm=train_ds.signal_norm)

    # Weight clips by mark count so rapid-burst clips dominate training
    clip_weights = [max(1, m) for m in train_ds.n_marks]  # at least 1 for zero-mark clips
    sampler = torch.utils.data.WeightedRandomSampler(
        clip_weights, num_samples=len(train_ds), replacement=True)

    n_burst = sum(1 for m in train_ds.n_marks if m >= 5)
    n_single = sum(1 for m in train_ds.n_marks if 0 < m < 5)
    n_zero = sum(1 for m in train_ds.n_marks if m == 0)
    print(f"  Clip weighting: {n_burst} burst (>=5), {n_single} single/spaced (1-4), {n_zero} zero-mark")

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Model
    model = PeakDetector1D(in_channels=train_ds.n_channels).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {param_count:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Weighted BCE: positive frames are rare (~3% with sigma=1)
    # Count actual positive ratio from targets
    pos_ratio = sum(float((t > 0.5).sum()) for t in train_ds.targets) / \
                sum(len(t) for t in train_ds.targets)
    pos_weight_val = (1 - pos_ratio) / max(pos_ratio, 1e-6)
    print(f"Positive ratio: {pos_ratio:.3f}, pos_weight: {pos_weight_val:.1f}")

    best_val_mae = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches = 0
        for signals_b, targets_b, mask_b, _ in train_loader:
            signals_b = signals_b.to(device)
            targets_b = targets_b.to(device)
            mask_b = mask_b.to(device)

            logits = model(signals_b)  # (B, T)

            # Masked BCE loss with per-element pos weighting
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets_b, reduction='none')
            # Weight positive frames more
            weights = torch.where(targets_b > 0.5, pos_weight_val, 1.0)
            loss = (bce * weights * mask_b).sum() / mask_b.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # ── Validate ── (event-level: count accuracy)
        model.eval()
        val_loss = 0.0
        total_human = 0
        total_predicted = 0
        total_abs_error = 0
        n_val_batches = 0

        with torch.no_grad():
            for signals_b, targets_b, mask_b, n_marks_b in val_loader:
                signals_b = signals_b.to(device)
                targets_b = targets_b.to(device)
                mask_b = mask_b.to(device)

                logits = model(signals_b)
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_b, reduction='none')
                weights = torch.where(targets_b > 0.5, pos_weight_val, 1.0)
                val_loss += (bce * weights * mask_b).sum().item() / mask_b.sum().item()
                n_val_batches += 1

                probs = torch.sigmoid(logits).cpu().numpy()
                lengths = mask_b.sum(dim=1).cpu().numpy()

                for i in range(len(n_marks_b)):
                    prob_i = probs[i, :int(lengths[i])]
                    predicted = count_peaks(prob_i, threshold=0.5, min_distance=3)
                    human = int(n_marks_b[i])
                    total_human += human
                    total_predicted += predicted
                    total_abs_error += abs(predicted - human)

        val_loss /= max(1, n_val_batches)
        train_loss /= max(1, n_batches)
        mae = total_abs_error / max(1, len(val_ds))

        lr_now = optimizer.param_groups[0]['lr']
        count_err = total_predicted - total_human
        print(f"  epoch {epoch:3d}  loss={train_loss:.4f}/{val_loss:.4f}  "
              f"counts={total_predicted}/{total_human} ({count_err:+d})  "
              f"MAE={mae:.2f}  lr={lr_now:.1e}")

        if mae < best_val_mae:
            best_val_mae = mae
            best_epoch = epoch
            no_improve = 0

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "signal_norm": train_ds.signal_norm.tolist(),
                "in_channels": train_ds.n_channels,
                "sigma": sigma,
                "val_mae": float(mae),
                "val_counts": f"{total_predicted}/{total_human}",
                "epoch": epoch,
            }, output_path)
            print(f"    -> saved (best MAE={mae:.2f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest: epoch {best_epoch}, val MAE={best_val_mae:.2f}")
    print(f"Model saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Train 1D CNN peak detector")
    ap.add_argument("data", help="Path to signals.npz from extract_signals.py")
    ap.add_argument("-o", "--output", default="models/peak_detector.pt", help="Output model path")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian target width in frames (default: 1.0)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(
        Path(args.data),
        Path(args.output),
        sigma=args.sigma,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
