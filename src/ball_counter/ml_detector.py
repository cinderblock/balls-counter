"""ML-based peak detector using a trained 1D CNN.

Replaces the threshold-based peak detection in MotionCounter while keeping
the same signal extraction pipeline. Maintains a rolling buffer of signal
values and runs the model to detect new events each frame.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ball_counter.counter import MotionEvent


if HAS_TORCH:
    class PeakDetector1D(nn.Module):
        """Fully-convolutional 1D model for detecting ball events.

        Must match the architecture in scripts/train_peak_model.py.
        """

        def __init__(self, in_channels: int = 1):
            super().__init__()
            self.in_channels = in_channels
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
            h = self.input_conv(x)
            for block in self.blocks:
                h = h + block(h)
            return self.output_conv(h).squeeze(1)


class MLPeakDetector:
    """Wraps the trained 1D CNN for live detection.

    Accumulates signal values during activity. When activity settles (signal
    drops to zero for `quiet_frames`), runs the model on the accumulated
    burst in one shot — exactly as the benchmark does — and emits all
    detected events at once.

    This avoids the frame-by-frame jitter problem: the model sees the same
    complete signal trace that it was trained and benchmarked on.

    Args:
        model_path: Path to the trained .pt checkpoint.
        threshold: Probability threshold for peak detection.
        min_distance: Minimum frames between detected events.
        quiet_frames: Frames of zero signal before finalizing a burst.
        ball_area: Pixel area of one ball (for n_balls estimation).
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.6,
        min_distance: int = 3,
        quiet_frames: int = 10,
        ball_area: int = 900,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ML detection. Install with: uv pip install -e '.[ml]'")

        self.threshold = threshold
        self.min_distance = min_distance
        self.quiet_frames = quiet_frames
        self.ball_area = ball_area

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self.in_channels = ckpt.get("in_channels", 1)
        self.model = PeakDetector1D(in_channels=self.in_channels).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.signal_norm = np.atleast_1d(np.asarray(ckpt["signal_norm"], dtype=np.float32))

        # Rolling signal history (keeps context around bursts)
        # Each entry is a tuple of (area, n_contours, cx, cy, bw, bh) or just (area,)
        self._all_signals: list[tuple] = []
        self._all_start_frame: int = 0  # frame_idx of _all_signals[0]
        # Burst tracking
        self._burst_start_idx: int = 0  # index into _all_signals
        self._quiet_count: int = 0
        self._in_burst: bool = False
        self._last_detected_frame: int = -1000

        # Pending events to emit one-per-frame
        self._pending_events: list[MotionEvent] = []

        # Running count
        self.count = 0

    def process_signal(self, signal_value: int | tuple[int, ...], frame_idx: int,
                       peak_area: int = 0) -> MotionEvent | None:
        """Process one frame's signal value(s). Returns MotionEvent if available.

        Args:
            signal_value: Either a single int (area) or a tuple of features
                          from counter.signal_features.
            frame_idx: Global frame index.
            peak_area: Unused, kept for API compat.
        """
        # Emit pending events first (one per call)
        if self._pending_events:
            return self._pending_events.pop(0)

        # Normalize input to tuple
        if isinstance(signal_value, (int, float)):
            features = (int(signal_value),)
        else:
            features = tuple(int(v) for v in signal_value)

        # Track signal history
        if not self._all_signals:
            self._all_start_frame = frame_idx
        self._all_signals.append(features)

        # Trim old history (keep ~600 frames = 20s)
        max_history = 600
        if len(self._all_signals) > max_history:
            excess = len(self._all_signals) - max_history
            self._all_signals = self._all_signals[excess:]
            self._all_start_frame += excess
            self._burst_start_idx = max(0, self._burst_start_idx - excess)

        # Activity is based on area (first channel)
        area = features[0]
        active = area > 50

        if active:
            if not self._in_burst:
                self._in_burst = True
                # Burst starts at current position in history
                self._burst_start_idx = len(self._all_signals) - 1
            self._quiet_count = 0
        elif self._in_burst:
            self._quiet_count += 1

            if self._quiet_count >= self.quiet_frames:
                # Burst ended — run detection on the full context window
                events = self._detect_burst(frame_idx)
                self._in_burst = False
                self._quiet_count = 0

                if events:
                    self._pending_events = events
                    return self._pending_events.pop(0)

        return None

    def _detect_burst(self, current_frame: int) -> list[MotionEvent]:
        """Run the model on the signal window around the burst."""
        # Include padding before and after the burst for context
        context_pad = 30  # frames of context on each side
        start_idx = max(0, self._burst_start_idx - context_pad)
        end_idx = len(self._all_signals)  # includes quiet tail

        window = self._all_signals[start_idx:end_idx]
        if len(window) < 5:
            return []

        # Build array: (T, C) where C = number of features per frame
        buf_array = np.array(window, dtype=np.float32)  # (T,) or (T, C)
        if buf_array.ndim == 1:
            buf_array = buf_array[:, np.newaxis]  # (T, 1)

        # Per-channel normalize
        normed = buf_array / self.signal_norm[np.newaxis, :]
        normed = np.clip(normed, 0, 3.0)

        with torch.no_grad():
            # (1, C, T) for Conv1d
            x = torch.tensor(normed.T, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        peaks = self._find_peaks(probs)
        window_start_frame = self._all_start_frame + start_idx

        events = []
        for buf_idx in peaks:
            abs_frame = window_start_frame + buf_idx
            # Skip events too close to previously detected ones
            if abs(abs_frame - self._last_detected_frame) <= self.min_distance:
                continue
            area = int(buf_array[buf_idx, 0]) if buf_idx < len(buf_array) else 0
            n_balls = max(1, round(area / self.ball_area)) if area > 0 else 1
            self.count += n_balls
            self._last_detected_frame = abs_frame
            events.append(MotionEvent(frame=abs_frame, n_balls=n_balls, peak_area=area))

        return events

    def _find_peaks(self, probs: np.ndarray) -> list[int]:
        """Find peak indices in probability trace."""
        peaks = []
        above = probs > self.threshold
        i = 0
        while i < len(above):
            if above[i]:
                j = i + 1
                while j < len(above) and above[j]:
                    j += 1
                # Peak = argmax in [i, j)
                peak_idx = i + int(np.argmax(probs[i:j]))
                peaks.append(peak_idx)
                i = j + self.min_distance
            else:
                i += 1
        return peaks
