"""Per-stream configuration loading and saving."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StreamConfig:
    """Configuration for a single camera stream."""

    name: str
    source: str  # file path or RTSP URL
    mode: str  # "inlet" or "outlet"

    # Geometry — provide either line or roi
    line: list[list[int]] | None = None  # [[x1,y1],[x2,y2]]
    roi_points: list[list[int]] = field(default_factory=list)  # polygon vertices

    # HSV thresholds
    hsv_low: tuple[int, int, int] = (20, 100, 100)
    hsv_high: tuple[int, int, int] = (35, 255, 255)

    # Motion counter tuning
    ball_area: int = 900
    band_width: int = 20
    min_peak: int = 0
    fall_ratio: float = 0.5
    cooldown: int = 0

    # Legacy detection parameters (for tracker-based approach)
    min_area: int = 500
    min_circularity: float = 0.5
    max_disappeared: int = 30


def load_configs(path: Path) -> list[StreamConfig]:
    """Load stream configurations from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    configs = []
    for entry in data["streams"]:
        configs.append(
            StreamConfig(
                name=entry["name"],
                source=entry["source"],
                mode=entry["mode"],
                line=entry.get("line"),
                roi_points=entry.get("roi_points", []),
                hsv_low=tuple(entry.get("hsv_low", [20, 100, 100])),
                hsv_high=tuple(entry.get("hsv_high", [35, 255, 255])),
                ball_area=entry.get("ball_area", 900),
                band_width=entry.get("band_width", 20),
                min_peak=entry.get("min_peak", 0),
                fall_ratio=entry.get("fall_ratio", 0.5),
                cooldown=entry.get("cooldown", 0),
                min_area=entry.get("min_area", 500),
                min_circularity=entry.get("min_circularity", 0.5),
                max_disappeared=entry.get("max_disappeared", 30),
            )
        )
    return configs


def save_configs(configs: list[StreamConfig], path: Path) -> None:
    """Save stream configurations to a JSON file."""
    data = {
        "streams": [
            {
                "name": c.name,
                "source": c.source,
                "mode": c.mode,
                **({"line": c.line} if c.line else {}),
                **({"roi_points": c.roi_points} if c.roi_points else {}),
                "hsv_low": list(c.hsv_low),
                "hsv_high": list(c.hsv_high),
                "ball_area": c.ball_area,
                "band_width": c.band_width,
                "min_peak": c.min_peak,
                "fall_ratio": c.fall_ratio,
                "cooldown": c.cooldown,
            }
            for c in configs
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Config saved to {path}")
