"""Per-stream configuration loading and saving."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoalConfig:
    """Configuration for a single counting zone (goal) on a stream."""

    name: str
    mode: str  # "inlet" or "outlet"

    # Geometry — provide either line or roi_points
    line: list[list[int]] | None = None        # [[x1,y1],[x2,y2]]
    roi_points: list[list[int]] = field(default_factory=list)  # polygon vertices

    # HSV thresholds for ball color
    hsv_low: tuple[int, int, int] = (20, 100, 100)
    hsv_high: tuple[int, int, int] = (35, 255, 255)

    # Overlay color in BGR (default red)
    draw_color: tuple[int, int, int] = (0, 0, 255)

    # Motion counter tuning
    ball_area: int = 900
    band_width: int = 20
    min_peak: int = 0
    fall_ratio: float = 0.5
    cooldown: int = 0


@dataclass
class SourceConfig:
    """Configuration for a single video source (camera or file) with one or more goals."""

    source: str  # file path or RTSP URL
    goals: list[GoalConfig]


def _parse_goal(entry: dict) -> GoalConfig:
    return GoalConfig(
        name=entry["name"],
        mode=entry["mode"],
        line=entry.get("line"),
        roi_points=entry.get("roi_points", []),
        hsv_low=tuple(entry.get("hsv_low", [20, 100, 100])),
        hsv_high=tuple(entry.get("hsv_high", [35, 255, 255])),
        draw_color=tuple(entry.get("draw_color", [0, 0, 255])),
        ball_area=entry.get("ball_area", 900),
        band_width=entry.get("band_width", 20),
        min_peak=entry.get("min_peak", 0),
        fall_ratio=entry.get("fall_ratio", 0.5),
        cooldown=entry.get("cooldown", 0),
    )


def load_configs(path: Path) -> list[SourceConfig]:
    """Load source configurations from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    sources = []
    for entry in data["streams"]:
        goals = [_parse_goal(g) for g in entry["goals"]]
        sources.append(SourceConfig(source=entry["source"], goals=goals))
    return sources


def save_configs(configs: list[SourceConfig], path: Path) -> None:
    """Save source configurations to a JSON file."""

    def _goal_dict(g: GoalConfig) -> dict:
        d: dict = {"name": g.name, "mode": g.mode}
        if g.line:
            d["line"] = g.line
        if g.roi_points:
            d["roi_points"] = g.roi_points
        d.update({
            "draw_color": list(g.draw_color),
            "hsv_low": list(g.hsv_low),
            "hsv_high": list(g.hsv_high),
            "ball_area": g.ball_area,
            "band_width": g.band_width,
            "min_peak": g.min_peak,
            "fall_ratio": g.fall_ratio,
            "cooldown": g.cooldown,
        })
        return d

    data = {
        "streams": [
            {"source": s.source, "goals": [_goal_dict(g) for g in s.goals]}
            for s in configs
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Config saved to {path}")
