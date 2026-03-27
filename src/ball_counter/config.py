"""Per-stream configuration loading and saving."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PfmsConfig:
    """Optional PFMS score-forwarding configuration (top-level)."""
    url: str
    key: str | None = None
    source: str = "ball-counter"


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

    # Processing options
    downsample: float = 1.0                    # scale factor before MotionCounter (0.25–1.0)
    crop_override: list[int] | None = None     # [x1,y1,x2,y2] user-set crop, overrides auto

    # AprilTag marker IDs associated with this goal (for alignment/auto-calibration)
    marker_ids: list[int] = field(default_factory=list)

    # PFMS integration
    pfms_element: str | None = None            # PFMS element ID; None = skip forwarding


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
        downsample=entry.get("downsample", 1.0),
        crop_override=entry.get("crop_override"),
        marker_ids=entry.get("marker_ids", []),
        pfms_element=entry.get("pfms_element"),
    )


def _parse_pfms(data: dict) -> PfmsConfig | None:
    url = data.get("pfms_url")
    if not url:
        return None
    return PfmsConfig(
        url=url,
        key=data.get("pfms_key") or None,
        source=data.get("pfms_source", "ball-counter"),
    )


def load_configs(path: Path) -> tuple[list[SourceConfig], PfmsConfig | None]:
    """Load source configurations from a JSON file.

    Returns (sources, pfms_config). pfms_config is None if not configured.
    """
    with open(path) as f:
        data = json.load(f)

    sources = []
    for entry in data["streams"]:
        goals = [_parse_goal(g) for g in entry["goals"]]
        sources.append(SourceConfig(source=entry["source"], goals=goals))
    return sources, _parse_pfms(data)


def save_configs(configs: list[SourceConfig], path: Path,
                 pfms: PfmsConfig | None = None) -> None:
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
        if g.downsample != 1.0:
            d["downsample"] = g.downsample
        if g.crop_override is not None:
            d["crop_override"] = g.crop_override
        if g.marker_ids:
            d["marker_ids"] = g.marker_ids
        if g.pfms_element:
            d["pfms_element"] = g.pfms_element
        return d

    data: dict = {
        "streams": [
            {"source": s.source, "goals": [_goal_dict(g) for g in s.goals]}
            for s in configs
        ]
    }
    if pfms is not None:
        data["pfms_url"] = pfms.url
        if pfms.key:
            data["pfms_key"] = pfms.key
        if pfms.source != "ball-counter":
            data["pfms_source"] = pfms.source
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Config saved to {path}")
