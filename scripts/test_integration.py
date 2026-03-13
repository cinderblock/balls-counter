"""Quick integration test: run the MotionCounter via StreamProcessor."""

import json
import sys

sys.path.insert(0, "src")
from ball_counter.config import StreamConfig
from ball_counter.stream import StreamProcessor


# Test red outlet with line
config = StreamConfig(
    name="red-outlet",
    source="samples/red/sample1-goal.mp4",
    mode="outlet",
    line=[[349, 386], [522, 363]],
    ball_area=900,
    band_width=20,
)

proc = StreamProcessor(config)
if not proc.open():
    print("Failed to open stream")
    sys.exit(1)

while proc.read_frame():
    event = proc.process_frame()
    if event:
        print(f"  Frame {event.frame:4d}: +{event.n_balls} (peak={event.peak_area}) Total: {proc.count}")

print(f"\nRed outlet final count: {proc.count}")
proc.release()

with open("samples/red/sample1-goal-drops2.scores.json") as f:
    gt = json.load(f)
print(f"Ground truth: {len(gt)}")
print(f"Match: {'YES' if proc.count == len(gt) else 'NO'}")

# Test blue inlet with ROI
print(f"\n{'='*40}")
config2 = StreamConfig(
    name="blue-inlet",
    source="samples/blue/sample1-goal.mp4",
    mode="inlet",
    roi_points=[[207, 421], [256, 378], [290, 408], [279, 494], [240, 533], [210, 507]],
    ball_area=1100,
    band_width=10,
)

proc2 = StreamProcessor(config2)
if not proc2.open():
    print("Failed to open blue stream")
    sys.exit(1)

while proc2.read_frame():
    event = proc2.process_frame()
    if event:
        print(f"  Frame {event.frame:4d}: +{event.n_balls} (peak={event.peak_area}) Total: {proc2.count}")

print(f"\nBlue inlet final count: {proc2.count}")
proc2.release()

with open("samples/blue/sample1-goal.scores.json") as f:
    gt_blue = json.load(f)
print(f"Ground truth: {len(gt_blue)}")
print(f"Difference: {proc2.count - len(gt_blue):+d}")
