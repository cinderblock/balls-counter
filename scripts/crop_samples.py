"""Crop sample videos to a closeup of the goal structure area."""

import cv2

crops = [
    {
        "input": "samples/blue/sample1-full.mp4",
        "output": "samples/blue/sample1-goal.mp4",
        # Goal structure area in the left-half frame
        "x": 50, "y": 100, "w": 600, "h": 600,
        "half": True,
    },
    {
        "input": "samples/red/sample1-full.mp4",
        "output": "samples/red/sample1-goal.mp4",
        # Goal structure area in the left-half frame
        "x": 0, "y": 0, "w": 700, "h": 600,
        "half": True,
    },
]

for c in crops:
    cap = cv2.VideoCapture(c["input"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(c["output"], fourcc, fps, (c["w"], c["h"]))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if c.get("half"):
            frame = frame[:, : frame.shape[1] // 2]
        crop = frame[c["y"] : c["y"] + c["h"], c["x"] : c["x"] + c["w"]]
        out.write(crop)
        count += 1

    cap.release()
    out.release()
    print(f"{c['output']}: {count} frames, {c['w']}x{c['h']} @ {fps:.0f}fps")
