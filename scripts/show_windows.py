"""Show crop windows on a frame from each new-angle sample."""

import json
import cv2
import numpy as np

PADDING = 150
SAMPLES = [
    ("samples/new-angle/sample-red.mp4", "samples/new-angle/sample-red.geometry.json", 200),
    ("samples/new-angle/sample-blue.mp4", "samples/new-angle/sample-blue.geometry.json", 300),
]

for video_path, geom_path, frame_num in SAMPLES:
    with open(geom_path) as f:
        geometries = json.load(f)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    # Dim the full frame
    dimmed = (frame * 0.3).astype(np.uint8)
    display = dimmed.copy()

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

    for i, geom in enumerate(geometries):
        pts = geom["points"]
        color = colors[i % len(colors)]

        # Compute crop window
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1 = max(0, min(xs) - PADDING)
        y1 = max(0, min(ys) - PADDING)
        x2 = min(w, max(xs) + PADDING)
        y2 = min(h, max(ys) + PADDING)

        # Paste bright original into the window
        display[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        # Draw the line
        cv2.line(display, tuple(pts[0]), tuple(pts[1]), color, 3)

        # Draw crop window border
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
        label = f"{geom['name']}: {x2-x1}x{y2-y1}"
        cv2.putText(display, label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    out_path = video_path.replace(".mp4", "_windows.png")
    cv2.imwrite(out_path, display)
    print(f"Saved: {out_path}")

    window = video_path.split("/")[-1]
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1920, 1080)
    cv2.imshow(window, display)

print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
