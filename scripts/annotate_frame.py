"""Annotate a frame with labeled regions of interest."""

import cv2
import numpy as np

frame = cv2.imread("frame_75.png")
h, w = frame.shape[:2]
frame = frame[:, : w // 2]

# --- Label key areas ---

# 1. Staging row at top — the pre-loaded balls along the top edge
cv2.rectangle(frame, (50, 0), (1100, 100), (255, 0, 0), 3)
cv2.putText(frame, "STAGING ROW", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# 2. Goal structure — the wooden box/chute where balls go in
pts_goal = np.array([(250, 150), (600, 150), (620, 500), (230, 500)], np.int32)
cv2.polylines(frame, [pts_goal], True, (0, 0, 255), 3)
cv2.putText(frame, "GOAL STRUCTURE", (260, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# 3. Dump zone — where scored balls scatter on the dark floor
pts_dump = np.array([(100, 100), (1050, 100), (1200, 600), (700, 700), (50, 500)], np.int32)
cv2.polylines(frame, [pts_dump], True, (0, 255, 0), 3)
cv2.putText(frame, "DUMP ZONE (outlet ROI)", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 4. Robot area
cv2.rectangle(frame, (750, 350), (1050, 600), (255, 255, 0), 3)
cv2.putText(frame, "ROBOT", (770, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# 5. Below the blue tape — field floor (no scoring action here)
cv2.line(frame, (0, 640), (1344, 640), (128, 128, 128), 2)
cv2.putText(frame, "FIELD FLOOR (below blue tape)", (20, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

# Legend
y = frame.shape[0] - 20
cv2.putText(frame, "BLUE=staging  RED=goal  GREEN=dump zone (where we count)  YELLOW=robot",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

cv2.imwrite("annotated_regions.png", frame)
print("Saved annotated_regions.png")
