"""Extract zoomed-in crops of the goal structures from sample videos."""

import cv2

# Blue goal - the wooden box/chute is roughly center-left of the left-half
cap = cv2.VideoCapture("samples/blue/sample1-full.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, blue = cap.read()
cap.release()
if ret:
    h, w = blue.shape[:2]
    blue = blue[:, : w // 2]
    # Goal structure with some surrounding context
    goal = blue[100:700, 50:650]
    cv2.imwrite("samples/blue/goal_closeup.png", goal)
    print(f"Blue goal closeup: {goal.shape[1]}x{goal.shape[0]}")

# Red goal - goal structure is upper-left
cap = cv2.VideoCapture("samples/red/sample1-full.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, red = cap.read()
cap.release()
if ret:
    h, w = red.shape[:2]
    red = red[:, : w // 2]
    goal = red[0:600, 0:700]
    cv2.imwrite("samples/red/goal_closeup.png", goal)
    print(f"Red goal closeup: {goal.shape[1]}x{goal.shape[0]}")
