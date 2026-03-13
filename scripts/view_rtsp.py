"""View an RTSP stream and optionally save a snapshot."""

import argparse
import sys

import cv2


def main():
    parser = argparse.ArgumentParser(description="View an RTSP stream")
    parser.add_argument("--url", default="rtsp://10.255.9.97:8554/the-field",
                        help="RTSP stream URL")
    parser.add_argument("--snapshot", help="Save a snapshot to this path and exit")
    args = parser.parse_args()

    print(f"Connecting to {args.url}...")
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Error: cannot open stream", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Stream: {w}x{h} @ {fps}fps")

    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read frame", file=sys.stderr)
        sys.exit(1)

    if args.snapshot:
        cv2.imwrite(args.snapshot, frame)
        print(f"Saved snapshot to {args.snapshot}")
        cap.release()
        return

    window = "RTSP Stream - Q to quit, S to save snapshot"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w, 1920), min(h, 1080))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or lost connection")
            break

        # Show frame info
        display = frame.copy()
        cv2.rectangle(display, (0, 0), (400, 30), (0, 0, 0), -1)
        cv2.putText(display, f"Frame {frame_idx}  {w}x{h} @ {fps:.0f}fps",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"samples/field_snapshot_{frame_idx}.png"
            cv2.imwrite(path, frame)
            print(f"Saved {path}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
