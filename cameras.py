#!/usr/bin/env python3
import time
import cv2

# adjust this import to point at where your RobotConfig lives
from config import robot_config


def main():
    """
    Capture a single frame from each configured camera and save it to disk,
    printing the camera name and index as it goes.
    """
    for name, cam_cfg in robot_config.cameras.items():
        idx = cam_cfg.camera_index
        print(f"Capturing from camera '{name}' (index {idx})...")

        cap = cv2.VideoCapture(idx)
        # apply your desired settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cam_cfg.fps)

        # give the camera a moment to warm up
        time.sleep(1.0)

        ret, frame = cap.read()
        if not ret:
            print(f"  ❌ Failed to capture from '{name}' (index {idx})")
        else:
            # sanitize the name for filenames
            safe_name = name.replace(" ", "_")
            filename = f"{idx}_{safe_name}.png"
            cv2.imwrite(filename, frame)
            print(f"  ✅ Saved image: {filename}")

        cap.release()


if __name__ == "__main__":
    main()
