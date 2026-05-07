#!/usr/bin/env python3
"""
Weed Detection Camera Script
=============================
Runs the trained YOLOv8 segmentation model on a live camera feed.

Classes:
  - Crop (class 0): Bean plants — shown in GREEN
  - Weed (class 1): Unwanted plants — shown in RED

Controls:
  - Press 'q' to quit
  - Press 's' to save a screenshot
  - Press 'c' to toggle confidence display
  - Press '+'/'-' to adjust confidence threshold
"""

import sys
import cv2
import numpy as np
import time
import os
from pathlib import Path
from ultralytics import YOLO

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed — LED control disabled. Run: pip install pyserial")


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "model_results" / "weed_segmentation" / "weights" / "best.pt"
CAMERA_INDEX = 0                 # 0 = default webcam; change if you have multiple cameras
MAX_FRAME_FAILURES = 30          # quit after this many consecutive failed frame reads
CONFIDENCE_THRESHOLD = 0.5       # minimum confidence to display a detection
IMG_SIZE = 640                   # inference resolution (must match training)
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"

# ── ESP32 Serial Configuration ────────────────────────────────────────────────
# Run `ls /dev/cu.*` to find your ESP32 port after plugging it in via USB
ESP32_PORT = "/dev/cu.usbserial-10"     # <-- CHANGE THIS to your port
ESP32_BAUD  = 9600
LED_SIGNAL_INTERVAL = 1.0               # seconds between LED signals (avoid flooding)

# Class colours (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 200, 0),    # Crop  → green
    1: (0, 0, 220),    # Weed  → red
}
CLASS_NAMES = {0: "Crop", 1: "Weed"}

# Overlay transparency for segmentation masks
MASK_ALPHA = 0.45


# ── Helper functions ──────────────────────────────────────────────────────────

def draw_results(frame, results, conf_thresh, show_conf=True):
    """Draw segmentation masks and bounding boxes on the frame."""
    overlay = frame.copy()

    for result in results:
        # ── Segmentation masks ────────────────────────────────────────────
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()          # (N, H, W)
            boxes = result.boxes

            for i, mask in enumerate(masks):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                if conf < conf_thresh:
                    continue

                color = CLASS_COLORS.get(cls_id, (255, 255, 255))

                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_bool = mask_resized > 0.5

                # Apply coloured overlay on the mask area
                overlay[mask_bool] = (
                    np.array(color, dtype=np.uint8) * MASK_ALPHA
                    + overlay[mask_bool] * (1 - MASK_ALPHA)
                ).astype(np.uint8)

            frame = overlay.copy()

        # ── Bounding boxes + labels ───────────────────────────────────────
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if conf < conf_thresh:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
                if show_conf:
                    label = f"{label} {conf:.0%}"

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, fps, conf_thresh, show_conf):
    """Draw a heads-up display with FPS, confidence threshold, and controls."""
    h, w = frame.shape[:2]

    # Semi-transparent banner at the top
    banner_h = 36
    banner = frame[:banner_h, :].copy()
    cv2.rectangle(frame, (0, 0), (w, banner_h), (30, 30, 30), -1)
    frame[:banner_h, :] = cv2.addWeighted(frame[:banner_h, :], 0.7, banner, 0.3, 0)

    # Info text
    info = (
        f"FPS: {fps:.1f}  |  Conf: {conf_thresh:.0%}  |  "
        f"[q] Quit  [s] Save  [c] Toggle conf  [+/-] Threshold"
    )
    cv2.putText(frame, info, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    # Legend (bottom-left)
    y_off = h - 20
    for cls_id in sorted(CLASS_NAMES):
        color = CLASS_COLORS[cls_id]
        name = CLASS_NAMES[cls_id]
        cv2.circle(frame, (18, y_off), 8, color, -1)
        cv2.putText(frame, name, (32, y_off + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y_off -= 28

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────

def connect_esp32():
    """Try to open serial connection to ESP32. Returns serial object or None."""
    if not SERIAL_AVAILABLE:
        return None
    try:
        esp = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
        print(f"[INFO] ESP32 connected on {ESP32_PORT}")
        return esp
    except Exception as e:
        print(f"[WARN] Could not connect to ESP32: {e}")
        print(f"       LED blinking disabled. Check ESP32_PORT in the script.")
        return None


def send_led_signal(esp32, signal: bytes):
    """Send a single-byte command to the ESP32 safely."""
    if esp32 is None:
        return
    try:
        esp32.write(signal)
    except Exception:
        pass


def detect_classes(results, conf_thresh):
    """Return set of class IDs found above the confidence threshold."""
    found = set()
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if float(box.conf.item()) >= conf_thresh:
                found.add(int(box.cls.item()))
    return found


def main():
    # Validate model exists
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model weights not found at: {MODEL_PATH}")
        print("        Make sure you have the trained model in the correct location.")
        return

    print(f"[INFO] Loading model from {MODEL_PATH} ...")
    model = YOLO(str(MODEL_PATH))
    print("[INFO] Model loaded successfully.")

    esp32 = connect_esp32()

    # Open camera — use AVFoundation backend on macOS for better compatibility
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index={CAMERA_INDEX}).")
        print("        Try changing CAMERA_INDEX at the top of this script.")
        return

    # Try to set camera resolution (may not work on all cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened — resolution {actual_w}×{actual_h}")
    print("[INFO] Press 'q' to quit.")

    # Screenshot directory
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    conf_thresh = CONFIDENCE_THRESHOLD
    show_conf = True
    prev_time = time.time()
    last_led_time = 0.0
    frame_failures = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            frame_failures += 1
            print(f"[WARN] Failed to grab frame ({frame_failures}/{MAX_FRAME_FAILURES}) ...")
            if frame_failures >= MAX_FRAME_FAILURES:
                print("[ERROR] Too many consecutive frame failures — exiting.")
                break
            time.sleep(0.1)
            continue
        frame_failures = 0

        # ── Run inference ─────────────────────────────────────────────────
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=conf_thresh,
            verbose=False,
        )

        # ── LED signal based on detections ────────────────────────────────
        now = time.time()
        if now - last_led_time >= LED_SIGNAL_INTERVAL:
            detected = detect_classes(results, conf_thresh)
            if 1 in detected:                        # Weed detected → red LED
                send_led_signal(esp32, b'W')
                print("[LED] Weed detected — blinking RED")
                last_led_time = now
            elif 0 in detected:                      # Crop only → green LED
                send_led_signal(esp32, b'C')
                print("[LED] Crop detected — blinking GREEN")
                last_led_time = now

        # ── Draw detections ───────────────────────────────────────────────
        annotated = draw_results(frame, results, conf_thresh, show_conf)

        # ── FPS calculation ───────────────────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # ── HUD ───────────────────────────────────────────────────────────
        annotated = draw_hud(annotated, fps, conf_thresh, show_conf)

        # ── Display ───────────────────────────────────────────────────────
        cv2.imshow("Weed Detection — YOLOv8 Segmentation", annotated)

        # ── Keyboard controls ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting ...")
            break

        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = SCREENSHOT_DIR / f"weed_detect_{ts}.jpg"
            cv2.imwrite(str(path), annotated)
            print(f"[INFO] Screenshot saved → {path}")

        elif key == ord("c"):
            show_conf = not show_conf

        elif key in (ord("+"), ord("=")):
            conf_thresh = min(conf_thresh + 0.05, 0.95)
            print(f"[INFO] Confidence threshold ↑ {conf_thresh:.0%}")

        elif key in (ord("-"), ord("_")):
            conf_thresh = max(conf_thresh - 0.05, 0.05)
            print(f"[INFO] Confidence threshold ↓ {conf_thresh:.0%}")

    cap.release()
    cv2.destroyAllWindows()
    if esp32:
        esp32.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
