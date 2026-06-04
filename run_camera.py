#!/usr/bin/env python3
"""
Crop / Weed / Soil Detection — Live Camera
============================================
Runs the trained YOLOv8 segmentation model on a live camera feed.

Classes:
  - Crop  (class 0): Bean plants        — GREEN mask
  - Weed  (class 1): Unwanted plants    — RED mask
  - Soil  (class 2): Bare soil          — BROWN mask

ESP32 LED signals (single-byte commands over Serial):
  'W' → Weed detected   → RED   LED blinks 3×  (highest priority)
  'C' → Crop detected   → GREEN LED blinks 2×
  'S' → Soil only       → BLUE  LED blinks 1×
  'N' → Nothing found   → all LEDs off

Controls:
  q        — quit
  s        — save screenshot
  c        — toggle confidence % display
  + / -    — raise / lower confidence threshold
  m        — toggle mask overlay on/off
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed — LED control disabled.")
    print("       Install with:  pip install pyserial")


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH = (
    Path(__file__).parent
    / "new_plant_weed_soil_instances_segmentaion"
    / "new_model_results"
    / "crop_weed_soil_seg"
    / "weights"
    / "best.pt"
)

CAMERA_INDEX        = 0        # 0 = default webcam; change if you have multiple cameras
CONFIDENCE_THRESHOLD = 0.45   # starting detection threshold
IMG_SIZE            = 640      # must match the training resolution
MAX_FRAME_FAILURES  = 30       # quit after this many consecutive bad frames
SCREENSHOT_DIR      = Path(__file__).parent / "screenshots"

# ── ESP32 Serial ───────────────────────────────────────────────────────────────
# Find your port with:  ls /dev/cu.*   (macOS)  or  ls /dev/tty*  (Linux)
ESP32_PORT          = "/dev/cu.usbserial-140"   # <-- CHANGE to your port
ESP32_BAUD          = 9600
LED_SIGNAL_INTERVAL = 1.0    # minimum seconds between LED commands (avoids flooding)



# ── Class definitions ──────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Crop", 1: "Weed", 2: "Soil"}

# BGR colours for OpenCV masks / boxes
CLASS_COLORS = {
    0: (0,  200,  0),    # Crop  → green
    1: (0,   0, 220),    # Weed  → red
    2: (40, 100, 160),   # Soil  → brown/tan
}

# ESP32 single-byte commands — must match the .ino sketch
ESP32_CMD = {
    "weed": b"W",
    "crop": b"C",
    "soil": b"S",
    "none": b"N",
}

MASK_ALPHA = 0.45   # mask overlay opacity


# ── ESP32 helpers ──────────────────────────────────────────────────────────────

def connect_esp32():
    if not SERIAL_AVAILABLE:
        return None
    try:
        esp = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
        print(f"[INFO] ESP32 connected on {ESP32_PORT} @ {ESP32_BAUD} baud")
        return esp
    except Exception as e:
        print(f"[WARN] Could not connect to ESP32 ({e})")
        print(f"       Check ESP32_PORT in this script.  LED control disabled.")
        return None


def send_led(esp32, cmd: bytes):
    if esp32 is None:
        return
    try:
        esp32.write(cmd)
    except Exception:
        pass


def decide_led_command(detected_classes: set) -> bytes:
    """
    Priority: Weed > Crop > Soil > Nothing.
    Returns the ESP32 command byte to send.
    """
    if 1 in detected_classes:
        return ESP32_CMD["weed"]
    if 0 in detected_classes:
        return ESP32_CMD["crop"]
    if 2 in detected_classes:
        return ESP32_CMD["soil"]
    return ESP32_CMD["none"]


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_results(frame, results, conf_thresh, show_conf=True, show_masks=True):
    """Overlay segmentation masks and labelled bounding boxes on *frame*."""
    overlay = frame.copy()

    for result in results:
        if show_masks and result.masks is not None:
            masks = result.masks.data.cpu().numpy()   # (N, H, W)
            for i, mask in enumerate(masks):
                cls_id = int(result.boxes.cls[i].item())
                conf   = float(result.boxes.conf[i].item())
                if conf < conf_thresh:
                    continue
                color        = CLASS_COLORS.get(cls_id, (255, 255, 255))
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_bool    = mask_resized > 0.5
                overlay[mask_bool] = (
                    np.array(color, dtype=np.float32) * MASK_ALPHA
                    + overlay[mask_bool] * (1 - MASK_ALPHA)
                ).astype(np.uint8)
            frame = overlay.copy()

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf   = float(box.conf.item())
                if conf < conf_thresh:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
                if show_conf:
                    label = f"{label} {conf:.0%}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, fps, conf_thresh, show_conf, show_masks, detected):
    """Heads-up display: top banner + bottom legend + active detections."""
    h, w = frame.shape[:2]

    # Top banner
    banner_h = 36
    snap = frame[:banner_h, :].copy()
    cv2.rectangle(frame, (0, 0), (w, banner_h), (30, 30, 30), -1)
    frame[:banner_h, :] = cv2.addWeighted(frame[:banner_h, :], 0.65, snap, 0.35, 0)

    info = (
        f"FPS {fps:.1f}  |  Conf {conf_thresh:.0%}  |  "
        f"[q] Quit  [s] Save  [c] Conf  [m] Masks  [+/-] Thresh"
    )
    cv2.putText(frame, info, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # Bottom legend
    y = h - 12
    for cls_id in sorted(CLASS_NAMES, reverse=True):
        color = CLASS_COLORS[cls_id]
        name  = CLASS_NAMES[cls_id]
        active = cls_id in detected
        radius = 10 if active else 7
        cv2.circle(frame, (18, y), radius, color, -1)
        if active:
            cv2.circle(frame, (18, y), radius + 2, (255, 255, 255), 1)
        cv2.putText(frame, name, (34, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255) if active else (160, 160, 160),
                    1, cv2.LINE_AA)
        y -= 30

    return frame


def detect_classes(results, conf_thresh) -> set:
    found = set()
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if float(box.conf.item()) >= conf_thresh:
                found.add(int(box.cls.item()))
    return found


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Train the model first (see the notebook), then update MODEL_PATH.")
        sys.exit(1)

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("[INFO] Model loaded.")

    esp32 = connect_esp32()

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index={CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {CAMERA_INDEX} opened — {aw}×{ah}")

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    conf_thresh    = CONFIDENCE_THRESHOLD
    show_conf      = True
    show_masks     = True
    prev_time      = time.time()
    last_led_time  = 0.0
    frame_failures = 0
    detected       = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            frame_failures += 1
            if frame_failures >= MAX_FRAME_FAILURES:
                print("[ERROR] Too many frame failures — exiting.")
                break
            time.sleep(0.05)
            continue
        frame_failures = 0

        # Inference
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=conf_thresh,
            verbose=False,
        )

        detected = detect_classes(results, conf_thresh)

        # LED signal — only W or C, only when detected
        now = time.time()
        if now - last_led_time >= LED_SIGNAL_INTERVAL:
            if 1 in detected:
                send_led(esp32, b'W')
                print("[LED] Weed detected — blinking RED")
                last_led_time = now
            elif 0 in detected:
                send_led(esp32, b'C')
                print("[LED] Crop detected — blinking GREEN")
                last_led_time = now

        # Draw
        annotated = draw_results(frame, results, conf_thresh, show_conf, show_masks)

        fps = 1.0 / max(time.time() - prev_time, 1e-6)
        prev_time = time.time()

        annotated = draw_hud(annotated, fps, conf_thresh, show_conf, show_masks, detected)

        cv2.imshow("Crop / Weed / Soil — YOLOv8 Segmentation", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting.")
            break

        elif key == ord("s"):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = SCREENSHOT_DIR / f"detection_{ts}.jpg"
            cv2.imwrite(str(path), annotated)
            print(f"[INFO] Screenshot → {path}")

        elif key == ord("c"):
            show_conf = not show_conf

        elif key == ord("m"):
            show_masks = not show_masks

        elif key in (ord("+"), ord("=")):
            conf_thresh = min(conf_thresh + 0.05, 0.95)
            print(f"[INFO] Threshold ↑ {conf_thresh:.0%}")

        elif key in (ord("-"), ord("_")):
            conf_thresh = max(conf_thresh - 0.05, 0.05)
            print(f"[INFO] Threshold ↓ {conf_thresh:.0%}")

    cap.release()
    cv2.destroyAllWindows()
    if esp32:
        esp32.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
