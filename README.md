# Weed Detection System — YOLOv8 + ESP32 LED Indicator

A precision agriculture system that detects weeds and crops in real time using a YOLOv8 segmentation model and signals the result via an ESP32-controlled LED system.

## Authors

- **Amos Maru**
- **Paul**

---

## Project Overview

This project combines computer vision and embedded hardware to automate weed detection in agricultural fields. A trained YOLOv8 model processes a live camera feed and classifies detected plants as either **crop** or **weed**. The result is communicated to an ESP32 microcontroller which blinks an LED:

- **Red LED** — Weed detected (3 blinks)
- **Green LED** — Crop detected (2 blinks)

---

## System Components

### 1. Machine Learning Model
- **Model**: YOLOv8 Segmentation (`best.pt`)
- **Framework**: Ultralytics YOLOv8
- **Backend**: PyTorch
- **Classes**:
  - `0` — Crop (Bean plants) — shown in Green
  - `1` — Weed — shown in Red
- **Input resolution**: 640×640
- **Training**: 100 epochs, batch size 16, Adam optimizer

### 2. ESP32 LED Hardware
- **Microcontroller**: ESP32 DevKit (38-pin)
- **Red LED**: GPIO2 → signals weed detection
- **Green LED**: GPIO4 → signals crop detection
- **Communication**: USB Serial at 9600 baud
- **Commands**: `W` = weed, `C` = crop

### 3. Camera Script (`run_camera.py`)
- Opens a live webcam feed
- Runs YOLOv8 inference on each frame
- Draws segmentation masks and bounding boxes
- Sends serial commands to ESP32 based on detections
- Controls: `q` quit, `s` screenshot, `c` toggle confidence, `+/-` threshold

---

## ESP32 Wiring

| Component | Hole 1 | Hole 2 |
|-----------|--------|--------|
| Wire (GPIO2 signal) | a5 | h30 |
| Red LED Resistor | i30 | i32 |
| Red LED (+) | h32 | — |
| Red LED (−) | h34 | — |
| Wire (Red GND) | j34 | a13 |
| Wire (GPIO4 signal) | a7 | h36 |
| Green LED Resistor | i36 | i38 |
| Green LED (+) | h38 | — |
| Green LED (−) | h40 | — |
| Wire (Green GND) | j40 | a19 |

ESP32 placement: left pins at column **b**, right pins at column **i**, rows 1–19.

---

## Project Structure

```
final_5th_project/
├── run_camera.py                  # Main script — camera + model + ESP32
├── esp32_led_blink/
│   └── esp32_led_blink.ino        # Arduino code for ESP32
├── model_results/
│   └── weed_segmentation/
│       └── weights/
│           ├── best.pt            # Trained model weights
│           └── best.onnx
├── test image/                    # Test images for the model
│   ├── crop_01.jpeg – crop_05.jpeg   # 5 crop samples
│   └── weed_01.jpg  – weed_05.jpeg  # 5 weed samples
├── traning images/                # Original training data
├── raw images/                    # Raw collected images
├── screenshots/                   # Saved detection screenshots
└── README.md
```

---

## How to Run

### Step 1 — Upload ESP32 code
1. Open `esp32_led_blink/esp32_led_blink.ino` in Arduino IDE
2. Select **Tools → Board → ESP32 Dev Module**
3. Select **Tools → Port → /dev/cu.usbserial-10**
4. Click Upload
5. **Close Serial Monitor** after uploading

### Step 2 — Run the detection script
```bash
python run_camera.py
```

### Step 3 — Test with images
Point the camera at images from `test_images/` to test detections without a live plant.

---

## Dependencies

```bash
pip install ultralytics opencv-python pyserial numpy
```

- Python ≥ 3.8
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- pyserial

---

## Dataset

- **Classes**: Crop (Bean), Weed
- **Format**: YOLO annotation style
- **Source**: Roboflow annotated dataset
- **Split**: Train / Val / Test

---

## License

This project is open-sourced under the MIT License.
