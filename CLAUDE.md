# ESP32 LED Blink Project

## Hardware Setup

### ESP32 Placement on Breadboard
- Left pins: column **b**, rows 1–19
- Right pins: column **i**, rows 1–19
- Column **j** is free for connections

### Key ESP32 Pins (right side, column i)
- i1 = GND
- i7 = GND
- i13 = GPIO4 (Green LED)
- i14 = GPIO2 (Red LED)
- i16 = GND

### Wiring Layout

#### Red LED (GPIO2)
| Step | Component | From | To |
|------|-----------|------|----|
| 1 | Wire (GPIO2 signal) | j14 | h30 |
| 2 | Resistor leg 1 | i30 | — |
| 2 | Resistor leg 2 | i32 | — |
| 3 | Red LED long leg (+) | h32 | — |
| 3 | Red LED short leg (−) | h34 | — |
| 4 | Wire (GND) | j34 | j7 |

#### Green LED (GPIO4)
| Step | Component | From | To |
|------|-----------|------|----|
| 5 | Wire (GPIO4 signal) | j13 | h36 |
| 6 | Resistor leg 1 | i36 | — |
| 6 | Resistor leg 2 | i38 | — |
| 7 | Green LED long leg (+) | h38 | — |
| 7 | Green LED short leg (−) | h40 | — |
| 8 | Wire (GND) | j40 | j16 |

---

## Code

File: `esp32_led_blink/esp32_led_blink.ino`

- Red LED (GPIO2) = Weed detected → blinks 3 times when serial command **W** is sent
- Green LED (GPIO4) = Crop detected → blinks 2 times when serial command **C** is sent
- Baud rate: 9600

---

## Upload Steps (Arduino IDE)

1. Open `esp32_led_blink/esp32_led_blink.ino` in Arduino IDE
2. Tools → Board → **ESP32 Dev Module**
3. Tools → Port → select ESP32 USB port (e.g. `/dev/cu.usbserial-...`)
4. Click Upload (→)
5. Open Serial Monitor, set baud to **9600**
6. Type **W** → Red LED blinks 3x
7. Type **C** → Green LED blinks 2x

## Pending Issue
- ESP32 USB port not showing in Arduino IDE
- Possible fix: try different USB cable (data cable, not charge-only)
- Possible fix: install CP2102 or CH340 driver on Mac and restart
