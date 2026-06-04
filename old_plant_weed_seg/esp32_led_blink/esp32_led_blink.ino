#define RED_LED   2   // Weed detected
#define GREEN_LED 4   // Crop detected

void setup() {
  Serial.begin(9600);
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  digitalWrite(RED_LED, LOW);
  digitalWrite(GREEN_LED, LOW);
}

void blinkLED(int pin, int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(pin, HIGH);
    delay(delayMs);
    digitalWrite(pin, LOW);
    delay(delayMs);
  }
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'W') {
      blinkLED(RED_LED, 3, 150);    // Weed → red blinks 3x
    } else if (cmd == 'C') {
      blinkLED(GREEN_LED, 2, 200);  // Crop → green blinks 2x
    }
  }
}
