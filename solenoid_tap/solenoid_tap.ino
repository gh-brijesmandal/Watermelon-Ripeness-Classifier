// solenoid_tap.ino
// Replaces the free-running loop() with a serial-triggered single tap.
// The PC sends a single byte 'T' over USB serial -> Arduino fires the
// solenoid for TAP_DURATION_MS, then releases. One tap per 'T' received.

const int SOLENOID_PIN   = 2;     // same pin you were already using
const int TAP_DURATION_MS = 40;   // how long the solenoid stays energized
                                   // tune this: too long = solenoid "pushes through"
                                   // and muffles the melon's natural ring,
                                   // too short = it may not strike with enough force.
                                   // 30-60ms is a reasonable starting range.

void setup() {
  pinMode(SOLENOID_PIN, OUTPUT);
  digitalWrite(SOLENOID_PIN, LOW);   // make sure it starts retracted
  Serial.begin(9600);
  Serial.setTimeout(10);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'T') {
      digitalWrite(SOLENOID_PIN, HIGH);   // push
      delay(TAP_DURATION_MS);
      digitalWrite(SOLENOID_PIN, LOW);    // pull back immediately
      Serial.println("TAPPED");           // ack back to the PC (optional, useful for debugging)
    }
  }
}
