"""
record_tap.py

Automates the full capture cycle for one (or many) watermelon taps:

    1. Open the mic and start recording into memory (non-blocking).
    2. Wait a short "settle" window so the recording buffer has clean
       background audio before the strike (your detect_tap() function
       relies on this for its background-noise estimate).
    3. Send a single 'T' byte to the Arduino -> solenoid pushes and
       immediately retracts, striking the melon once.
    4. Let the recording continue long enough to capture the full
       resonance decay.
    5. Stop, save to disk as a .wav file in ./audio/, named so you can
       group recordings by melon for cross-validation later
       (melon{ID}_tap{N}.wav).

Dependencies (install once):
    pip install sounddevice soundfile pyserial

Run:
    python record_tap.py --melon-id 3 --taps 5
"""

import argparse
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------------
# Config -- adjust these to match your hardware
# ---------------------------------------------------------------------------
SERIAL_PORT = None          # e.g. "COM5" on Windows, "/dev/ttyACM0" on Linux/Mac.
                             # Leave as None to auto-detect the first Arduino-like port.
BAUD_RATE = 9600            # must match Serial.begin() in the .ino file

SAMPLE_RATE = 48000         # Hz, matches what librosa.load() will read back
CHANNELS = 1                # mono; your pipeline assumes a 1D audio array

PRE_TAP_SETTLE_SEC = 0.4    # silence captured before the tap (background noise estimate)
POST_TAP_CAPTURE_SEC = 1.5  # how long after the tap to keep recording (resonance decay)
TOTAL_DURATION_SEC = PRE_TAP_SETTLE_SEC + POST_TAP_CAPTURE_SEC

AUDIO_DIR = "./audio"


# ---------------------------------------------------------------------------
# Arduino connection
# ---------------------------------------------------------------------------
def find_arduino_port():
    """Best-effort auto-detect: look for a port whose description mentions
    'Arduino' or a common USB-serial chip name. Falls back to None."""
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb serial" in desc:
            return p.device
    return None


def connect_arduino(port=None, baud=BAUD_RATE):
    port = port or find_arduino_port()
    if port is None:
        raise RuntimeError(
            "Could not find an Arduino serial port automatically. "
            "Set SERIAL_PORT at the top of this file to the correct port "
            "(check Arduino IDE > Tools > Port for the exact name)."
        )
    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(2.0)  # Arduino resets when the serial connection opens;
                      # give it time to finish booting before sending commands
    print(f"Connected to Arduino on {port}")
    return ser


def trigger_tap(ser):
    """Send the single-byte command that fires the solenoid once."""
    ser.reset_input_buffer()
    ser.write(b'T')
    # Optional: confirm the Arduino actually fired
    ack = ser.readline().decode(errors="ignore").strip()
    if ack != "TAPPED":
        print(f"  warning: no/unexpected ack from Arduino (got: {ack!r})")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def record_one_tap(ser, output_path):
    """
    Records audio while triggering exactly one solenoid tap partway through.
    sd.rec() is non-blocking, so we can start it, sleep for the settle
    window, fire the tap, then sd.wait() for the rest of the buffer to fill.
    """
    n_samples = int(TOTAL_DURATION_SEC * SAMPLE_RATE)

    recording = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")

    time.sleep(PRE_TAP_SETTLE_SEC)
    tap_time = time.time()
    trigger_tap(ser)

    sd.wait()  # blocks until the full n_samples buffer has been recorded

    audio = recording[:, 0] if recording.ndim > 1 else recording
    sf.write(output_path, audio, SAMPLE_RATE)

    approx_tap_offset = PRE_TAP_SETTLE_SEC  # seconds into the file where the tap should land
    print(f"  saved: {output_path}  (tap fired ~{approx_tap_offset:.2f}s into the clip)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Record watermelon taps for the ripeness pipeline.")
    parser.add_argument("--melon-id", type=str, required=True,
                         help="Identifier for this melon, e.g. 3 or A. Used in filenames so you "
                              "can group-by-melon during cross-validation later.")
    parser.add_argument("--taps", type=int, default=1,
                         help="Number of taps to record in this session (default: 1).")
    parser.add_argument("--gap", type=float, default=2.0,
                         help="Seconds to pause between taps so you can reposition the mic/melon "
                              "if needed (default: 2.0).")
    parser.add_argument("--port", type=str, default=SERIAL_PORT,
                         help="Arduino serial port. Auto-detected if not given.")
    args = parser.parse_args()

    os.makedirs(AUDIO_DIR, exist_ok=True)

    ser = connect_arduino(args.port)

    try:
        for i in range(1, args.taps + 1):
            input(f"\nTap {i}/{args.taps} for melon {args.melon_id} -- "
                  f"position the mic, then press Enter to record...")
            filename = f"melon{args.melon_id}_tap{i}.wav"
            output_path = os.path.join(AUDIO_DIR, filename)
            record_one_tap(ser, output_path)

            if i < args.taps:
                time.sleep(args.gap)
    finally:
        ser.close()

    print(f"\nDone. {args.taps} recording(s) saved to {AUDIO_DIR}/")
    print("Point AUDIO_PATH in your feature-extraction script at one of these files to process it.")


if __name__ == "__main__":
    main()