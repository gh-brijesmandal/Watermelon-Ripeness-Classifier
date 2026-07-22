# Watermelon Ripeness Classifier

Detect watermelon ripeness (unripe / ripe / overripe) from acoustic taps.

This repository contains two main pipelines:

- Data capture: `record_tap.py` — controls a solenoid (Arduino) to strike a melon
    and records the audio for each tap into `./audio/`.
- Feature extraction & logging: `main.py` — extracts time/frequency features
    from a single tap audio file, saves helpful diagnostic plots to `./plots/`,
    and appends one row per tap to `Watermelon_Features.xlsx`.

There is also a simple example machine-learning workflow under
`Machine Learning Sample/` that trains and evaluates classifiers from the
feature table and saves a `ripeness_model.joblib` bundle.

**Repository layout**

- `main.py` — feature extraction, plotting, and Excel logging for one WAV
- `record_tap.py` — automated capture using a USB serial Arduino + solenoid
- `audio/` — place or record .wav files here (created at runtime)
- `plots/` — automatically created by `main.py` for diagnostic images
- `Watermelon_Features.xlsx` — generated/updated by `main.py` (single table)
- `Machine Learning Sample/` — `train_model.py`, `evaluate_models.py`, and
    `classifier.py` example that load / save models

Getting started
---------------

1) Create a Python environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install the main packages used here:

```bash
pip install numpy scipy librosa matplotlib pandas sounddevice soundfile pyserial scikit-learn joblib openpyxl
```

2) Recording taps (hardware required)

- Connect an Arduino running the companion `.ino` (in `solenoid_tap/solenoid_tap.ino`) to a USB port.
- Position the solenoid so it will strike the melon consistently and place a microphone that records the impact.
- Run the recorder:

```bash
python record_tap.py --melon-id 3 --taps 5 --gap 2.0
```

- Options:
    - `--melon-id`: identifier used in output filenames (e.g. `melon3_tap1.wav`)
    - `--taps`: number of taps to record in the session
    - `--gap`: seconds between taps
    - `--port`: serial port (auto-detected if omitted)

Recorded WAV files are saved into `./audio/`.

3) Extract features from a WAV file

Place or confirm a file exists in `./audio/` (e.g. `melon3_tap1.wav`) and run:

```bash
python main.py melon3_tap1.wav
```

- This will:
    - Load and bandpass-filter the audio
    - Detect the tap and extract a resonance window
    - Compute time-domain features (peak amplitude, RMS, zero-crossing rate,
        damping coefficient) and frequency-domain features (peak frequency,
        spectral skewness, kurtosis, Fmax from FFT)
    - Save diagnostic plots to `./plots/`
    - Append a new row to `Watermelon_Features.xlsx`

Notes on naming: `main.py` expects the argument to be just the filename
(e.g. `melon3_tap1.wav`) and looks for it in `./audio/`.

Machine learning
----------------

See `Machine Learning Sample/` for training and evaluation scripts.

- Train a classifier from labeled rows in `Watermelon_Features.xlsx`:

```bash
python "Machine Learning Sample/train_model.py"
```

- Useful flags:
    - `--predict-unlabeled`: after training, predict ripeness for unlabeled
        rows that have full features
    - `--min-per-class N`: require at least `N` examples per class

- Evaluate grouping by melon id to avoid data leakage:

```bash
python "Machine Learning Sample/evaluate_models.py"
```

The training script saves a model bundle to `ripeness_model.joblib` which
contains a `pipeline`, `label_encoder`, `feature_columns`, and some metadata.
An example `Machine Learning Sample/classifier.py` demonstrates how to load
that bundle and run predictions.

Feature summary (what `main.py` logs)
------------------------------------

- Time-domain features (from the resonance window):
    - `peak_amplitude` — max absolute amplitude
    - `rms_energy` — RMS energy of the window
    - `zero_crossing_rate` — normalized count of sign changes
    - `damping_coefficient` — slope of the log amplitude envelope (linear fit)

- Frequency-domain features:
    - `peak_frequency_hz` — dominant frequency between 50–500 Hz
    - `spectral_skewness`, `spectral_kurtosis` — shape descriptors of the
        resonance-band spectrum
    - `f_max_hz` — peak frequency found from FFT analysis

Troubleshooting & tips
----------------------

- If `main.py` cannot find the file, ensure `./audio/` exists and the
    filename you pass matches exactly (case-sensitive on some OSes).
- If `record_tap.py` fails to find an Arduino port, set `--port COMX` with
    your port name (Windows) or check Arduino IDE > Tools > Port.
- Recommended microphone setup: keep mic at fixed distance, record in a
    quiet environment, and make the strike position consistent across taps.
- The detector may fail if the tap is too quiet or noisy. Try increasing
    tap strength, reducing background noise, or adjusting `TAP_K` in `main.py`.

Development notes
-----------------

- The pipeline expects mono WAVs sampled at ~48 kHz. `record_tap.py` uses
    `sounddevice`/`soundfile` with `SAMPLE_RATE=48000` which matches the
    `librosa.load()` default behavior used by `main.py`.
- `main.py` writes a single Excel workbook `Watermelon_Features.xlsx`. When
    training, label rows in the `LABEL` column with `unripe`, `ripe`, or
    `overripe` (case-insensitive). The training script will normalize common
    variants (e.g. `over ripe`, `Overripe`, etc.).

Next steps (suggested)
----------------------

- Add a `requirements.txt` for reproducible installs.
- Add unit tests for signal-processing helpers and a small end-to-end
    integration test that runs `main.py` on a synthetic tap file.
- Improve `classifier.py` example (there is a minor formatting bug) and
    provide an inference CLI that accepts raw feature rows.

If you want, I can:

- Add a `requirements.txt` listing the exact packages used now.
- Fix and improve `Machine Learning Sample/classifier.py` so it runs.
- Create a short example WAV and run `main.py` to show the resulting plots.

---
Original quick usage:

```bash
python main.py audio_file_name
python record_tap.py --melon-id 1 --taps 5 --port COM13
```