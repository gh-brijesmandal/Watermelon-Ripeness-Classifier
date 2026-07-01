import os                                          # for creating the plots folder and building file paths
import sys                                          # for reading command-line arguments (the filename)
import re                                           # for parsing melon_id / tap_id out of the filename
from datetime import datetime                       # timestamp for each Excel row
import numpy as np                                  # numeric arrays, thresholds, math
import librosa                                       # audio loading + RMS energy framing
import matplotlib.pyplot as plt                      # plotting and saving figures
from scipy.signal import butter, sosfiltfilt, find_peaks, hilbert   # filter design + peak finding + envelope
import librosa.display                              # for spectrogram matrix
from scipy.stats import skew, kurtosis
import pandas as pd                                 # for reading/writing the Excel log

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python main.py <filename.wav>\n"
        "Example: python main.py melon1_tap1.wav"
    )

INPUT_FILENAME = sys.argv[1]                        # e.g. "melon1_tap1.wav"
AUDIO_PATH = "./audio/" + INPUT_FILENAME
PLOTS_DIR = "plots"
EXCEL_PATH = "Watermelon_Features.xlsx"             # single growing results file, created if missing

LOW_HZ = 80                         # bandpass: frequencies below this are removed (rumble/DC drift)
HIGH_HZ = 2000                      # bandpass: frequencies above this are removed (hiss/high-freq noise)
FILTER_ORDER = 4                    # Butterworth filter steepness (4 = gentle, safe default)
TAP_K = 8.0                         # tap detector: how many MADs above background energy counts as a tap
TAP_IGNORE_START_SEC = 0.03         # tap detector: ignore this much audio at the very start (edge artifacts)

# RESONANCE_GAP_SEC lowered from 0.03 -> 0.008 (8ms). We only need to skip the
# impact click itself, not the early resonance decay. Skipping too much (30ms)
# was throwing away the highest-amplitude part of the signal, which matters
# for an accurate damping-coefficient fit.
RESONANCE_GAP_SEC = 0.008
RESONANCE_WINDOW_SEC = 0.3          # resonance window: total length of the extracted resonance window
RESONANCE_SEARCH_SEC = 0.5          # resonance window: how far past the gap to search for the resonance peak


# ---------------------------------------------------------------------------
# Filename parsing (for Excel logging)
# ---------------------------------------------------------------------------
def parse_melon_tap_ids(filename):
    """
    Extracts melon_id and tap_id from a filename like 'melon1_tap1.wav'.
    Falls back to storing the raw filename if the pattern doesn't match,
    so the script never crashes on an unexpected name -- it just logs
    less-structured metadata instead.
    """
    match = re.match(r"melon(\w+)_tap(\d+)", filename, re.IGNORECASE)
    if match:
        melon_id, tap_id = match.group(1), match.group(2)
    else:
        print(f"Warning: filename '{filename}' didn't match 'melonX_tapY' pattern. "
              f"Logging raw filename instead of parsed IDs.")
        melon_id, tap_id = filename, ""
    return melon_id, tap_id


# ---------------------------------------------------------------------------
# audio extraction and filtering pipeline
# ---------------------------------------------------------------------------
def load_audio(path, sr=None, mono=True):
    """Load an audio file from disk and return (audio_samples, sample_rate)."""
    audio, sr = librosa.load(path, sr=sr, mono=mono)
    return audio, sr


def plot_audio(audio, sr, title, save_path):
    """Plot a waveform against a real time axis (seconds) and save it to disk."""
    t = np.arange(len(audio)) / sr

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, audio, color="red", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def bandpass_filter(audio, sr, low_hz=LOW_HZ, high_hz=HIGH_HZ, order=FILTER_ORDER):
    """
    Bandpass filter the audio with a zero-phase Butterworth filter.
    Removes rumble/DC drift below low_hz and hiss/noise above high_hz,
    keeping mainly the tap + watermelon resonance frequency range.
    """
    nyquist = sr / 2
    low = low_hz / nyquist
    high = high_hz / nyquist

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, audio)

    out_dtype = audio.dtype if np.issubdtype(audio.dtype, np.floating) else np.float32
    return filtered.astype(out_dtype)


def detect_tap(raw_audio, sr, frame_length=128, hop_length=32, k=TAP_K,
               ignore_start_sec=TAP_IGNORE_START_SEC, bg_window_sec=0.03):
    """
    Detect the first tap onset in the RAW (unfiltered) audio.
    Uses frame-wise RMS energy + a robust hybrid threshold to find
    the approximate tap location, then refines to the exact sample.
    """
    ignore_samples = int(ignore_start_sec * sr)

    energy = librosa.feature.rms(y=raw_audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy_times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)

    ignore_frames = int(ignore_samples / hop_length) + 1
    valid_energy = energy[ignore_frames:]
    valid_times = energy_times[ignore_frames:]

    if len(valid_energy) == 0:
        raise ValueError("ignore_start_sec is longer than the audio itself.")

    median = np.median(valid_energy)
    mad = np.median(np.abs(valid_energy - median)) + 1e-12

    base_threshold = median + k * mad
    peak_threshold = 0.30 * np.max(valid_energy)
    threshold = max(base_threshold, peak_threshold)

    above = np.where(valid_energy > threshold)[0]
    if len(above) == 0:
        raise ValueError("No tap detected above threshold. Try lowering k or check the input audio.")

    coarse_sample = int(valid_times[above[0]] * sr)

    bg_win = int(bg_window_sec * sr)
    bg_end = max(ignore_samples, coarse_sample - frame_length)
    bg_start = max(ignore_samples, bg_end - bg_win)
    bg = np.abs(raw_audio[bg_start:bg_end])

    if len(bg) < 10:
        tap_sample = coarse_sample
    else:
        bg_median = np.median(bg)
        bg_mad = np.median(np.abs(bg - bg_median)) + 1e-12

        sample_base_threshold = bg_median + k * bg_mad
        sample_peak_threshold = 0.25 * np.max(np.abs(raw_audio[bg_end:min(len(raw_audio), coarse_sample + hop_length)]))
        sample_threshold = max(sample_base_threshold, sample_peak_threshold)

        search_start = max(ignore_samples, coarse_sample - frame_length)
        search_end = min(len(raw_audio), coarse_sample + hop_length)
        local = np.abs(raw_audio[search_start:search_end])
        above_sample = np.where(local > sample_threshold)[0]

        tap_sample = (search_start + above_sample[0]) if len(above_sample) else coarse_sample

    tap_time = tap_sample / sr
    return tap_sample, tap_time, energy, energy_times


def plot_tap_detection(audio, sr, tap_sample, energy, energy_times, title, save_path):
    """Plot the waveform with the detected tap marked, plus the energy curve below it, and save to disk."""
    t = np.arange(len(audio)) / sr
    tap_time = tap_sample / sr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(t, audio, color="red", linewidth=0.5)
    ax1.axvline(tap_time, color="blue", linestyle="--", linewidth=1.5,
                label=f"Detected tap @ {tap_time:.3f}s")
    ax1.set_title(title)
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.5)

    ax2.plot(energy_times, energy, color="black", linewidth=1)
    ax2.axvline(tap_time, color="blue", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Frame RMS energy")
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def extract_resonance_window(audio, sr, tap_sample, gap=RESONANCE_GAP_SEC,
                              window_duration=RESONANCE_WINDOW_SEC, search_after=RESONANCE_SEARCH_SEC):
    """
    Extract the watermelon's resonance window following the tap.
    Skips a short gap after the tap (clears the click itself), searches
    the next stretch of audio for the loudest point (the resonance peak),
    then extracts a fixed-length window starting just before that peak.
    """
    gap_samples = int(gap * sr)
    search_samples = int(search_after * sr)
    window_samples = int(window_duration * sr)

    search_start = tap_sample + gap_samples
    search_end = min(search_start + search_samples, len(audio))

    if search_start >= len(audio):
        raise ValueError("Tap occurs too close to the end of the audio; no room to search for resonance.")

    search_region = audio[search_start:search_end]

    frame_len = max(int(0.005 * sr), 32)
    hop = max(frame_len // 4, 1)

    # center=False is the key fix here: with the default center=True, librosa
    # zero-pads the signal so frame i is CENTERED at sample i*hop, not
    # STARTING at i*hop. Since we manually convert peak_frame -> sample offset
    # via "peak_frame * hop", we need center=False so that math is actually
    # correct and the resonance window lands where we think it does.
    local_energy = librosa.feature.rms(y=search_region, frame_length=frame_len,
                                        hop_length=hop, center=False)[0]

    if len(local_energy) == 0:
        peak_offset = 0
    else:
        peak_frame = np.argmax(local_energy)
        peak_offset = peak_frame * hop

    resonance_peak_sample = search_start + peak_offset

    pre_peak = int(0.02 * sr)
    start_sample = max(resonance_peak_sample - pre_peak, search_start)
    end_sample = min(start_sample + window_samples, len(audio))

    resonance_audio = audio[start_sample:end_sample]
    return resonance_audio, start_sample, end_sample


def plot_resonance_window(audio, sr, tap_sample, start_sample, end_sample, title, save_path):
    """
    Plot the FULL waveform with the tap and resonance window highlighted.

    IMPORTANT: this must be called with the full filtered signal, not the
    short resonance_audio slice. tap_sample/start_sample/end_sample are all
    absolute sample indices measured from the start of the full recording --
    plotting them against a resonance_audio-length time axis (only ~0.3s)
    put the markers outside the visible range, which was the root cause of
    the "wrong images" you were seeing.
    """
    t = np.arange(len(audio)) / sr
    tap_time = tap_sample / sr
    start_time = start_sample / sr
    end_time = end_sample / sr

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, audio, color="red", linewidth=0.5)
    ax.axvline(tap_time, color="blue", linestyle="--", linewidth=1.5,
               label=f"Tap @ {tap_time:.3f}s")
    ax.axvspan(start_time, end_time, color="green", alpha=0.25,
               label=f"Resonance window [{start_time:.3f}s, {end_time:.3f}s]")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# feature extraction pipeline
# ---------------------------------------------------------------------------
def extract_time_domain_features(audio, sample_rate):
    """
    Extracts features directly from the raw time-domain voltage waveform.
    Returns: dict of peak_amplitude, rms_energy, zero_crossing_rate, damping_coefficient.
    """
    time_features = {}

    time_features['peak_amplitude'] = float(np.max(np.abs(audio)))
    time_features['rms_energy'] = float(np.sqrt(np.mean(audio ** 2)))

    zero_crossings = np.where(np.diff(np.sign(audio)))[0]
    time_features['zero_crossing_rate'] = float(len(zero_crossings) / len(audio))

    analytic_signal = hilbert(audio)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = np.where(amplitude_envelope == 0, 1e-8, amplitude_envelope)
    log_envelope = np.log(amplitude_envelope)
    time_axis = np.arange(len(audio)) / sample_rate
    slope, _ = np.polyfit(time_axis, log_envelope, 1)

    time_features['damping_coefficient'] = float(slope)

    return time_features


def extract_frequency_domain_features(audio_data, sample_rate):
    """
    Converts audio to the frequency domain via FFT and extracts spectral features
    within the watermelon resonance band (50Hz - 500Hz).

    Mass-dependent stiffness_index has been removed since mass isn't
    currently measurable -- this returns pure acoustic features only.
    """
    audio = np.array(audio_data, dtype=float)
    audio = audio - np.mean(audio)

    freq_features = {}

    fft_vals = np.abs(np.fft.rfft(audio))
    fft_freqs = np.fft.rfftfreq(len(audio), d=1.0/sample_rate)

    band_mask = (fft_freqs >= 50) & (fft_freqs <= 500)
    filtered_freqs = fft_freqs[band_mask]
    filtered_amps = fft_vals[band_mask]

    if len(filtered_amps) > 0 and np.sum(filtered_amps) > 0:
        peak_idx = np.argmax(filtered_amps)
        peak_freq = filtered_freqs[peak_idx]
        freq_features['peak_frequency_hz'] = float(peak_freq)

        freq_features['spectral_skewness'] = float(skew(filtered_amps))
        freq_features['spectral_kurtosis'] = float(kurtosis(filtered_amps))
    else:
        freq_features['peak_frequency_hz'] = 0.0
        freq_features['spectral_skewness'] = 0.0
        freq_features['spectral_kurtosis'] = 0.0

    return freq_features


def analyze_resonance_fft(resonance_audio, sr, title, save_path, low_hz=LOW_HZ, high_hz=HIGH_HZ):
    """
    Compute the FFT of the resonance window, find the dominant frequency (Fmax),
    locate the top 5 local peaks, mark them with an 'x', and save the spectrum plot.
    """
    n = len(resonance_audio)
    if n == 0:
        raise ValueError("Resonance audio window is empty. Cannot compute FFT.")

    fft_vals = np.fft.fft(resonance_audio)
    freqs = np.fft.fftfreq(n, d=1/sr)

    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitude = np.abs(fft_vals)[pos_mask]
    magnitude = magnitude / n

    max_idx = np.argmax(magnitude)
    f_max = freqs[max_idx]

    peaks, _ = find_peaks(magnitude, distance=5)

    if len(peaks) > 0:
        top_peak_indices = peaks[np.argsort(magnitude[peaks])[-5:]]
    else:
        top_peak_indices = np.argsort(magnitude)[-5:]

    top_freqs = freqs[top_peak_indices]
    top_mags = magnitude[top_peak_indices]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(freqs, magnitude, color="purple", linewidth=1, label="Frequency Spectrum")
    ax.scatter(top_freqs, top_mags, color="red", marker="x", s=100, zorder=5,
               label=f"Top {len(top_freqs)} Peaks")
    ax.axvline(f_max, color="green", linestyle="--", linewidth=1.5,
               label=f"Fmax = {f_max:.1f} Hz")

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (Normalized)")
    ax.set_xlim(max(0, low_hz - 100), high_hz + 100)
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved FFT Plot: {save_path} | Fmax = {f_max:.2f} Hz")
    return f_max


# ---------------------------------------------------------------------------
# Excel logging
# ---------------------------------------------------------------------------
def save_to_excel(melon_id, tap_id, source_filename, time_features, freq_features, f_max):
    """
    Appends one row of results to EXCEL_PATH. Creates the file with headers
    on the first run; reads-and-rewrites on every subsequent run so results
    from every past run stay in the same growing table.
    """
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "melon_id": melon_id,
        "tap_id": tap_id,
        "source_file": source_filename,
        **time_features,
        **freq_features,
        "f_max_hz": f_max,
    }

    new_row_df = pd.DataFrame([row])

    if os.path.exists(EXCEL_PATH):
        existing_df = pd.read_excel(EXCEL_PATH)
        combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        combined_df = new_row_df

    combined_df.to_excel(EXCEL_PATH, index=False)
    print(f"Logged results to {EXCEL_PATH} (melon {melon_id}, tap {tap_id})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(AUDIO_PATH):
        raise FileNotFoundError(
            f"Could not find '{AUDIO_PATH}'. Make sure the file exists in ./audio/ "
            f"and you passed the correct filename as an argument."
        )

    os.makedirs(PLOTS_DIR, exist_ok=True)

    melon_id, tap_id = parse_melon_tap_ids(INPUT_FILENAME)

    audio, sr = load_audio(AUDIO_PATH, sr=None)
    print(f"Loaded '{AUDIO_PATH}': sr={sr} Hz, duration={len(audio)/sr:.3f}s")
    plot_audio(audio, sr, "Raw Audio", os.path.join(PLOTS_DIR, "1. raw_audio.png"))

    filtered = bandpass_filter(audio, sr)
    plot_audio(filtered, sr, "Bandpass Filtered Audio",
               os.path.join(PLOTS_DIR, "2. bandpass_filtered_audio.png"))

    tap_sample, tap_time, energy, energy_times = detect_tap(filtered, sr)
    print(f"Detected tap at t = {tap_time:.4f}s")
    plot_tap_detection(filtered, sr, tap_sample, energy, energy_times,
                        "Tap Detection", os.path.join(PLOTS_DIR, "3. tap_detection.png"))

    resonance_audio, start_sample, end_sample = extract_resonance_window(filtered, sr, tap_sample)
    print(f"Resonance window: {start_sample/sr:.4f}s -> {end_sample/sr:.4f}s "
          f"({(end_sample-start_sample)/sr:.4f}s long)")

    # FIX: pass the full `filtered` signal here, not `resonance_audio`.
    # tap_sample/start_sample/end_sample are absolute indices into the full
    # recording -- plotting them against resonance_audio's own short time
    # axis was the bug producing your broken images.
    plot_resonance_window(filtered, sr, tap_sample, start_sample, end_sample,
                           "Resonance Window Extraction",
                           os.path.join(PLOTS_DIR, "4. resonance_window.png"))

    time_domain_features = extract_time_domain_features(resonance_audio, sample_rate=sr)
    print(time_domain_features)

    frequency_domain_features = extract_frequency_domain_features(resonance_audio, sr)
    print(frequency_domain_features)

    f_max = analyze_resonance_fft(resonance_audio, sr, title="Frequency Distribution",
                                   save_path=os.path.join(PLOTS_DIR, "5. FFT Graph.png"))
    print("Maximum Frequency from this audio file is: ", f_max)

    plot_audio(resonance_audio, sr, "Resonance Audio Graph",
               os.path.join(PLOTS_DIR, "6. resonance_audio.png"))
    print(f"\nAll plots saved to ./{PLOTS_DIR}/")

    save_to_excel(melon_id, tap_id, INPUT_FILENAME,
                  time_domain_features, frequency_domain_features, f_max)


if __name__ == "__main__":
    main()