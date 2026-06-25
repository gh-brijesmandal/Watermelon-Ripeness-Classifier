import os                                          # for creating the plots folder and building file paths
import numpy as np                                 # numeric arrays, thresholds, math
import librosa                                      # audio loading + RMS energy framing
import matplotlib.pyplot as plt                     # plotting and saving figures
from scipy.signal import butter, sosfiltfilt, find_peaks        # Butterworth bandpass filter design + zero-phase application
import librosa.display                             # for spectrogram matrix


AUDIO_PATH = "./audio/sample_2.wav"   
PLOTS_DIR = "plots"                
LOW_HZ = 80                         # bandpass: frequencies below this are removed (rumble/DC drift)
HIGH_HZ = 2000                      # bandpass: frequencies above this are removed (hiss/high-freq noise)
FILTER_ORDER = 4                    # Butterworth filter steepness (4 = gentle, safe default)
TAP_K = 8.0                         # tap detector: how many MADs above background energy counts as a tap
TAP_IGNORE_START_SEC = 0.02         # tap detector: ignore this much audio at the very start (edge artifacts)
RESONANCE_GAP_SEC = 0.02            # resonance window: skip this much time right after the tap (clears the click)
RESONANCE_WINDOW_SEC = 0.3          # resonance window: total length of the extracted resonance window
RESONANCE_SEARCH_SEC = 0.5          # resonance window: how far past the gap to search for the resonance peak

# audio extraction and filtering pipeline
def load_audio(path, sr=None, mono=True):
    """Load an audio file from disk and return (audio_samples, sample_rate)."""
    audio, sr = librosa.load(path, sr=sr, mono=mono)   # librosa reads the file and resamples if sr is given
    return audio, sr                                    # hand back both the waveform and the rate used
def plot_audio(audio, sr, title, save_path):
    """Plot a waveform against a real time axis (seconds) and save it to disk."""
    t = np.arange(len(audio)) / sr           # build a time axis: sample index / sample rate = seconds

    fig, ax = plt.subplots(figsize=(11, 5))  # new figure + single axis to draw on
    ax.plot(t, audio, color="red", linewidth=0.5)  # plot waveform in red, thin line (matches your reference plot)
    ax.set_title(title)                      # label the plot so saved files are self-explanatory
    ax.set_xlabel("Time")                    # x-axis = time in seconds
    ax.set_ylabel("Amplitude")               # y-axis = waveform amplitude
    ax.grid(True, alpha=0.5)                 # light grid for easier visual reading

    plt.tight_layout()                       # remove excess whitespace around the plot
    plt.savefig(save_path, dpi=150)          # write the figure to disk as an image file
    plt.close(fig)                           # close the figure to free memory (important in batch scripts)
    print(f"Saved: {save_path}")             # confirm to the user where the file went
def bandpass_filter(audio, sr, low_hz=LOW_HZ, high_hz=HIGH_HZ, order=FILTER_ORDER):
    """
    Bandpass filter the audio with a zero-phase Butterworth filter.
    Removes rumble/DC drift below low_hz and hiss/noise above high_hz,
    keeping mainly the tap + watermelon resonance frequency range.
    """
    nyquist = sr / 2                                        # Nyquist frequency = half the sample rate
    low = low_hz / nyquist                                  # normalize cutoff to [0, 1] range scipy expects
    high = high_hz / nyquist                                # same normalization for the upper cutoff

    sos = butter(order, [low, high], btype="bandpass", output="sos")  # design filter as stable second-order sections
    filtered = sosfiltfilt(sos, audio)                       # apply filter forward+backward -> zero phase shift (no time delay)

    # cast back to a safe float32 dtype matching audio's type, since sosfiltfilt returns float64
    out_dtype = audio.dtype if np.issubdtype(audio.dtype, np.floating) else np.float32
    return filtered.astype(out_dtype)                        # return filtered samples, same shape as input
def detect_tap(raw_audio, sr, frame_length=128, hop_length=32, k=TAP_K,
               ignore_start_sec=TAP_IGNORE_START_SEC, bg_window_sec=0.03):
    """
    Detect the first tap onset in the RAW (unfiltered) audio.
    Uses frame-wise RMS energy + a robust hybrid threshold to find
    the approximate tap location, then refines to the exact sample.
    """
    ignore_samples = int(ignore_start_sec * sr)   # convert the "ignore window" from seconds to samples

    # compute short-time energy: RMS value inside each frame, sliding by hop_length each step
    energy = librosa.feature.rms(y=raw_audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy_times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)  # frame index -> seconds

    ignore_frames = int(ignore_samples / hop_length) + 1   # how many energy frames fall inside the ignored start window
    valid_energy = energy[ignore_frames:]                  # drop those frames so edge artifacts can't trigger detection
    valid_times = energy_times[ignore_frames:]              # keep matching timestamps for the trimmed energy array

    if len(valid_energy) == 0:                              # safety check: ignore window swallowed the whole signal
        raise ValueError("ignore_start_sec is longer than the audio itself.")

    median = np.median(valid_energy)                        # robust "typical" background energy level
    mad = np.median(np.abs(valid_energy - median)) + 1e-12   # median absolute deviation: robust spread measure (+eps avoids 0)
    
    # --- THE FIX: HYBRID THRESHOLD ---
    # Calculates the statistical threshold, but forces it to be at least 30% of the highest peak.
    # This completely immunizes the detector against background noise "grass".
    base_threshold = median + k * mad
    peak_threshold = 0.30 * np.max(valid_energy)
    threshold = max(base_threshold, peak_threshold)

    above = np.where(valid_energy > threshold)[0]            # indices of all frames that cross the threshold
    if len(above) == 0:                                       # no frame ever got loud enough
        raise ValueError(f"No tap detected above threshold. Try lowering k or check the input audio.")

    coarse_sample = int(valid_times[above[0]] * sr)           # first crossing frame's time, converted to a sample index

    # --- sample-level refinement: pin down the exact first sample, not just the frame ---
    bg_win = int(bg_window_sec * sr)                          # length (in samples) of the "clean background" window to inspect
    bg_end = max(ignore_samples, coarse_sample - frame_length)   # background window ends just before the coarse hit
    bg_start = max(ignore_samples, bg_end - bg_win)            # background window starts bg_win samples earlier
    bg = np.abs(raw_audio[bg_start:bg_end])                    # absolute amplitude of that local background region

    if len(bg) < 10:                                            # not enough background samples to estimate noise reliably
        tap_sample = coarse_sample                              # fall back to the coarse frame-based estimate
    else:
        bg_median = np.median(bg)                               # robust background amplitude level
        bg_mad = np.median(np.abs(bg - bg_median)) + 1e-12      # robust spread of the background amplitude
        
        # Apply a similar peak-safe constraint to the local sample threshold refinement
        sample_base_threshold = bg_median + k * bg_mad
        sample_peak_threshold = 0.25 * np.max(np.abs(raw_audio[bg_end:min(len(raw_audio), coarse_sample + hop_length)]))
        sample_threshold = max(sample_base_threshold, sample_peak_threshold)

        search_start = max(ignore_samples, coarse_sample - frame_length)  # widen search slightly before the coarse hit
        search_end = min(len(raw_audio), coarse_sample + hop_length)       # and slightly after, to catch the true edge
        local = np.abs(raw_audio[search_start:search_end])       # absolute amplitude across that narrow search window
        above_sample = np.where(local > sample_threshold)[0]      # first sample(s) exceeding the local threshold

        # use the first sample that crosses, or fall back to coarse estimate if none found
        tap_sample = (search_start + above_sample[0]) if len(above_sample) else coarse_sample

    tap_time = tap_sample / sr                                   # convert final sample index to seconds
    return tap_sample, tap_time, energy, energy_times            # return detection + energy curve (for plotting)
def plot_tap_detection(audio, sr, tap_sample, energy, energy_times, title, save_path):
    """Plot the waveform with the detected tap marked, plus the energy curve below it, and save to disk."""
    t = np.arange(len(audio)) / sr        # time axis for the waveform plot
    tap_time = tap_sample / sr            # convert detected tap sample to seconds for plotting

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)  # two stacked plots sharing the x-axis

    ax1.plot(t, audio, color="red", linewidth=0.5)               # top panel: raw/normalized waveform
    ax1.axvline(tap_time, color="blue", linestyle="--", linewidth=1.5,
                label=f"Detected tap @ {tap_time:.3f}s")          # vertical line marking the detected tap
    ax1.set_title(title)                                          # title for context
    ax1.set_ylabel("Amplitude")                                   # y-axis label for waveform panel
    ax1.legend(loc="upper right")                                 # show the tap-time label
    ax1.grid(True, alpha=0.5)                                     # light grid

    ax2.plot(energy_times, energy, color="black", linewidth=1)    # bottom panel: frame energy curve used for detection
    ax2.axvline(tap_time, color="blue", linestyle="--", linewidth=1.5)  # same tap marker for visual alignment
    ax2.set_xlabel("Time")                                        # x-axis label (shared, but set here for clarity)
    ax2.set_ylabel("Frame RMS energy")                            # y-axis label for energy panel
    ax2.grid(True, alpha=0.5)                                     # light grid

    plt.tight_layout()                 # tidy spacing between the two panels
    plt.savefig(save_path, dpi=150)    # save the combined figure to disk
    plt.close(fig)                     # free memory
    print(f"Saved: {save_path}")       # confirm output location
def extract_resonance_window(audio, sr, tap_sample, gap=RESONANCE_GAP_SEC,
                              window_duration=RESONANCE_WINDOW_SEC, search_after=RESONANCE_SEARCH_SEC):
    """
    Extract the watermelon's resonance window following the tap.
    Skips a short gap after the tap (clears the click itself), searches
    the next stretch of audio for the loudest point (the resonance peak),
    then extracts a fixed-length window starting just before that peak.
    """
    gap_samples = int(gap * sr)                  # convert the post-tap gap from seconds to samples
    search_samples = int(search_after * sr)      # convert the search duration from seconds to samples
    window_samples = int(window_duration * sr)   # convert the desired resonance window length to samples

    search_start = tap_sample + gap_samples                     # start searching just after the gap following the tap
    search_end = min(search_start + search_samples, len(audio))  # don't search past the end of the audio

    if search_start >= len(audio):                               # tap was too close to the end of the clip
        raise ValueError("Tap occurs too close to the end of the audio; no room to search for resonance.")

    search_region = audio[search_start:search_end]               # slice out just the region we'll search in

    # use short frames to find the resonance's energy peak (more stable than a single loudest sample)
    frame_len = max(int(0.005 * sr), 32)                          # ~5ms frames, at least 32 samples long
    hop = max(frame_len // 4, 1)                                  # hop = quarter of frame length, at least 1 sample
    local_energy = librosa.feature.rms(y=search_region, frame_length=frame_len, hop_length=hop)[0]  # frame-wise energy

    if len(local_energy) == 0:               # edge case: search region too short to produce any frames
        peak_offset = 0                       # default to the very start of the search region
    else:
        peak_frame = np.argmax(local_energy)  # index of the loudest frame within the search region
        peak_offset = peak_frame * hop         # convert that frame index back to a sample offset

    resonance_peak_sample = search_start + peak_offset   # absolute sample index of the resonance's loudest point

    pre_peak = int(0.02 * sr)                              # back up 20ms before the peak to capture the resonance's attack
    start_sample = max(resonance_peak_sample - pre_peak, search_start)  # clamp so we don't go before the search region
    end_sample = min(start_sample + window_samples, len(audio))          # clamp so we don't run past the end of the audio

    resonance_audio = audio[start_sample:end_sample]   # slice out the final resonance window
    return resonance_audio, start_sample, end_sample     # return the audio plus its sample-index bounds
def plot_resonance_window(audio, sr, tap_sample, start_sample, end_sample, title, save_path):
    """Plot the full waveform with the tap and resonance window highlighted, and save to disk."""
    t = np.arange(len(audio)) / sr           # time axis for the full waveform
    tap_time = tap_sample / sr               # tap time in seconds, for the vertical marker line
    start_time = start_sample / sr           # resonance window start time in seconds
    end_time = end_sample / sr               # resonance window end time in seconds

    fig, ax = plt.subplots(figsize=(11, 5))                      # single-panel figure
    ax.plot(t, audio, color="red", linewidth=0.5)                # full waveform in red
    ax.axvline(tap_time, color="blue", linestyle="--", linewidth=1.5,
               label=f"Tap @ {tap_time:.3f}s")                    # mark the tap location
    ax.axvspan(start_time, end_time, color="green", alpha=0.25,
               label=f"Resonance window [{start_time:.3f}s, {end_time:.3f}s]")  # shade the resonance window
    ax.set_title(title)               # plot title
    ax.set_xlabel("Time")             # x-axis label
    ax.set_ylabel("Amplitude")        # y-axis label
    ax.legend(loc="upper right")      # show tap + window labels
    ax.grid(True, alpha=0.5)          # light grid

    plt.tight_layout()                # tidy spacing
    plt.savefig(save_path, dpi=150)   # save figure to disk
    plt.close(fig)                    # free memory
    print(f"Saved: {save_path}")      # confirm output location

# feature extraction pipeline
def analyze_resonance_fft(resonance_audio, sr, title, save_path, low_hz=LOW_HZ, high_hz=HIGH_HZ):
    """
    Compute the FFT of the resonance window, find the dominant frequency (Fmax),
    locate the top 5 local peaks, mark them with an 'x', and save the spectrum plot.
    """
    n = len(resonance_audio)
    if n == 0:
        raise ValueError("Resonance audio window is empty. Cannot compute FFT.")

    # Compute the FFT and get the absolute magnitudes
    fft_vals = np.fft.fft(resonance_audio)
    freqs = np.fft.fftfreq(n, d=1/sr)

    # Keep only the positive frequencies (one-sided spectrum)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitude = np.abs(fft_vals)[pos_mask]

    # Normalize magnitude relative to window length for physical consistency
    magnitude = magnitude / n

    # Find Fmax (the frequency corresponding to the absolute highest amplitude)
    max_idx = np.argmax(magnitude)
    f_max = freqs[max_idx]

    # Find all local peaks (maxima) in the spectrum
    # 'distance=5' prevents picking adjacent noisy samples on the exact same peak slope
    peaks, _ = find_peaks(magnitude, distance=5)

    # Sort the detected local peaks by their magnitude to isolate the top 5
    if len(peaks) > 0:
        # Sort peak indices based on magnitude values ascending, grab the last 5
        top_peak_indices = peaks[np.argsort(magnitude[peaks])[-5:]]
    else:
        # Fallback: if no local maxima are found, just take the 5 highest individual bins
        top_peak_indices = np.argsort(magnitude)[-5:]

    top_freqs = freqs[top_peak_indices]
    top_mags = magnitude[top_peak_indices]

    # --- Plotting the Spectrum ---
    fig, ax = plt.subplots(figsize=(11, 5))
    
    # Plot the continuous frequency spectrum line
    ax.plot(freqs, magnitude, color="purple", linewidth=1, label="Frequency Spectrum")
    
    # Mark the top 5 peaks with a red 'x' cross sign
    ax.scatter(top_freqs, top_mags, color="red", marker="x", s=100, zorder=5, 
               label=f"Top {len(top_freqs)} Peaks")
    
    # Draw a vertical dashed line right down the center of Fmax for visibility
    ax.axvline(f_max, color="green", linestyle="--", linewidth=1.5,
               label=f"Fmax = {f_max:.1f} Hz")

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (Normalized)")
    
    # Dynamic axis limit tailored to your bandpass filters (plus a 100Hz buffer)
    ax.set_xlim(max(0, low_hz - 100), high_hz + 100)
    
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    print(f"Saved FFT Plot: {save_path} | Fmax = {f_max:.2f} Hz")
    return f_max

def main():
    if not os.path.exists(AUDIO_PATH):  
        raise FileNotFoundError(
            f"Could not find '{AUDIO_PATH}'. Set AUDIO_PATH at the top of this file "
            "to the path of your recording."
        )                                
    
    os.makedirs(PLOTS_DIR, exist_ok=True)  

    audio, sr = load_audio(AUDIO_PATH, sr=None)                     # load the file at its native sample rate
    print(f"Loaded '{AUDIO_PATH}': sr={sr} Hz, duration={len(audio)/sr:.3f}s")  # quick console summary
    plot_audio(audio, sr, " Raw Audio", os.path.join(PLOTS_DIR, "raw_audio.png"))  

    filtered = bandpass_filter(audio, sr)                            # remove rumble + hiss, keep tap/resonance band
    plot_audio(filtered, sr, "Bandpass Filtered Audio",
               os.path.join(PLOTS_DIR, "bandpass_filtered_audio.png"))     

    tap_sample, tap_time, energy, energy_times = detect_tap(filtered, sr)  
    print(f"Detected tap at t = {tap_time:.4f}s")                       # report the detected tap time
    plot_tap_detection(filtered, sr, tap_sample, energy, energy_times,
                        "Tap Detection", os.path.join(PLOTS_DIR, "tap_detection.png")) 

    # resonance window extraction (on filtered+normalized audio, for cleaner resonance shape) 
    resonance_audio, start_sample, end_sample = extract_resonance_window(
        filtered, sr, tap_sample
    )                                                                   
    print(f"Resonance window: {start_sample/sr:.4f}s -> {end_sample/sr:.4f}s "
          f"({(end_sample-start_sample)/sr:.4f}s long)")                  # report the extracted window's bounds
    plot_resonance_window(filtered, sr, tap_sample, start_sample, end_sample,
                           "Resonance Window Extraction",
                           os.path.join(PLOTS_DIR, "resonance_window.png"))  

    print("Maximum Frequency from this audio file is: ", analyze_resonance_fft(resonance_audio, sr, title="Frequency Distribution", save_path=os.path.join(PLOTS_DIR,"FFT Graph.png")))
    
    plot_audio(resonance_audio, sr, "Resonance Audio Graph", os.path.join(PLOTS_DIR, "resonance_audio.png"))  # resonance audio plot
    print(f"\nAll plots saved to ./{PLOTS_DIR}/")   # final confirmation message


if __name__ == "__main__":   
    main()                     