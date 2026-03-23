"""
Timbre Demo — Lesson 1.1: What is Sound?

Generates synthetic tones with controlled harmonic content to demonstrate
what timbre is. Each tone plays the same note (A4 = 440 Hz) at the same
volume, but with different overtone recipes — producing different timbres.

Outputs:
  - WAV files for each tone (listen to hear the timbre differences)
  - Waveform comparison (see the wave shapes differ)
  - Frequency spectrum comparison (see which harmonics are present)
"""

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
SR = 44100          # sample rate
DURATION = 3.0      # seconds per tone
FREQ = 440.0        # A4
OUTPUT = Path("/workspace/MusicEducation/lessons/1.1_sound_and_timbre")

# ── Tone Generators ─────────────────────────────────────────────────────

def make_time(duration=DURATION, sr=SR):
    return np.linspace(0, duration, int(sr * duration), endpoint=False)


def pure_sine(t):
    """Just the fundamental — no overtones at all."""
    return np.sin(2 * np.pi * FREQ * t)


def flute_like(t):
    """Fundamental dominant, faint 2nd and 3rd harmonics.
    Flutes produce mostly the fundamental with very weak overtones,
    giving them that pure, airy quality."""
    return (
        1.0  * np.sin(2 * np.pi * FREQ * 1 * t) +
        0.12 * np.sin(2 * np.pi * FREQ * 2 * t) +
        0.06 * np.sin(2 * np.pi * FREQ * 3 * t)
    )


def clarinet_like(t):
    """Odd harmonics only (1, 3, 5, 7...) — characteristic of cylindrical
    bore instruments like the clarinet. The missing even harmonics give it
    a hollow, woody quality."""
    signal = np.zeros_like(t)
    for n in [1, 3, 5, 7, 9, 11]:
        signal += (1.0 / n) * np.sin(2 * np.pi * FREQ * n * t)
    return signal


def sawtooth_like(t):
    """All harmonics (1, 2, 3, 4, 5...) decreasing as 1/n.
    This is close to a sawtooth wave — sounds buzzy and bright.
    Think: bowed string, brass, or a raw synth oscillator."""
    signal = np.zeros_like(t)
    for n in range(1, 16):
        signal += (1.0 / n) * np.sin(2 * np.pi * FREQ * n * t)
    return signal


def bell_like(t):
    """Non-integer frequency ratios — inharmonic overtones.
    Bells and metallic percussion have overtones that AREN'T neat
    multiples of the fundamental. This is why they sound 'metallic'
    and don't have a clear pitch."""
    return (
        1.0  * np.sin(2 * np.pi * FREQ * 1.0   * t) +
        0.7  * np.sin(2 * np.pi * FREQ * 2.76   * t) +
        0.5  * np.sin(2 * np.pi * FREQ * 5.404  * t) +
        0.3  * np.sin(2 * np.pi * FREQ * 8.933  * t) +
        0.2  * np.sin(2 * np.pi * FREQ * 13.344 * t)
    )


# ── Envelope (fade in/out to avoid clicks) ──────────────────────────────

def apply_envelope(signal, sr=SR, fade_ms=30):
    fade_samples = int(sr * fade_ms / 1000)
    env = np.ones_like(signal)
    env[:fade_samples] = np.linspace(0, 1, fade_samples)
    env[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return signal * env


def normalize(signal):
    peak = np.max(np.abs(signal))
    if peak > 0:
        return signal / peak * 0.9  # leave a little headroom
    return signal


# ── Generate All Tones ──────────────────────────────────────────────────

TONES = {
    "1_pure_sine":     ("Pure Sine Wave", "Just 440 Hz — no overtones", pure_sine),
    "2_flute_like":    ("Flute-like", "Strong fundamental, faint 2nd & 3rd harmonics", flute_like),
    "3_clarinet_like": ("Clarinet-like", "Odd harmonics only (1,3,5,7,9,11)", clarinet_like),
    "4_sawtooth_like": ("Sawtooth-like", "All harmonics 1-15, decreasing", sawtooth_like),
    "5_bell_like":     ("Bell-like", "Non-integer ratios (inharmonic)", bell_like),
}

def generate_tones():
    t = make_time()
    audio = {}
    for key, (label, desc, func) in TONES.items():
        sig = normalize(apply_envelope(func(t)))
        audio[key] = (sig, label, desc)
        wav_path = OUTPUT / f"{key}.wav"
        sf.write(str(wav_path), sig, SR)
        print(f"  Wrote {wav_path.name}")
    return t, audio


# ── Visualization: Waveform Comparison ──────────────────────────────────

def plot_waveforms(t, audio):
    """Zoomed-in waveforms showing ~4.5 cycles so you can see the shape."""
    fig, axes = plt.subplots(len(audio), 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Waveform Comparison — Same Note (A4 = 440 Hz), Different Timbres",
        fontsize=15, fontweight="bold", y=0.98
    )

    # Show ~4.5 cycles of 440 Hz, from the MIDDLE of the signal
    # (avoids the fade-in envelope at the start)
    cycles_to_show = 4.5
    samples_to_show = int(cycles_to_show / FREQ * SR)
    mid = len(t) // 2
    start = mid - samples_to_show // 2
    end = start + samples_to_show

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for i, (key, (sig, label, desc)) in enumerate(audio.items()):
        ax = axes[i]
        t_window = (t[start:end] - t[start]) * 1000  # zero-base the time axis
        ax.plot(t_window, sig[start:end],
                color=colors[i], linewidth=1.5)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_title(f"{label}  —  {desc}", fontsize=11, loc="left")
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (milliseconds)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUTPUT / "waveform_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path.name}")


# ── Visualization: Frequency Spectrum ───────────────────────────────────

def plot_spectra(t, audio):
    """FFT magnitude plots showing which harmonics are present."""
    fig, axes = plt.subplots(len(audio), 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        "Frequency Spectrum — Which Harmonics Are Present?",
        fontsize=15, fontweight="bold", y=0.98
    )

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    # Mark harmonic positions
    harmonic_freqs = [FREQ * n for n in range(1, 17)]

    for i, (key, (sig, label, desc)) in enumerate(audio.items()):
        ax = axes[i]

        # Compute FFT
        N = len(sig)
        fft_mag = np.abs(np.fft.rfft(sig)) / N * 2
        fft_freqs = np.fft.rfftfreq(N, 1 / SR)

        # Only show up to ~8000 Hz (covers harmonics we care about)
        mask = fft_freqs <= 8000
        ax.plot(fft_freqs[mask], fft_mag[mask], color=colors[i], linewidth=1.0)

        # Mark integer harmonics with dotted lines
        for n, hf in enumerate(harmonic_freqs[:16], 1):
            if hf <= 8000:
                ax.axvline(hf, color="gray", linewidth=0.5, alpha=0.3, linestyle=":")
                if i == 0:  # label only on first subplot
                    ax.text(hf, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.9,
                            f"{n}×", fontsize=7, ha="center", va="bottom",
                            color="gray", alpha=0.6)

        ax.set_ylabel("Magnitude", fontsize=9)
        ax.set_title(f"{label}  —  {desc}", fontsize=11, loc="left")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=11)

    # Re-do harmonic labels on top plot after all limits are set
    axes[0].set_ylim(auto=True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUTPUT / "frequency_spectrum.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path.name}")


# ── Visualization: Harmonic Recipe Bar Chart ────────────────────────────

def plot_harmonic_recipes(audio):
    """Bar charts showing the 'recipe' of each timbre — how much of each
    harmonic is present. This is the most direct visualization of timbre."""
    fig, axes = plt.subplots(1, len(audio), figsize=(16, 5), sharey=True)
    fig.suptitle(
        "Timbre = Harmonic Recipe  —  How Much of Each Harmonic?",
        fontsize=14, fontweight="bold", y=1.02
    )

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    max_harmonic = 16

    for i, (key, (sig, label, desc)) in enumerate(audio.items()):
        ax = axes[i]

        # Measure amplitude at each harmonic via FFT
        N = len(sig)
        fft_mag = np.abs(np.fft.rfft(sig)) / N * 2
        fft_freqs = np.fft.rfftfreq(N, 1 / SR)

        harmonic_amps = []
        for n in range(1, max_harmonic + 1):
            target = FREQ * n
            # Find nearest FFT bin
            idx = np.argmin(np.abs(fft_freqs - target))
            # Take max in small neighborhood to handle spectral leakage
            lo = max(0, idx - 3)
            hi = min(len(fft_mag), idx + 4)
            harmonic_amps.append(np.max(fft_mag[lo:hi]))

        # Normalize to max
        peak = max(harmonic_amps) if max(harmonic_amps) > 0 else 1
        harmonic_amps = [a / peak for a in harmonic_amps]

        bars = ax.bar(range(1, max_harmonic + 1), harmonic_amps,
                      color=colors[i], alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Harmonic #", fontsize=9)
        ax.set_xticks(range(1, max_harmonic + 1))
        ax.set_xticklabels(range(1, max_harmonic + 1), fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.grid(True, axis="y", alpha=0.2)

    axes[0].set_ylabel("Relative Amplitude", fontsize=10)
    fig.tight_layout()
    path = OUTPUT / "harmonic_recipes.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path.name}")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating tones...")
    t, audio = generate_tones()

    print("\nPlotting waveforms...")
    plot_waveforms(t, audio)

    print("\nPlotting frequency spectra...")
    plot_spectra(t, audio)

    print("\nPlotting harmonic recipes...")
    plot_harmonic_recipes(audio)

    print("\nDone! Files are in:", OUTPUT)
