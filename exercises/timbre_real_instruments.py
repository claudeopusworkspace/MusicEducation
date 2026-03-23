"""
Real Instrument Timbre Comparison — Lesson 1.1

Generates the same musical phrase on different instruments using Foundation-1,
then creates comparative analysis visualizations. This demonstrates timbre
differences as they actually sound in real instruments — with attack, decay,
texture, and evolving harmonic content.
"""

import sys
sys.path.insert(0, "/workspace/MusicTest")

from src.generate import generate, save_wav, build_prompt, load_model
from src.analyze import spectrogram, chromagram, waveform
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT = Path("/workspace/MusicEducation/lessons/1.1_sound_and_timbre/real_instruments")
OUTPUT.mkdir(parents=True, exist_ok=True)

# All instruments play the same thing: 2 bars, A minor, 120 BPM
# Using the same seed for each to get as similar a musical phrase as possible
COMMON = dict(key="A minor", bars=2, bpm=120)
SEED = 42

INSTRUMENTS = [
    {
        "name": "piano",
        "instrument": "Piano",
        "timbre": "warm acoustic grand piano",
        "behavior": "sustained chords, legato",
    },
    {
        "name": "guitar",
        "instrument": "Acoustic Guitar",
        "timbre": "warm nylon string acoustic guitar",
        "behavior": "gentle fingerpicked arpeggios",
    },
    {
        "name": "strings",
        "instrument": "String Ensemble",
        "timbre": "warm orchestral string ensemble, lush",
        "behavior": "sustained legato chords, slow movement",
    },
    {
        "name": "synth_pad",
        "instrument": "Synthesizer Pad",
        "timbre": "warm analog synth pad, soft",
        "behavior": "sustained evolving pad, slow filter sweep",
    },
    {
        "name": "flute",
        "instrument": "Flute",
        "timbre": "breathy concert flute, airy",
        "behavior": "gentle melodic phrase, legato",
    },
]


def generate_all():
    """Generate each instrument sample."""
    print("Loading Foundation-1...")
    load_model()

    results = {}
    for inst in INSTRUMENTS:
        name = inst["name"]
        prompt = build_prompt(
            instrument=inst["instrument"],
            timbre=inst["timbre"],
            behavior=inst["behavior"],
            **COMMON
        )
        print(f"\n  Generating {name}...")
        print(f"    Prompt: {prompt}")

        audio, sr = generate(prompt, bars=2, bpm=120, steps=100,
                             cfg_scale=7.0, seed=SEED)
        wav_path = OUTPUT / f"{name}.wav"
        save_wav(audio, sr, str(wav_path))
        print(f"    Saved: {wav_path.name}")
        results[name] = (wav_path, sr)

    return results


def create_comparison_spectrogram(results):
    """Create a side-by-side spectrogram comparison showing how each
    instrument distributes energy across frequencies differently."""
    import librosa
    import librosa.display

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 3.2 * len(results)))
    fig.suptitle(
        "Spectrogram Comparison — Same Key, Tempo & Seed, Different Instruments\n"
        "(Brighter = more energy at that frequency at that time)",
        fontsize=14, fontweight="bold", y=1.01
    )

    for i, (name, (wav_path, sr)) in enumerate(results.items()):
        ax = axes[i]
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                           fmax=8000, n_fft=2048,
                                           hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(S_dB, sr=sr, hop_length=512,
                                       x_axis="time", y_axis="mel",
                                       ax=ax, cmap="magma", vmin=-60, vmax=0)
        label = name.replace("_", " ").title()
        ax.set_title(f"{label}", fontsize=12, loc="left", fontweight="bold")
        ax.set_xlabel("")

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.tight_layout()
    path = OUTPUT / "spectrogram_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Wrote {path.name}")


def create_waveform_comparison(results):
    """Show the waveform envelopes — how each instrument's volume
    evolves over time (attack, sustain, decay)."""
    import librosa

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 2.5 * len(results)),
                             sharex=True)
    fig.suptitle(
        "Waveform Comparison — Notice How Each Instrument Starts and Sustains Differently",
        fontsize=14, fontweight="bold", y=1.01
    )

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for i, (name, (wav_path, sr)) in enumerate(results.items()):
        ax = axes[i]
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        t = np.arange(len(y)) / sr

        ax.plot(t, y, color=colors[i % len(colors)], linewidth=0.3, alpha=0.6)

        # Overlay RMS envelope for clarity
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
        ax.plot(rms_t, rms, color=colors[i % len(colors)], linewidth=2.0,
                label="Volume envelope")
        ax.plot(rms_t, -rms, color=colors[i % len(colors)], linewidth=2.0)

        label = name.replace("_", " ").title()
        ax.set_title(f"{label}", fontsize=12, loc="left", fontweight="bold")
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.tight_layout()
    path = OUTPUT / "waveform_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path.name}")


if __name__ == "__main__":
    results = generate_all()

    print("\nCreating comparison visualizations...")
    create_comparison_spectrogram(results)
    create_waveform_comparison(results)
    print("\nDone! Files in:", OUTPUT)
