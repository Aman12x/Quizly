#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Set the input path to the provided video file
input_path = Path(
    "/Users/amansingh/Desktop/PROJECT/temp/Introduction_to_neural_network.mp4"
)


def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it's on your PATH.\n"
            "macOS: brew install ffmpeg\n"
            "Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "Windows: download from ffmpeg.org and add to PATH"
        )


def to_wav_16k_mono(input_path: Path, out_dir: Path) -> Path:
    out_wav = out_dir / (input_path.stem + "_16k_mono.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16 kHz
        "-vn",  # no video
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Show a short error; ffmpeg dumps a lot — include tail for debugging
        tail = e.stderr.decode(errors="ignore").splitlines()[-15:]
        raise RuntimeError("ffmpeg failed to extract audio:\n" + "\n".join(tail)) from e
    return out_wav


def format_timestamp(t):
    # SRT requires comma as decimal separator
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    millis = int(round((t - int(t)) * 1000))
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def write_srt(segments, srt_path: Path):
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line between cues
    srt_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print("Checking ffmpeg installation...")
    try:
        ensure_ffmpeg()
        print("ffmpeg is installed and available on PATH.")
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    # Create a temp directory for audio conversion
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        print(f"Converting {input_path} to 16kHz mono wav...")
        wav_path = to_wav_16k_mono(input_path, tmp_dir)

        try:
            import whisper
        except ImportError:
            print(
                "openai-whisper is not installed. Run: pip install -U openai-whisper",
                file=sys.stderr,
            )
            sys.exit(1)

        print("Loading Whisper model (base)...")
        model = whisper.load_model("base")
        print(f"Transcribing {wav_path.name}...")
        result = model.transcribe(str(wav_path), verbose=False)

        text = (result.get("text") or "").strip()
        segments = result.get("segments", [])

        # Write transcript and subtitles
        out_txt = input_path.with_suffix(".txt")
        out_srt = input_path.with_suffix(".srt")
        out_txt.write_text(text, encoding="utf-8")
        write_srt(segments, out_srt)
        print(f"✓ Wrote transcript: {out_txt}")
        print(f"✓ Wrote subtitles: {out_srt}")
