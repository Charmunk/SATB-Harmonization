#!/usr/bin/env python3
"""
Render a chorale CSV (note0..note3 = MIDI, S A T B) to a WAV file.
Each data row is one sixteenth note; default tempo is 120 BPM (quarter = 0.5 s).
Consecutive rows with the same four pitches are rendered as one sustained chord.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import wave
from pathlib import Path


def midi_to_hz(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def sixteenth_seconds(bpm: float) -> float:
    return 60.0 / bpm / 4.0


def fade_edges(buf: list[float], sample_rate: int, fade_ms: float = 4.0) -> None:
    n = len(buf)
    if n == 0:
        return
    flen = max(1, int(sample_rate * fade_ms / 1000.0))
    flen = min(flen, n // 2)
    for i in range(flen):
        g = i / flen
        buf[i] *= g
        buf[n - 1 - i] *= g


def row_to_samples(
    midis: list[int],
    duration: float,
    sample_rate: int,
    per_voice_amp: float,
) -> list[float]:
    n = max(1, int(round(duration * sample_rate)))
    buf = [0.0] * n
    t_step = 1.0 / sample_rate
    for m in midis:
        if m <= 0:
            continue
        hz = midi_to_hz(m)
        w = 2.0 * math.pi * hz
        for i in range(n):
            buf[i] += per_voice_amp * math.sin(w * (i * t_step))
    fade_edges(buf, sample_rate)
    return buf


def mix_to_int16(samples: list[float]) -> bytes:
    peak = max((abs(x) for x in samples), default=0.0)
    if peak > 1.0:
        scale = 1.0 / peak
        samples = [x * scale for x in samples]
    out = bytearray()
    for x in samples:
        v = int(round(max(-1.0, min(1.0, x)) * 32767.0))
        out.extend(struct.pack("<h", v))
    return bytes(out)


def read_chorale_rows(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows
        for row in reader:
            if len(row) < 4:
                continue
            try:
                rows.append([int(row[i]) for i in range(4)])
            except ValueError:
                continue
    return rows


def merged_segments(
    rows: list[list[int]], sixteenth_dur: float
) -> list[tuple[list[int], float]]:
    """Collapse runs of identical SATB tuples; duration is in seconds."""
    if not rows:
        return []
    out: list[tuple[list[int], float]] = []
    cur = rows[0]
    n_run = 1
    for midis in rows[1:]:
        if midis == cur:
            n_run += 1
        else:
            out.append((cur, n_run * sixteenth_dur))
            cur = midis
            n_run = 1
    out.append((cur, n_run * sixteenth_dur))
    return out


def render_wav(
    csv_path: Path,
    wav_path: Path,
    bpm: float = 120.0,
    sample_rate: int = 44100,
) -> None:
    chord_rows = read_chorale_rows(csv_path)
    if not chord_rows:
        raise SystemExit(f"No note rows found in {csv_path}")

    dur = sixteenth_seconds(bpm)
    per_voice = 0.18
    all_samples: list[float] = []
    for midis, seg_dur in merged_segments(chord_rows, dur):
        all_samples.extend(row_to_samples(midis, seg_dur, sample_rate, per_voice))

    pcm = mix_to_int16(all_samples)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def main() -> None:
    p = argparse.ArgumentParser(description="Chorale CSV → WAV (1 row = 1/16 note).")
    p.add_argument("csv", type=Path, help="Path to chorale_*.csv (note0..note3)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .wav path (default: same basename as CSV)",
    )
    p.add_argument("--bpm", type=float, default=120.0, help="Tempo in BPM (default 120)")
    p.add_argument("--sr", type=int, default=44100, help="Sample rate (default 44100)")
    args = p.parse_args()

    out = args.output
    if out is None:
        out = args.csv.with_suffix(".wav")

    render_wav(args.csv, out, bpm=args.bpm, sample_rate=args.sr)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
