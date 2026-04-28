#!/usr/bin/env python3
"""
Transpose all chorales in jsb_chorales_in_c/{train,valid,test} to:
- C major if detected as major
- A minor if detected as minor

The script rewrites each chorale_*.csv in place and deletes stale
chorale_*_chords.csv files.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

KS_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KS_MINOR = (6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def read_chorale_rows(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip note0,note1,note2,note3
        for row in reader:
            if len(row) < 4:
                continue
            try:
                rows.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
            except ValueError:
                continue
    return rows


def write_chorale_rows(path: Path, rows: list[list[int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["note0", "note1", "note2", "note3"])
        writer.writerows(rows)


def pitch_class_histogram(rows: list[list[int]]) -> list[float]:
    hist = [0.0] * 12
    for chord in rows:
        for midi_num in chord:
            if midi_num > 0:
                hist[midi_num % 12] += 1.0
    return hist


def rotate_profile(profile: tuple[float, ...], tonic_pc: int) -> list[float]:
    return [profile[(i - tonic_pc) % 12] for i in range(12)]


def pearson_corr(a: list[float], b: list[float]) -> float:
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    num = 0.0
    den_a = 0.0
    den_b = 0.0
    for xa, xb in zip(a, b):
        da = xa - mean_a
        db = xb - mean_b
        num += da * db
        den_a += da * da
        den_b += db * db
    den = math.sqrt(den_a * den_b)
    if den == 0.0:
        return -1.0
    return num / den


def detect_key_ks(rows: list[list[int]]) -> tuple[str, int, float]:
    hist = pitch_class_histogram(rows)
    best_mode = "major"
    best_tonic = 0
    best_score = float("-inf")

    for tonic in range(12):
        major_score = pearson_corr(hist, rotate_profile(KS_MAJOR, tonic))
        if major_score > best_score:
            best_mode = "major"
            best_tonic = tonic
            best_score = major_score

        minor_score = pearson_corr(hist, rotate_profile(KS_MINOR, tonic))
        if minor_score > best_score:
            best_mode = "minor"
            best_tonic = tonic
            best_score = minor_score

    return best_mode, best_tonic, best_score


def transpose_shift(mode: str, tonic_pc: int) -> int:
    target_pc = 0 if mode == "major" else 9  # C major or A minor
    shift = (target_pc - tonic_pc) % 12
    if shift > 6:
        shift -= 12
    return shift


def transpose_rows(rows: list[list[int]], shift: int) -> list[list[int]]:
    if shift == 0:
        return rows
    return [[m + shift if m > 0 else m for m in chord] for chord in rows]


def process_split(split_dir: Path) -> tuple[int, int]:
    deleted = 0
    transposed = 0

    for chords_path in sorted(split_dir.glob("chorale_*_chords.csv")):
        chords_path.unlink(missing_ok=True)
        deleted += 1

    for csv_path in sorted(split_dir.glob("chorale_*.csv")):
        rows = read_chorale_rows(csv_path)
        if not rows:
            print(f"{csv_path.relative_to(split_dir.parent)}: skipped (no note rows)")
            continue

        mode, tonic_pc, score = detect_key_ks(rows)
        shift = transpose_shift(mode, tonic_pc)
        new_rows = transpose_rows(rows, shift)
        if shift != 0:
            write_chorale_rows(csv_path, new_rows)
            transposed += 1

        tonic_name = PC_NAMES[tonic_pc]
        key_name = f"{tonic_name} {'major' if mode == 'major' else 'minor'}"
        print(
            f"{csv_path.relative_to(split_dir.parent)}: {key_name} "
            f"(score={score:.3f}) -> shift {shift}"
        )

    return deleted, transposed


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    total_deleted = 0
    total_transposed = 0

    for split in ("train", "valid", "test"):
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        deleted, transposed = process_split(split_dir)
        total_deleted += deleted
        total_transposed += transposed

    print(
        f"Done. Deleted {total_deleted} _chords.csv files; "
        f"transposed {total_transposed} chorale CSV files."
    )


if __name__ == "__main__":
    main()
