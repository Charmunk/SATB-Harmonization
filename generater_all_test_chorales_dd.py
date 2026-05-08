"""
Batch-generate SATB CSVs: keep each test-set soprano from jsb_chorales/test,
infer (A,T,B) chord with the same pipeline as generate_alto_tenor_bass_c-normalized.ipynb.

Lookup tables and part transitions are trained on jsb_chorales_in_c/train (C-normalized).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from lookup_delta_3d import csv_to_tracks, delta_2_index, get_delta_3d_lookup
from mm_part_to_part import train_part_to_part_markov


def get_part_line(lookup_table: np.ndarray, A: np.ndarray, soprano_input: np.ndarray, l: float = 1.0) -> np.ndarray:
    """Viterbi-style decoding from notebook generate_alto_tenor_bass_c-normalized.ipynb."""
    N = np.size(soprano_input)
    M = 128

    A = np.clip(A, 1e-9, 1.0)

    D = np.zeros((M, N))
    B = np.zeros((M, N), dtype=np.int64)

    D[:, 0] = np.sum(lookup_table[:, :, :], axis=(1, 2))

    for j in range(1, N):
        delta = soprano_input[j] - soprano_input[j - 1]
        prev_s = int(soprano_input[j - 1])
        d_idx = delta_2_index(delta)
        for i in range(M):
            scores = D[:, j - 1] + np.log(A[:, i]) + l * lookup_table[:, prev_s, d_idx]
            D[i, j] = np.max(scores)
            B[i, j] = np.argmax(scores)

    part_est = np.zeros(N, dtype=np.int64)
    endpoint = int(np.argmax(D[:, -1]))
    part_est[-1] = endpoint

    for i in range(N - 2, -1, -1):
        part_est[i] = B[int(part_est[i + 1]), i + 1]

    return part_est


def write_satb_csv(path: Path, S: np.ndarray, A: np.ndarray, T: np.ndarray, B: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["note0", "note1", "note2", "note3"])
        for row in zip(S, A, T, B):
            w.writerow([int(row[0]), int(row[1]), int(row[2]), int(row[3])])


def main() -> None:
    repo = Path(__file__).resolve().parent
    os.chdir(repo)

    parser = argparse.ArgumentParser(description="Generate SATB CSVs from test sopranos.")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=repo / "jsb_chorales" / "test",
        help="Directory of ground-truth chorales (only soprano is kept). Default: jsb_chorales/test",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=repo / "jsb_chorales_in_c" / "train",
        help="Training directory for lookups / transitions. Default: jsb_chorales_in_c/train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "generated",
        help="Output directory for generated SATB CSVs.",
    )
    parser.add_argument("--l", type=float, default=1.0, help="Likelihood scale (same as notebook l=1).")
    args = parser.parse_args()

    train_dir = str(args.train_dir.resolve())
    if not args.train_dir.is_dir():
        print(f"Training directory not found: {args.train_dir}", file=sys.stderr)
        sys.exit(1)

    print("Building lookups and transitions from", train_dir)
    lookup_table_bass = get_delta_3d_lookup(train_dir)
    lookup_table_tenor = get_delta_3d_lookup(train_dir, part="tenor")
    lookup_table_alto = get_delta_3d_lookup(train_dir, part="alto")
    alto_to_alto_transition, tenor_to_tenor_transition, bass_to_bass_transition = train_part_to_part_markov(train_dir)

    test_dir = args.test_dir.resolve()
    if not test_dir.is_dir():
        print(f"Test directory not found: {test_dir}", file=sys.stderr)
        sys.exit(1)

    csvs = sorted(
        p for p in test_dir.glob("chorale_*.csv") if not p.name.endswith("_chords.csv")
    )
    if not csvs:
        print(f"No chorale_*.csv files in {test_dir}", file=sys.stderr)
        sys.exit(1)

    out_root = args.output_dir.resolve()
    ok = 0
    for src in csvs:
        try:
            S, _, _, _ = csv_to_tracks(str(src))
            S = np.asarray(S, dtype=np.float64).astype(np.int64)
            B = get_part_line(lookup_table_bass, bass_to_bass_transition, S, l=args.l)
            A = get_part_line(lookup_table_alto, alto_to_alto_transition, S, l=args.l)
            T = get_part_line(lookup_table_tenor, tenor_to_tenor_transition, S, l=args.l)
            dest = out_root / src.name
            write_satb_csv(dest, S, A, T, B)
            ok += 1
            print(dest)
        except Exception as e:
            print(f"FAILED {src}: {e}", file=sys.stderr)

    print(f"Wrote {ok}/{len(csvs)} files to {out_root}")


if __name__ == "__main__":
    main()
