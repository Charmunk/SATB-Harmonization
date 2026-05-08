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
from mm_chord_to_chord_dd import *


def get_part_line(lookup_table, A, soprano_input, l=1):
    """
    Inputs
      - lookup_table: a matrix containing log probabilities P(delta | soprano_{n-1}, part_n) of size ((numchords), 128, 2*num_semitones+2), indexed (part, prev_soprano, delta)
      - A: an numchords x numchords matrix specifying transitions of part's notes between moments in time
      - soprano input -- a 1d array of soprano midi notes
        
    Outputs
      - part_est: the supporting part line
    """
    ### INSERT CODE BELOW ###
    numchords = lookup_table.shape[0]
    #grab dimensions and initialize cumulative cost matrix
    N = np.size(soprano_input)
    M = numchords

    #make sure A and pi has no zeros
    A = np.clip(A, 1E-9, 1) # we don't even have to renormalize?

    D = np.zeros((M, N))
    B = np.zeros((M, N))

    #initialize probabilities
    D[:, 0] = np.sum(lookup_table[:, :, :], axis=(1,2)) #should add log(P(O_1 | S_1)) term
    #just writes total probability of being in that bass note (sums over soprano and delta dimensions)

    #lookup table is indexed (bass, prev_soprano, delta)
    #walk through the matrix
    for j in range(1, N):
      delta = soprano_input[j] - soprano_input[j-1]
      for i in range(M):
        D[i, j] = np.max(D[:, j-1] + np.log(A[:, i])+ l*lookup_table[:, int(soprano_input[j-1]), delta_2_index(delta)])
        B[i,j] = np.argmax(D[:, j-1] + np.log(A[:, i])+ l*lookup_table[:, int(soprano_input[j-1]), delta_2_index(delta)])

        #best_index = int(B[i,j])
        #print(np.log(A[best_index, i]), l*lookup_table[best_index, int(soprano_input[j-1]), delta_2_index(delta)])
  
    part_est = np.zeros(N, dtype = int)
    endpoint = np.argmax(D[:, -1])
    part_est[-1] = endpoint

    for i in range(N-1)[::-1]:
      part_est[i] = B[int(part_est[i+1]), i+1]

    return part_est

def chordseq_to_parts(chordseq, idx_to_chord):
    """
    Inputs:
        - chordseq: Nx1 arr of chord indices
        - idx_to_chord: dict mapping chord_idx(int) --> (A, T, B) (tuple of midi ints)
    Outputs:
        - part_A: Nx1 arr of ints, corresponding to alto line (midi notes)
        - part_T: Nx1 arr of ints, corresponding to tenor line (midi notes)
        - part_B: Nx1 arr of ints, corresponding to base line (midi notes)
    """
    N = len(chordseq)
    part_A = np.zeros(N, dtype = int)
    part_T = np.zeros(N, dtype = int)
    part_B = np.zeros(N, dtype = int)

    for i in range(len(chordseq)):
        chord_idx = chordseq[i]
        chord = idx_to_chord[chord_idx]
        (A, T, B) = chord
        part_A[i] = A
        part_T[i] = T
        part_B[i] = B

    return (part_A, part_T, part_B)


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
        default=repo / "generated_dd",
        help="Output directory for generatede SATB CSVs.",
    )
    parser.add_argument("--l", type=float, default=1.0, help="Likelihood scale (same as notebook l=1).")
    args = parser.parse_args()

    train_dir = str(args.train_dir.resolve())
    if not args.train_dir.is_dir():
        print(f"Training directory not found: {args.train_dir}", file=sys.stderr)
        sys.exit(1)

    print("Building lookups and transitions from", train_dir)
    chord_to_idx, idx_to_chord, chord_to_chord = get_chord_dict(train_dir)
    lookup_table_chord = get_delta_3d_lookup_chord(train_dir, chord_to_idx)
    


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
            chordseq = get_part_line(lookup_table_chord, chord_to_chord, S, l=1)
            A, T, B = chordseq_to_parts(chordseq, idx_to_chord)
            dest = out_root / src.name
            write_satb_csv(dest, S, A, T, B)
            ok += 1
            print(dest)
        except Exception as e:
            print(f"FAILED {src}: {e}", file=sys.stderr)

    print(f"Wrote {ok}/{len(csvs)} files to {out_root}")


if __name__ == "__main__":
    main()
