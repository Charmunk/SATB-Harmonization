#!/usr/bin/env python3
"""
Build a 'bad' benchmark: random MIDI SATB CSVs with the same shape as a template folder,
then optionally run bach_grader.py and save scores next to the CSVs.

Example:
  python3 build_random_baseline.py
  python3 build_random_baseline.py --no-grade   # only write CSVs
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def row_count_chorale_csv(path: Path) -> int:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for row in reader if len(row) >= 4)


def write_random_csv(path: Path, n_rows: int, rng, lo: int, hi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["note0", "note1", "note2", "note3"])
        for _ in range(n_rows):
            w.writerow([int(rng.integers(lo, hi + 1)) for _ in range(4)])


def main() -> None:
    repo = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Random SATB CSVs for grader lower bound.")
    p.add_argument(
        "--template-dir",
        type=Path,
        default=repo / "jsb_chorales_in_c" / "test",
        help="Use each chorale_*.csv here only for filename and row count.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "random_chorales",
        help="Where to write random chorale_*.csv files.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible garbage).")
    p.add_argument("--lo", type=int, default=40, help="Inclusive low MIDI note.")
    p.add_argument("--hi", type=int, default=84, help="Inclusive high MIDI note.")
    p.add_argument(
        "--no-grade",
        action="store_true",
        help="Only write CSVs; do not run bach_grader.py.",
    )
    args = p.parse_args()

    import numpy as np

    tmpl = args.template_dir.resolve()
    if not tmpl.is_dir():
        print(f"Template dir not found: {tmpl}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(
        x for x in tmpl.glob("chorale_*.csv") if not x.name.endswith("_chords.csv")
    )
    if not paths:
        print(f"No chorale_*.csv in {tmpl}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    out_root = args.output_dir.resolve()
    for src in paths:
        n = row_count_chorale_csv(src)
        write_random_csv(out_root / src.name, n, rng, args.lo, args.hi)
        print(out_root / src.name)

    print(f"Wrote {len(paths)} files to {out_root}")

    if args.no_grade:
        return

    grader = repo / "bach_grader.py"
    if not grader.is_file():
        print(f"bach_grader.py not found at {grader}", file=sys.stderr)
        sys.exit(1)

    json_out = out_root / "bach_grader_random_baseline.json"
    csv_out = out_root / "bach_grader_random_baseline.csv"
    cmd = [
        sys.executable,
        str(grader),
        "--target-dir",
        str(out_root),
        "--output-json",
        str(json_out),
    ]
    print("Running:", " ".join(cmd))
    with csv_out.open("w") as fout:
        r = subprocess.run(cmd, cwd=str(repo), stdout=fout, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"Wrote {csv_out}")
    print(f"Wrote {json_out}")


if __name__ == "__main__":
    main()
