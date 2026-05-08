# SATB-Harmonization
For a given input soprano melody, we generate reasonable alto, tenor, and bass harmonies. 

Dataset: https://github.com/ageron/handson-ml2/blob/master/datasets/jsb_chorales/README.md
Chords separated into MIDI notes, sampled every 1/16th note. 
Format:
S, A, T, B
67,63,60,48
67,63,60,48
67,63,60,48
67,63,60,48
72,67,63,60
72,67,63,60

Assumptions: SATB voices never overlap.

## Bach-style grader results (`bach_grader.py`)

The script scores every `chorale_*.csv` in a folder by comparing simple **distributions** (pitch, rhythm, per-voice intervals, rough chord quality, parallel motion counts, repeated-motif stats) against a **reference corpus**. It is a loose implementation of the idea in [*Bach or Mock?* (arXiv:2006.13329)](https://arxiv.org/abs/2006.13329): the grade is a sum of distances between distributions; **lower overall is closer to the reference** on these features, not a universal “musical quality” rating.

### How to read the numbers

- **Overall** — Sum of the per-feature distances. Use it to **rank models or folders** on the same reference split; it is not calibrated to a fixed “pass/fail” scale.
- **pitch / rhythm** — Distance in how often each pitch class (or note length) appears vs reference.
- **intS, intA, intT, intB** — Distance in **directed melodic intervals** (steps/leaps) in each voice vs reference.
- **harmony** — Distance in a coarse chord-quality histogram (major/minor/dominant 7 / other) from each timestep’s four notes.
- **parallel** — Parallel fifth/octave statistics vs reference, with an extra penalty when the error rate is high relative to Bach in the reference set.
- **repeated** — Distance in statistics of repeated SATB patterns (approximation of repeated-sequence behavior in the paper).

Large spikes in **one** column usually mean that chorale differs most from the reference **on that dimension**, which helps debugging (e.g. bad tenor leaps show up under **intT**).

### Baseline vs generated (this repo)

- **Reference** for default runs: `jsb_chorales_in_c/valid` (edit `--reference-dir` to change it).
- **Baseline (real Bach, same grading recipe):** scores for the ground-truth test split are in `jsb_chorales_in_c/test/bach_grader_baseline.csv` and `.json` — use these as a ballpark for how low scores can go when the inner voices are real Bach, compared to the same reference.
- **Model output:** scores for harmonizations in `generated/` are in `generated/bach_grader_results.csv` and `.json`.

Run:

```bash
python3 bach_grader.py --target-dir generated/
python3 bach_grader.py --target-dir jsb_chorales_in_c/test   # baseline
```

Optional: `--cache-reference ref.json` to avoid rebuilding the reference distributions every time.
