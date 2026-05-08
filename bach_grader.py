import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np


STEP_QUARTERS = 0.25  # dataset uses 16th-note grid
VOICE_NAMES = ["S", "A", "T", "B"]
NUMERIC_FEATURES = {
    "pitch",
    "rhythm",
    "interval_S",
    "interval_A",
    "interval_T",
    "interval_B",
    "repeated_sequences",
}


def load_chorale_csv(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            rows.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
    if not rows:
        raise ValueError(f"No SATB rows parsed from {path}")
    return np.asarray(rows, dtype=np.int32)


def iter_csvs(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("chorale_*.csv")):
        if p.name.endswith("_chords.csv"):
            continue
        yield p


def note_durations_quarters(voice: Sequence[int]) -> List[float]:
    out: List[float] = []
    run_len = 1
    for i in range(1, len(voice)):
        if voice[i] == voice[i - 1]:
            run_len += 1
        else:
            out.append(run_len * STEP_QUARTERS)
            run_len = 1
    out.append(run_len * STEP_QUARTERS)
    return out


def directed_intervals(voice: Sequence[int]) -> List[int]:
    intervals: List[int] = []
    last = voice[0]
    for n in voice[1:]:
        if n != last:
            intervals.append(int(n - last))
            last = n
    return intervals


def classify_chord_quality(chord_notes: Sequence[int]) -> str:
    pcs = sorted({n % 12 for n in chord_notes})
    if len(pcs) < 3:
        return "other"

    # Try every pitch class as a root candidate.
    for root in pcs:
        rel = sorted(((p - root) % 12) for p in pcs)
        rel_set = set(rel)
        if {0, 4, 7}.issubset(rel_set):
            if 10 in rel_set:
                return "dominant7"
            return "major"
        if {0, 3, 7}.issubset(rel_set):
            return "minor"
        if {0, 3, 6}.issubset(rel_set):
            return "diminished"
        if {0, 4, 8}.issubset(rel_set):
            return "augmented"
    return "other"


def parallel_errors(chorale: np.ndarray) -> Tuple[int, int]:
    p5 = 0
    p8 = 0
    for t in range(len(chorale) - 1):
        before = chorale[t]
        after = chorale[t + 1]
        for i in range(4):
            for j in range(i + 1, 4):
                d1_i = int(after[i] - before[i])
                d1_j = int(after[j] - before[j])
                if d1_i == 0 or d1_j == 0:
                    continue
                int_before = abs(int(before[i] - before[j])) % 12
                int_after = abs(int(after[i] - after[j])) % 12
                if int_before == int_after == 7:
                    p5 += 1
                if int_before in (0,) and int_after in (0,):
                    p8 += 1
                elif abs(int(before[i] - before[j])) % 12 == 0 and abs(int(after[i] - after[j])) % 12 == 0:
                    p8 += 1
    return p5, p8


def repeated_sequence_lengths(chorale: np.ndarray, min_len: int = 2) -> List[float]:
    # Approximation of repeated-motif feature: repeated SATB n-grams.
    tuples = [tuple(int(x) for x in row) for row in chorale]
    n = len(tuples)
    lengths: List[float] = []
    for k in range(min_len, max(min_len, n // 2) + 1):
        counts: Dict[Tuple[Tuple[int, int, int, int], ...], int] = defaultdict(int)
        for i in range(0, n - k + 1):
            seq = tuple(tuples[i : i + k])
            counts[seq] += 1
        for c in counts.values():
            if c >= 2:
                lengths.extend([k * STEP_QUARTERS] * c)
    return lengths


def normalize_counter(counter: Counter) -> Dict[Hashable, float]:
    total = float(sum(counter.values()))
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def wasserstein_from_pmfs(p: Dict[float, float], q: Dict[float, float]) -> float:
    support = sorted(set(p.keys()) | set(q.keys()))
    if not support:
        return 0.0
    cdf_p = 0.0
    cdf_q = 0.0
    dist = 0.0
    for i, x in enumerate(support):
        cdf_p += p.get(x, 0.0)
        cdf_q += q.get(x, 0.0)
        if i < len(support) - 1:
            dx = support[i + 1] - x
            dist += abs(cdf_p - cdf_q) * dx
    return float(dist)


def categorical_l1_distance(p: Dict[Hashable, float], q: Dict[Hashable, float]) -> float:
    support = set(p.keys()) | set(q.keys())
    return float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in support))


def distribution_distance(p: Dict[Hashable, float], q: Dict[Hashable, float]) -> float:
    if not p and not q:
        return 0.0
    keys = set(p.keys()) | set(q.keys())
    if all(isinstance(k, (int, float, np.integer, np.floating)) for k in keys):
        pn = {float(k): v for k, v in p.items()}
        qn = {float(k): v for k, v in q.items()}
        return wasserstein_from_pmfs(pn, qn)
    return categorical_l1_distance(p, q)


def chorale_features(chorale: np.ndarray) -> Dict[str, Dict[Hashable, float]]:
    feats: Dict[str, Dict[Hashable, float]] = {}

    pitch_counter = Counter(int(n % 12) for n in chorale.flatten())
    feats["pitch"] = normalize_counter(pitch_counter)

    rhythm_counter: Counter = Counter()
    for vi in range(4):
        rhythm_counter.update(note_durations_quarters(chorale[:, vi].tolist()))
    feats["rhythm"] = normalize_counter(rhythm_counter)

    for vi, vn in enumerate(VOICE_NAMES):
        inter_counter = Counter(directed_intervals(chorale[:, vi].tolist()))
        feats[f"interval_{vn}"] = normalize_counter(inter_counter)

    qual_counter = Counter(classify_chord_quality(row.tolist()) for row in chorale)
    feats["harmonic_quality"] = normalize_counter(qual_counter)

    p5, p8 = parallel_errors(chorale)
    feats["parallel_errors"] = normalize_counter(Counter({"p5": p5, "p8": p8}))
    feats["parallel_count"] = {"count": float(p5 + p8)}
    feats["note_count"] = {"count": float(chorale.size)}

    rep_counter = Counter(repeated_sequence_lengths(chorale))
    feats["repeated_sequences"] = normalize_counter(rep_counter)
    return feats


def aggregate_reference(feature_list: List[Dict[str, Dict[Hashable, float]]]) -> Dict[str, Dict[Hashable, float]]:
    raw_values: Dict[str, Counter] = defaultdict(Counter)
    total_parallel = 0.0
    total_notes = 0.0

    for f in feature_list:
        for k, pmf in f.items():
            if k in ("parallel_count", "note_count"):
                continue
            for value, prob in pmf.items():
                raw_values[k][value] += prob
        total_parallel += f["parallel_count"]["count"]
        total_notes += f["note_count"]["count"]

    ref: Dict[str, Dict[Hashable, float]] = {}
    for k, cnt in raw_values.items():
        ref[k] = normalize_counter(cnt)
    ref["parallel_ratio"] = {"ratio": total_parallel / max(total_notes, 1.0)}
    return ref


def score_one(
    feats: Dict[str, Dict[Hashable, float]],
    ref: Dict[str, Dict[Hashable, float]],
) -> Dict[str, float]:
    feature_names = [
        "pitch",
        "rhythm",
        "interval_S",
        "interval_A",
        "interval_T",
        "interval_B",
        "harmonic_quality",
        "parallel_errors",
        "repeated_sequences",
    ]
    out: Dict[str, float] = {}
    total = 0.0

    for name in feature_names:
        d = distribution_distance(feats.get(name, {}), ref.get(name, {}))
        if name == "parallel_errors":
            ratio = feats["parallel_count"]["count"] / max(feats["note_count"]["count"], 1.0)
            bach_ratio = ref.get("parallel_ratio", {}).get("ratio", 1e-9)
            w = ratio / max(bach_ratio, 1e-9)
            d *= w
        out[name] = d
        total += d
    out["overall_grade"] = total
    return out


def compute_reference_from_dir(reference_dir: Path) -> Dict[str, Dict[Hashable, float]]:
    feats = [chorale_features(load_chorale_csv(p)) for p in iter_csvs(reference_dir)]
    if not feats:
        raise ValueError(f"No chorales found in reference dir: {reference_dir}")
    return aggregate_reference(feats)


def serialize_reference(ref: Dict[str, Dict[Hashable, float]]) -> str:
    payload: Dict[str, Dict[str, float]] = {}
    for feature, dist in ref.items():
        payload[feature] = {str(k): float(v) for k, v in dist.items()}
    return json.dumps(payload)


def deserialize_reference(raw: str) -> Dict[str, Dict[Hashable, float]]:
    payload = json.loads(raw)
    ref: Dict[str, Dict[Hashable, float]] = {}
    for feature, dist in payload.items():
        if feature == "parallel_ratio":
            ref[feature] = {"ratio": float(dist["ratio"])}
            continue
        if feature in NUMERIC_FEATURES:
            ref[feature] = {float(k): float(v) for k, v in dist.items()}
        else:
            ref[feature] = {k: float(v) for k, v in dist.items()}
    return ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Bach-style grading function benchmark (approximation of arXiv:2006.13329).")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("jsb_chorales_in_c/valid"),
        help="Directory of Bach chorale CSVs for reference distributions (default: jsb_chorales_in_c/valid).",
    )
    parser.add_argument("--target-dir", type=Path, required=True, help="Directory of generated chorales (or any chorales) to score.")
    parser.add_argument("--cache-reference", type=Path, default=None, help="Optional JSON path for saving/loading reference distributions.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save per-file scores as JSON.")
    args = parser.parse_args()

    if args.cache_reference and args.cache_reference.exists():
        ref = deserialize_reference(args.cache_reference.read_text())
    else:
        ref = compute_reference_from_dir(args.reference_dir)
        if args.cache_reference:
            args.cache_reference.write_text(serialize_reference(ref))

    rows = []
    for p in iter_csvs(args.target_dir):
        feats = chorale_features(load_chorale_csv(p))
        score = score_one(feats, ref)
        rows.append((p, score))

    if not rows:
        raise ValueError(f"No target chorales found under {args.target_dir}")

    rows.sort(key=lambda x: x[1]["overall_grade"])
    print("file,overall,pitch,rhythm,intS,intA,intT,intB,harmony,parallel,repeated")
    for p, s in rows:
        print(
            f"{p},{s['overall_grade']:.4f},{s['pitch']:.4f},{s['rhythm']:.4f},"
            f"{s['interval_S']:.4f},{s['interval_A']:.4f},{s['interval_T']:.4f},{s['interval_B']:.4f},"
            f"{s['harmonic_quality']:.4f},{s['parallel_errors']:.4f},{s['repeated_sequences']:.4f}"
        )

    if args.output_json:
        args.output_json.write_text(json.dumps({str(p): s for p, s in rows}, indent=2))


if __name__ == "__main__":
    main()
