#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a small SYNTHETIC radiomic feature workbook with exactly the same
schema as the real study data, so that the radiomics pipeline can be executed
and verified end-to-end without access to clinical records.

The real dataset (75 cases x 3,981 features) cannot be shared because it
derives from veterinary patient records. This generator produces structurally
identical but entirely artificial data:

    * 3 header rows  (group / subgroup / feature name)
    * column 0 = "case", column 1 = "tumor"  (labels 1 and 2)
    * feature columns laid out phase-block-wise:
      [arterial block][delayed block][portal block]

A weak, deliberately modest signal is injected into a handful of features so
that the Top-N sweep produces a non-degenerate ROC curve. The resulting AUCs
are NOT comparable to the published results and must not be interpreted.

Usage
-----
    python examples/make_synthetic_example.py
    python examples/make_synthetic_example.py --n-cases 75 --n-features-per-phase 40

Then point the pipeline at it:
    DOGCT_FEATURES_XLSX=examples/synthetic_features.xlsx \
        python radiomics/run_phase_shap_topn.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Mirrors the group/subgroup vocabulary of the real PyRadiomics export.
GROUPS = ["original", "wavelet-LLH", "wavelet-LHL", "wavelet-HLL", "log-sigma-3-0-mm"]
SUBGROUPS = ["firstorder", "glcm", "glrlm", "glszm", "shape"]
FEATURES = [
    "Mean", "Median", "Minimum", "Maximum", "Range", "Variance",
    "Skewness", "Kurtosis", "Entropy", "Energy", "Uniformity",
    "Contrast", "Correlation", "JointEnergy", "JointEntropy",
    "ClusterShade", "ClusterProminence", "Autocorrelation",
    "GrayLevelVariance", "SizeZoneVariance",
]


def build_feature_names(n_per_phase: int) -> list[tuple[str, str, str]]:
    """Return n_per_phase (group, subgroup, feature) header triples."""
    triples: list[tuple[str, str, str]] = []
    i = 0
    while len(triples) < n_per_phase:
        g = GROUPS[(i // (len(SUBGROUPS) * len(FEATURES))) % len(GROUPS)]
        s = SUBGROUPS[(i // len(FEATURES)) % len(SUBGROUPS)]
        f = FEATURES[i % len(FEATURES)]
        triples.append((g, s, f))
        i += 1
    return triples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cases", type=int, default=75,
                    help="number of synthetic cases (default: 75, matching the study)")
    ap.add_argument("--n-features-per-phase", type=int, default=40,
                    help="features per contrast phase (default: 40; real data has 1327)")
    ap.add_argument("--phases", nargs="+", default=["arterial", "delayed", "portal"])
    ap.add_argument("--n-informative", type=int, default=6,
                    help="features per phase carrying a weak synthetic signal")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "synthetic_features.xlsx")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n_cases
    n_phase = args.n_features_per_phase
    n_total = n_phase * len(args.phases)

    # Balanced-ish binary outcome, encoded 1/2 exactly as in the real workbook.
    y = rng.permutation(np.r_[np.ones(n // 2, int), np.full(n - n // 2, 2)])

    # Base noise, then inject a weak class offset into the first few features
    # of each phase block so that SHAP ranking has something to find.
    X = rng.normal(0.0, 1.0, size=(n, n_total))
    for p in range(len(args.phases)):
        offset = p * n_phase
        for k in range(min(args.n_informative, n_phase)):
            effect = rng.uniform(0.45, 0.85) * (1.0 if k % 2 == 0 else -1.0)
            X[:, offset + k] += effect * (y == 2)

    # Give features a plausible HU-like scale and a few near-constant columns
    # so that the VarianceThreshold step is actually exercised.
    X = X * rng.uniform(5.0, 150.0, size=n_total) + rng.uniform(-200.0, 800.0, size=n_total)
    for j in rng.choice(n_total, size=max(1, n_total // 20), replace=False):
        X[:, j] = 1.0

    triples = build_feature_names(n_phase)

    # --- assemble the 3-row header exactly as the loader expects ---
    row0 = ["", ""] + [t[0] for _ in args.phases for t in triples]
    row1 = ["", ""] + [t[1] for _ in args.phases for t in triples]
    row2 = ["case", "tumor"] + [t[2] for _ in args.phases for t in triples]

    body = np.column_stack([np.arange(1, n + 1), y, X])
    frame = pd.DataFrame(
        np.vstack([np.array(row0, dtype=object),
                   np.array(row1, dtype=object),
                   np.array(row2, dtype=object),
                   body.astype(object)])
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(args.out, header=False, index=False)

    print(f"[OK] wrote synthetic workbook -> {args.out}")
    print(f"     cases={n}  phases={args.phases}  features/phase={n_phase}  total={n_total}")
    print(f"     class balance: label1={int((y == 1).sum())}  label2={int((y == 2).sum())}")
    print()
    print("     Run the pipeline on it with:")
    print(f"       DOGCT_FEATURES_XLSX={args.out} python radiomics/run_phase_shap_topn.py")
    print()
    print("     NOTE: synthetic data. Resulting metrics are meaningless and")
    print("           must not be compared with the published results.")


if __name__ == "__main__":
    main()
