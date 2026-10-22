# Differentiation of canine hepatocellular adenoma and carcinoma from multiphase CT

Custom analysis code accompanying the manuscript:

> **Machine Learning-Based Multiphase CT Radiomics for Differentiating
> Hepatocellular Carcinoma from Hepatocellular Adenoma in Dogs**
>
> Mihyun Choi, Jimin Yoo, Seungheon Chae, Hee-Jung Jee, Byungjin Kim,
> Sunghoon Jeon, Gyu Sang Yoo, Dongwoo Chang, Ahnryul Choi\*, Namsoon Lee\*
> *[JOURNAL]* (2026). doi:[ARTICLE DOI]
>
> \* Corresponding authors: Ahnryul Choi (achoi@chungbuk.ac.kr),
> Namsoon Lee (ultravet@cbnu.ac.kr)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete custom pipeline used to classify canine
hepatocellular adenoma (HCA) versus hepatocellular carcinoma (HCC) from
triple-phase contrast-enhanced CT in a cohort of 75 dogs (45 HCC, 30 HCA):
SHAP-based feature ranking followed by a sequential Top-N feature-selection
sweep and six-model comparison under stratified cross-validation.

---

## Table of contents

- [What is and is not included](#what-is-and-is-not-included)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running without the clinical data](#running-without-the-clinical-data)
- [Repository layout](#repository-layout)
- [Execution order](#execution-order)
- [Expected input data format](#expected-input-data-format)
- [Reproducibility](#reproducibility)
- [Software environment](#software-environment)
- [Data availability](#data-availability)
- [Citation](#citation)
- [License](#license)

---

## What is and is not included

**Included** — all custom code required to reproduce the analyses reported in
the manuscript: multiphase feature preprocessing, SHAP-based feature ranking,
and the sequential Top-N feature-selection and six-model comparison framework,
together with the out-of-fold probability export used to build the reported ROC
curves.

**Not included** — the imaging and clinical data. These derive from clinical
veterinary patient records and cannot be redistributed. See
[Data availability](#data-availability). A
[synthetic data generator](#running-without-the-clinical-data) is provided so
that the radiomics pipeline can still be executed end-to-end.

Radiomic feature *extraction* itself was performed outside this repository:
hepatic masses were manually segmented in **3D Slicer 5.2.1**
(http://www.slicer.org) and **1,316 radiomic features** per phase were
extracted with its PyRadiomics extension (documentation version 2.1.2) using
intensity normalisation, 1 x 1 x 1 mm resampling, a fixed bin width of 25, and
LoG sigma values of 1.0-5.0. This repository takes the resulting feature
workbook as its input.

---

## Installation

```bash
git clone https://github.com/[USER]/dogct-hca-hcc.git
cd dogct-hca-hcc

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` pins the exact version of every package used to produce the
published results (see [Software environment](#software-environment)).

**GPU vs CPU.** `requirements.txt` lists `torch==2.8.0` / `torchvision==0.23.0`.
The published deep-learning results were produced with the CUDA 12.8 build. To
install that build explicitly:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only use, substitute `--index-url https://download.pytorch.org/whl/cpu`.
The radiomics branch does not require a GPU; the imaging branch will run on CPU
but is substantially slower.

---

## Configuration

No absolute paths are hard-coded. Every machine-specific path and the main
hyperparameters live in a configuration file:

```bash
cp config.example.yaml config.yaml
# then edit config.yaml for your machine
```

`config.yaml` is listed in `.gitignore`, so local paths are never committed.

Any setting can also be overridden by an environment variable with a `DOGCT_`
prefix, which takes precedence over the file:

```bash
DOGCT_FEATURES_XLSX=/path/to/features.xlsx python radiomics/run_phase_shap_topn.py
```

Resolution order is: environment variable → `config.yaml` →
`config.example.yaml` → in-code default.

---

## Running without the clinical data

The radiomics pipeline can be exercised end-to-end on synthetic data having
exactly the same schema as the study workbook:

```bash
# 1. generate a synthetic workbook (75 cases x 3 phases, as in the study)
python examples/make_synthetic_example.py

# 2. run the full pipeline against it
DOGCT_FEATURES_XLSX=examples/synthetic_features.xlsx \
    python radiomics/run_phase_shap_topn.py
```

This verifies that the installation is correct and that the pipeline runs to
completion. Use `--n-features-per-phase` to control runtime; the default of 40
finishes in a few minutes, whereas the real data has 1,327 features per phase.

> **The synthetic data carry only a weak artificial signal. The resulting AUC
> and accuracy values are meaningless and must not be compared with the
> published results.**

---

## Repository layout

```
.
├── CITATION.cff                    citation metadata
├── LICENSE                         MIT
├── README.md
├── config.example.yaml             template configuration
├── dogct_config.py                 configuration loader
├── requirements.txt                pinned dependencies
├── radiomics/
│   └── run_phase_shap_topn.py      the analysis pipeline
└── examples/
    └── make_synthetic_example.py   synthetic data generator
```

---

## Execution order

| Command | Produces |
|---|---|
| `python radiomics/run_phase_shap_topn.py` | SHAP rankings, Top-N metrics, ROC curves, out-of-fold predictions |

This is the single entry point of the analysis. For each of `arterial`, `delayed`, `portal`, and
`all` (the three phases concatenated) it:

1. consolidates the three-row Excel header into single feature names and
   appends a `__<phase>` suffix to each feature;
2. drops near-constant features (`VarianceThreshold`, threshold `1e-4`);
3. ranks features by mean absolute SHAP value, aggregated across 5 stratified
   CV folds of a LightGBM model;
4. sweeps Top-N feature subsets for N = 1…min(50, n_features) in steps of 1,
   evaluating all six classifiers — logistic regression, random forest,
   LightGBM, XGBoost, CatBoost, and histogram-based gradient boosting — under
   5-fold stratified CV with class-imbalance handling;
5. writes per-fold metrics (AUC, accuracy, sensitivity, specificity, F1,
   confusion-matrix counts), the out-of-fold probabilities used to build the
   reported ROC curves, and the corresponding plots.

Outputs are written to `<radiomics_outdir>_<YYYYMMDD>/phase_<phase>/`.


---

## Expected input data format

An Excel workbook with **three header rows** followed by one row per case:

| Row | Content | Example |
|---|---|---|
| 1 | feature group | `original`, `wavelet-LLH` |
| 2 | feature subgroup | `Image-original`, `glcm` |
| 3 | feature name; first two columns are the metadata keys | `case`, `tumor`, `Mean` |

- Column 1 `case` — case identifier; column 2 `tumor` — outcome label.
  Labels may be coded `{1, 2}` or `{0, 1}`; `{1, 2}` is binarised as
  `tumor == 2 → 1`. Set `meta_cols` in the config if you have a different
  number of leading metadata columns.
- Feature columns must be an exact multiple of the number of phases and
  **ordered phase-block-wise**: all arterial features, then all delayed, then
  all portal, matching the `phases` list in the config. If the count is not
  divisible by the number of phases the script emits a warning and skips the
  phase-suffix renaming, which will silently invalidate the per-phase analysis
  — check for that warning.
- Non-numeric cells are coerced to `NaN` and filled with `0`.

The study data were 75 cases x 3,981 columns, i.e. 1,327 per phase: the 1,316
radiomic features reported in the manuscript plus 11 PyRadiomics `diagnostics`
columns, which are retained in the workbook and removed downstream by the
variance filter.


---

## Reproducibility

The analysis is deterministic. A single global seed of **42** is applied to
NumPy, to every `StratifiedKFold` split, and to the `random_state` of all six
classifiers, matching the seed stated in the manuscript Methods. Re-running
`radiomics/run_phase_shap_topn.py` on the same input workbook reproduces the
reported SHAP rankings and performance metrics exactly.

Note that `shap`, `lightgbm`, `xgboost`, and `catboost` must match the pinned
versions in `requirements.txt`; tree-ensemble implementations change across
releases and newer versions may alter the SHAP ranking slightly.

## Software environment

Results were produced with **Python 3.13.5** on Linux.

| Package | Version | | Package | Version |
|---|---|---|---|---|
| numpy | 2.1.3 | | shap | 0.50.0 |
| pandas | 2.2.3 | | sdv | 1.36.1 |
| scipy | 1.15.3 | | ctgan | 0.12.1 |
| scikit-learn | 1.8.0 | | torch | 2.8.0 (cu128) |
| lightgbm | 4.6.0 | | torchvision | 0.23.0 (cu128) |
| xgboost | 3.1.2 | | pydicom | 3.0.1 |
| catboost | 1.2.8 | | opencv-python | 4.12.0 |
| matplotlib | 3.10.0 | | openpyxl | 3.1.5 |

See `requirements.txt` for the canonical, installable list.

---

## Data availability

The CT images and clinical records underlying this study derive from clinical
veterinary patient records and are not publicly redistributable. Requests for
access should be directed to the corresponding author and are subject to
institutional approval and an appropriate data-sharing agreement. Refer to the
Data Availability statement in the manuscript for the governing conditions.

Do not commit clinical data to this repository. `.gitignore` is configured to
exclude Excel workbooks, DICOM files, manifests, model checkpoints, and run
outputs, because manifest files in particular may embed patient- or
institution-identifying filesystem paths.

---

## Citation

If you use this code, please cite both the article and the archived software.
`CITATION.cff` in this repository carries machine-readable metadata.

```bibtex
@software{dogct_hca_hcc,
  author    = {[AUTHORS]},
  title     = {Differentiation of canine hepatocellular adenoma and
               carcinoma from multiphase CT: analysis code},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/[USER]/dogct-hca-hcc}
}
```

---

## License

Released under the [MIT License](LICENSE).

Third-party dependencies retain their own licenses. Note that the
EfficientNet-B2 weights used for initialisation are the torchvision
`IMAGENET1K_V1` weights and are subject to their original terms.
