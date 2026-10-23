# dogct-hca-hcc

Analysis code for the study:

> Choi M, Yoo J, Chae S, Jee H-J, Kim B, Jeon S, Yoo GS, Chang D, Choi A\*, Lee N\*.
> **Machine Learning-Based Multiphase CT Radiomics for Differentiating
> Hepatocellular Carcinoma from Hepatocellular Adenoma in Dogs.**
> *[JOURNAL]* (2026). doi:[ARTICLE DOI]

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23582592.svg)](https://doi.org/10.5281/zenodo.23582592)

The script takes a table of PyRadiomics features extracted from triple-phase CT
(arterial, portal, delayed) and, for each phase and for all phases combined,
ranks features by mean |SHAP| of a LightGBM model across 5 stratified CV folds,
then evaluates six classifiers (LR, RF, LightGBM, XGBoost, CatBoost, HGB) on the
Top-N features for N = 1–50.

Segmentation and feature extraction were done in 3D Slicer 5.2.1 with the
SlicerRadiomics extension (see the Methods section of the paper for settings)
and are not part of this repository.

## Installation

```bash
git clone https://github.com/Seungheon354/dogct-hca-hcc.git
cd dogct-hca-hcc
pip install -r requirements.txt
```

Results in the paper were produced with Python 3.13.5 and the package versions
in `requirements.txt`.

## Usage

Copy `config.example.yaml` to `config.yaml` and set `features_xlsx` to the
feature workbook, then run:

```bash
python radiomics/run_phase_shap_topn.py
```

The path can also be given as an environment variable
(`DOGCT_FEATURES_XLSX=/path/to/features.xlsx`).

For each of `arterial`, `delayed`, `portal` and `all`, results are written to
`results/radiomics_<YYYYMMDD>/phase_<phase>/`:

- SHAP feature ranking (`*_shap_rank.csv`)
- mean ± SD of AUC, accuracy, sensitivity, specificity and F1 for every
  model and N (`*_model_topN_metrics.csv`), and per-fold values (`*_folds.csv`)
- out-of-fold probabilities for the best N of each model (`*_best_model_oof_raw.csv`)
- ROC curves and AUC/accuracy vs N plots

A seed of 42 is used throughout.

### Input format

An Excel file with three header rows (feature group / subgroup / name) and one
row per case. The first two columns are `case` and `tumor` (label coded 1/2 or
0/1; 2 or 1 is the positive class). The remaining columns are the features of the three
phases placed one after another in the order `arterial`, `delayed`, `portal`,
each block having the same feature columns.

### Synthetic example

The clinical data cannot be shared. To check that the pipeline runs, a random
workbook with the same layout can be generated:

```bash
python examples/make_synthetic_example.py
DOGCT_FEATURES_XLSX=examples/synthetic_features.xlsx python radiomics/run_phase_shap_topn.py
```

Results on this data have no meaning.

## Data availability

The CT images and radiomic features come from clinical patient records and are
available from the corresponding authors on reasonable request.

## Citation

Please cite the paper above and the archived code (doi:10.5281/zenodo.23582592).
See `CITATION.cff`.

## License

MIT
