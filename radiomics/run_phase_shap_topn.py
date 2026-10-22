#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 16:44:10 2026

@author: medict
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase별(arterial/delayed/portal)
- LGBM(GBDT)로 5-fold CV SHAP(mean(|SHAP|)) 랭킹 산출
- phase별 SHAP Top-20 콘솔 출력 + CSV 저장
- SHAP Top-N feature selection 후 모델 6개(LR, RF, LGBM, XGBoost, CatBoost, HGB) 성능 비교(5-fold CV)
- AUC/ACC mean±std 저장 + ROC/Trend plot 저장
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dogct_config as cfg

import os, pickle, json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
import lightgbm as lgb
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix

# ===== 옵션 모델(설치 여부 체크) =====
HAS_XGB = HAS_CAT = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# =========================
# 설정
# =========================
SEED = 42
np.random.seed(SEED)

DATE_TAG = datetime.now().strftime("%Y%m%d")
FILE_PATH = str(cfg.require_path("features_xlsx"))
META_COLS = 2
PHASES = ["arterial", "delayed", "portal"]

#TOPN_LIST = [3, 5, 10, 20, 40, 60, 80, 100]
TOPN_MAX = 50   # 원하는 최대 Top-N (예: 200, 300, 500 등)
N_SPLITS = 5

OUTDIR = f'{cfg.get("radiomics_outdir", "results/radiomics")}_{DATE_TAG}'
os.makedirs(OUTDIR, exist_ok=True)

assert os.path.exists(FILE_PATH), f"File not found: {FILE_PATH}"
try:
    from tqdm.auto import tqdm
    USE_TQDM = True
except Exception:
    USE_TQDM = False
    def tqdm(x, **kwargs):  # fallback
        return x
# =========================
# 유틸
# =========================
def get_phase_cols(X_all: pd.DataFrame, phase: str, phases: list) -> list:
    """
    phase == "all" 이면 arterial/delayed/portal 모든 feature를 합쳐서 반환
    그 외에는 해당 phase suffix(__phase)만 반환
    """
    if phase == "all":
        suffixes = tuple([f"__{ph}" for ph in phases])
        return [c for c in X_all.columns if str(c).endswith(suffixes)]
    else:
        return [c for c in X_all.columns if str(c).endswith(f"__{phase}")]
    
def merge_3row_header(excel_path: str, header_rows: int = 3) -> pd.DataFrame:
    df_raw = pd.read_excel(excel_path, header=None)
    hdr = df_raw.iloc[0:header_rows, :]
    feature_names = hdr.apply(lambda col: "_".join(col.dropna().astype(str)), axis=0)
    data = df_raw.iloc[header_rows:, :].reset_index(drop=True)
    data.columns = feature_names
    return data

def apply_phase_suffix(data: pd.DataFrame, meta_cols: int, phases) -> pd.DataFrame:
    data = data.copy()
    meta = list(data.columns[:meta_cols])
    feats = list(data.columns[meta_cols:])
    k = len(phases)
    n = len(feats)
    if n % k == 0 and n > 0:
        block = n // k
        base = list(pd.Index(feats[:block]).map(str))
        new_feats = []
        for ph in phases:
            new_feats += [f"{bn}__{ph}" for bn in base]
        data.columns = meta + new_feats
    else:
        print(f"[WARN] Feature count {n} not divisible by {k}; phase renaming skipped.")
    return data

def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    cols = pd.Series(df.columns)
    out = df.copy()
    out.columns = (cols + "_" + cols.groupby(cols).cumcount().astype(str)).values
    return out

def find_target_column(df: pd.DataFrame) -> str:
    if "tumor" in df.columns:
        return "tumor"
    cand = [c for c in df.columns if str(c).lower().startswith("tumor")]
    if cand:
        return cand[0]
    raise KeyError("Target column not found. Expected 'tumor' or columns starting with 'tumor*'.")

def binarize_target(y_raw: np.ndarray) -> np.ndarray:
    uvals = np.sort(pd.unique(y_raw))
    if set(uvals) == {1, 2}:
        return (y_raw == 2).astype(int)
    if set(uvals) == {0, 1}:
        return y_raw.astype(int)
    if len(uvals) == 2:
        return (y_raw == uvals[-1]).astype(int)
    raise ValueError(f"Target has {len(uvals)} unique values {uvals}; need binary.")

def build_models(seed: int, y: np.ndarray):
    """
    요청 모델: LR, RF, LGBM, XGBoost, CatBoost, HGB
    (XGBoost/CatBoost 미설치 시 자동 스킵)
    """
    models = {
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced",
                                       solver="lbfgs", random_state=seed))
        ]),
        "RF": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=500, random_state=seed,
                class_weight="balanced", n_jobs=-1))
        ]),
        "LGBM": Pipeline([
            ("clf", lgb.LGBMClassifier(
                boosting_type="gbdt",
                n_estimators=600, learning_rate=0.03,
                num_leaves=64, min_child_samples=10,
                subsample=0.9, colsample_bytree=0.8,
                random_state=seed, is_unbalance=True,
                n_jobs=-1, verbosity=-1))
        ]),
        "HGB": Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=600, learning_rate=0.05, random_state=seed))
        ]),
    }

    # XGBoost
    if HAS_XGB:
        pos = int(y.sum()); neg = int(len(y) - pos)
        spw = float(neg / max(pos, 1))
        models["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=900, max_depth=6, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.8,
                tree_method="hist",
                reg_lambda=1.0, reg_alpha=0.0,
                random_state=seed, n_jobs=-1,
                eval_metric="logloss",
                scale_pos_weight=spw,
                verbosity=0
            ))
        ])
    else:
        print("[WARN] XGBoost not installed -> skipping XGBoost model")

    # CatBoost
    if HAS_CAT:
        models["CatBoost"] = Pipeline([
            ("clf", CatBoostClassifier(
                iterations=2500, depth=6, learning_rate=0.03,
                loss_function="Logloss", eval_metric="AUC",
                random_seed=seed,
                verbose=False,
                auto_class_weights="Balanced"
            ))
        ])
    else:
        print("[WARN] CatBoost not installed -> skipping CatBoost model")

    return models

def cv_shap_ranking_lgbm(X_df: pd.DataFrame, y: np.ndarray, n_splits=5, seed=42) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    agg = np.zeros(X_df.shape[1], dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_df, y), start=1):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr = y[tr_idx]

        model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            n_estimators=600, learning_rate=0.03,
            num_leaves=64, min_child_samples=10,
            subsample=0.9, colsample_bytree=0.8,
            random_state=seed, is_unbalance=True,
            n_jobs=-1, verbosity=-1
        )
        model.fit(X_tr, y_tr)

        expl = shap.TreeExplainer(model)
        phi = expl.shap_values(X_va)
        if isinstance(phi, list):
            phi = phi[1] if len(phi) > 1 else phi[0]

        agg += np.abs(phi).mean(axis=0)

    agg /= n_splits
    shap_df = pd.DataFrame({
        "feature": X_df.columns.to_numpy(),
        "mean_abs_shap": agg
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return shap_df
def evaluate_models_topn(
    X: pd.DataFrame,
    y: np.ndarray,
    shap_rank: pd.DataFrame,
    topN_list,
    models: dict,
    n_splits=5,
    seed=42,
    save_oof=True,   # ✅ 추가
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    model_topn_results = {}
    rows = []
    fold_rows = []

    def _safe_div(a, b):
        return float(a / b) if b != 0 else np.nan

    def _mean_std(arr):
        arr = [v for v in arr if not np.isnan(v)]
        if len(arr) == 0:
            return np.nan, 0.0
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return mean, std

    model_iter = tqdm(models.items(), desc="Models", total=len(models)) if USE_TQDM else models.items()

    for model_name, pipe in model_iter:
        model_topn_results[model_name] = {}

        topn_iter = tqdm(topN_list, desc=f"{model_name}: TopN", leave=False) if USE_TQDM else topN_list

        for top_n in topn_iter:
            selected = shap_rank.head(top_n)["feature"].tolist()
            X_top = X[selected].to_numpy()

            # ✅ OOF 저장용 (전체 샘플 길이)
            oof_prob = np.full(len(y), np.nan, dtype=float)
            oof_fold = np.full(len(y), -1, dtype=int)

            fold_aucs, fold_accs = [], []
            fold_sens, fold_spec, fold_f1s = [], [], []

            for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_top, y), start=1):
                X_tr, X_va = X_top[tr_idx], X_top[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                pipe.fit(X_tr, y_tr)

                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(X_va)[:, 1]
                elif hasattr(pipe, "decision_function"):
                    proba = MinMaxScaler().fit_transform(
                        pipe.decision_function(X_va).reshape(-1, 1)
                    ).ravel()
                else:
                    proba = pipe.predict(X_va).astype(float)

                pred = (proba > 0.5).astype(int)

                # ✅ OOF에 기록
                oof_prob[va_idx] = proba
                oof_fold[va_idx] = fold_i

                # AUC/ACC
                try:
                    auc_fold = roc_auc_score(y_va, proba)
                except ValueError:
                    auc_fold = np.nan
                acc_fold = accuracy_score(y_va, pred)

                # Sens/Spec + confusion
                cm = confusion_matrix(y_va, pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sens = _safe_div(tp, tp + fn)
                    spec = _safe_div(tn, tn + fp)
                else:
                    tn = fp = fn = tp = np.nan
                    sens = np.nan
                    spec = np.nan

                f1 = f1_score(y_va, pred, zero_division=0)

                fold_aucs.append(auc_fold)
                fold_accs.append(acc_fold)
                fold_sens.append(sens)
                fold_spec.append(spec)
                fold_f1s.append(f1)

                fold_rows.append({
                    "model": model_name, "topN": top_n, "fold": fold_i,
                    "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
                    "pos_train": int(y_tr.sum()), "pos_val": int(y_va.sum()),
                    "auc": float(auc_fold) if not np.isnan(auc_fold) else np.nan,
                    "acc": float(acc_fold),
                    "sens": float(sens) if not np.isnan(sens) else np.nan,
                    "spec": float(spec) if not np.isnan(spec) else np.nan,
                    "f1": float(f1),
                    "tn": float(tn) if not np.isnan(tn) else np.nan,
                    "fp": float(fp) if not np.isnan(fp) else np.nan,
                    "fn": float(fn) if not np.isnan(fn) else np.nan,
                    "tp": float(tp) if not np.isnan(tp) else np.nan,
                })

            # mean/std
            auc_mean, auc_std = _mean_std(fold_aucs)
            acc_mean, acc_std = _mean_std(fold_accs)
            sens_mean, sens_std = _mean_std(fold_sens)
            spec_mean, spec_std = _mean_std(fold_spec)
            f1_mean, f1_std = _mean_std(fold_f1s)

            # ✅ OOF 기반 ROC (thresholds까지 저장!)
            valid = ~np.isnan(oof_prob)
            y_oof = y[valid]
            p_oof = oof_prob[valid]
            try:
                fpr, tpr, thr = roc_curve(y_oof, p_oof)
            except ValueError:
                fpr, tpr, thr = np.array([0, 1]), np.array([0, 1]), np.array([np.inf, 0.0])

            key = f"Top{top_n}"
            model_topn_results[model_name][key] = {
                "selected_features": selected,

                "auc_mean": auc_mean, "auc_std": auc_std,
                "acc_mean": acc_mean, "acc_std": acc_std,
                "sens_mean": sens_mean, "sens_std": sens_std,
                "spec_mean": spec_mean, "spec_std": spec_std,
                "f1_mean": f1_mean, "f1_std": f1_std,

                "fold_auc": fold_aucs,
                "fold_acc": fold_accs,
                "fold_sens": fold_sens,
                "fold_spec": fold_spec,
                "fold_f1": fold_f1s,

                # ✅ ROC 재현 핵심 데이터
                "fpr": fpr, "tpr": tpr, "thr": thr,
            }

            if save_oof:
                model_topn_results[model_name][key]["oof_prob"] = oof_prob
                model_topn_results[model_name][key]["oof_fold"] = oof_fold

            rows.append({
                "model": model_name, "topN": top_n,
                "auc_mean": auc_mean, "auc_std": auc_std,
                "acc_mean": acc_mean, "acc_std": acc_std,
                "sens_mean": sens_mean, "sens_std": sens_std,
                "spec_mean": spec_mean, "spec_std": spec_std,
                "f1_mean": f1_mean, "f1_std": f1_std
            })

    df_metrics = pd.DataFrame(rows).sort_values(["model", "topN"]).reset_index(drop=True)
    df_folds = pd.DataFrame(fold_rows).sort_values(["model", "topN", "fold"]).reset_index(drop=True)
    return model_topn_results, df_metrics, df_folds



def plot_best_roc_per_model(model_topn_results, phase_tag, outdir):
    plt.figure(figsize=(8, 6))
    for m, items in model_topn_results.items():
        best_key = max(
            items.keys(),
            key=lambda k: (items[k]["auc_mean"] if not np.isnan(items[k]["auc_mean"]) else -1)
        )
        fpr, tpr = items[best_key]["fpr"], items[best_key]["tpr"]
        auc_m, acc_m = items[best_key]["auc_mean"], items[best_key]["acc_mean"]
        plt.plot(fpr, tpr, label=f"{m} ({best_key}, AUC={auc_m:.3f}, ACC={acc_m:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"[{phase_tag}] Best TopN ROC per Model")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(outdir, f"{phase_tag}_bestROC_per_model.png")
    plt.savefig(path, dpi=150)
    plt.show()
    return path

def plot_trend(model_topn_results, topN_list, metric_key, phase_tag, outdir):
    plt.figure(figsize=(8, 6))
    for m in model_topn_results.keys():
        xs = topN_list
        ys = [model_topn_results[m][f"Top{n}"][metric_key] for n in xs]
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("Top-N features")
    plt.ylabel(metric_key.replace("_", " ").upper())
    plt.title(f"[{phase_tag}] {metric_key.upper()} vs Top-N")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(outdir, f"{phase_tag}_{metric_key}_vs_topN.png")
    plt.savefig(path, dpi=150)
    plt.show()
    return path

# =========================
# 1) 데이터 로드/전처리
# =========================
data = merge_3row_header(FILE_PATH, header_rows=3)
data = apply_phase_suffix(data, meta_cols=META_COLS, phases=PHASES)
data = dedup_columns(data)

data = data.apply(pd.to_numeric, errors="coerce").fillna(0)

target_col = find_target_column(data)
y_raw = data[target_col].to_numpy()
y = binarize_target(y_raw)

X_all = data.iloc[:, META_COLS:].copy()
X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"[INFO] target_col={target_col}, y_pos={int(y.sum())}/{len(y)}")
print(f"[INFO] X_all shape={X_all.shape}")

# 모델 구성(요청 6개)
models = build_models(SEED, y)
print("[INFO] Using models:", list(models.keys()))

# =========================
# 2) Phase별 실행
# =========================
all_phase_summary = []

PHASES_RUN = PHASES + ["all"]   # ✅ arterial/delayed/portal + all(합본)
for phase in PHASES_RUN:   # ✅ all 포함해서 loop
    phase_tag = f"phase_{phase}" 
    phase_dir = os.path.join(OUTDIR, phase_tag)
    os.makedirs(phase_dir, exist_ok=True)

    # phase feature만 추출
    phase_cols = get_phase_cols(X_all, phase, PHASES)
    if len(phase_cols) == 0:
        print(f"[WARN] No features found for {phase_tag}. Skipping.")
        continue

    Xp = X_all[phase_cols].copy()

    # VarianceThreshold
    selector = VarianceThreshold(threshold=1e-4)
    Xp_reduced = selector.fit_transform(Xp)
    reduced_cols = Xp.columns[selector.get_support()].to_numpy()
    Xp_reduced_df = pd.DataFrame(Xp_reduced, columns=reduced_cols)

    print(f"\n===== [{phase_tag}] =====")
    print(f"[INFO] before VT: {Xp.shape} | after VT: {Xp_reduced_df.shape}")

    # (A) SHAP 랭킹 생성
    shap_rank = cv_shap_ranking_lgbm(Xp_reduced_df, y, n_splits=N_SPLITS, seed=SEED)

    # ✅ 요청: SHAP 중요도별 Top-20 "상별로 제시"
    print(f"\n[SHAP Top-20 | {phase_tag}]")
    print(shap_rank.head(20).to_string(index=False))

    shap_rank_csv = os.path.join(phase_dir, f"{phase_tag}_shap_rank.csv")
    shap_top20_csv = os.path.join(phase_dir, f"{phase_tag}_shap_top20.csv")
    shap_rank.to_csv(shap_rank_csv, index=False)
    shap_rank.head(20).to_csv(shap_top20_csv, index=False)

    # (B) Top-N 평가 (feature 수보다 큰 topN은 자동 제외)
    maxN = min(TOPN_MAX, Xp_reduced_df.shape[1])
    valid_topN = list(range(1, maxN + 1))   # ✅ 1개 단위로 Top-N 평가
    if len(valid_topN) == 0:
        print(f"[WARN] Not enough features after VT for {phase_tag}. Skipping eval.")
        continue

    model_topn_results, df_metrics, df_folds = evaluate_models_topn(
    X=Xp_reduced_df,
    y=y,
    shap_rank=shap_rank,
    topN_list=valid_topN,
    models=models,
    n_splits=N_SPLITS,
    seed=SEED
    )

    # 저장
    pkl_path = os.path.join(phase_dir, f"{phase_tag}_model_topN_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_topn_results, f)

    metrics_csv = os.path.join(phase_dir, f"{phase_tag}_model_topN_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    folds_csv = os.path.join(phase_dir, f"{phase_tag}_model_topN_folds.csv")
    df_folds.to_csv(folds_csv, index=False)
    # 플롯
    roc_png = plot_best_roc_per_model(model_topn_results, phase_tag, phase_dir)
    auc_png = plot_trend(model_topn_results, valid_topN, "auc_mean", phase_tag, phase_dir)
    acc_png = plot_trend(model_topn_results, valid_topN, "acc_mean", phase_tag, phase_dir)

    # phase 요약(best AUC per model)
    for m, items in model_topn_results.items():
        best_key = max(items.keys(), key=lambda k: (items[k]["auc_mean"] if not np.isnan(items[k]["auc_mean"]) else -1))
        all_phase_summary.append({
            "phase": phase,
            "model": m,
            "best_topN": best_key,
            "best_auc": items[best_key]["auc_mean"],
            "best_acc": items[best_key]["acc_mean"]
        })

    print(f"\n✅ [{phase_tag}] Saved:")
    print(f"- SHAP rank   : {shap_rank_csv}")
    print(f"- SHAP top20  : {shap_top20_csv}")
    print(f"- Metrics CSV : {metrics_csv}")
    print(f"- PKL         : {pkl_path}")
    print(f"- Plots       : {roc_png}, {auc_png}, {acc_png}")
    
    
    best_oof_rows = []
    best_summary_rows = []
    
    for m, items in model_topn_results.items():
        # best_key = AUC mean이 최대인 TopN
        best_key = max(
            items.keys(),
            key=lambda k: (items[k]["auc_mean"] if not np.isnan(items[k]["auc_mean"]) else -1)
        )
        best = items[best_key]
    
        if ("oof_prob" not in best) or ("oof_fold" not in best):
            print(f"[WARN] {phase_tag} | {m} | {best_key} has no OOF data -> skip raw export")
            continue
    
        oof_prob = best["oof_prob"]
        oof_fold = best["oof_fold"]
    
        # OOF 유효한 샘플만
        valid = ~np.isnan(oof_prob)
        idxs = np.where(valid)[0]
    
        # 0.5 기준 label (원하면 threshold 바꿔도 됨)
        oof_pred = (oof_prob[valid] > 0.5).astype(int)
    
        # 모델/phase 요약도 같이 저장
        best_summary_rows.append({
            "phase": phase,
            "phase_tag": phase_tag,
            "model": m,
            "best_topN": best_key,
            "best_auc_mean": float(best["auc_mean"]) if not np.isnan(best["auc_mean"]) else np.nan,
            "best_acc_mean": float(best["acc_mean"]) if not np.isnan(best["acc_mean"]) else np.nan,
            "n_oof": int(valid.sum()),
            "selected_features": "|".join(best["selected_features"])  # 길면 빼도 됨
        })
    
        # raw row 생성
        for j, i in enumerate(idxs):
            best_oof_rows.append({
                "phase": phase,
                "model": m,
                "best_topN": best_key,          # 예: "Top3"
                "idx": int(i),                  # 원본 row index
                "fold": int(oof_fold[i]),       # 어느 fold의 val이었는지
                "y_true": int(y[i]),
                "y_prob": float(oof_prob[i]),
                "y_pred": int(oof_pred[j]),
            })
    
    # DataFrame으로 저장
    df_best_oof = pd.DataFrame(best_oof_rows).sort_values(["model", "fold", "idx"]).reset_index(drop=True)
    df_best_summary = pd.DataFrame(best_summary_rows).sort_values(["model"]).reset_index(drop=True)
    
    best_oof_csv = os.path.join(phase_dir, f"{phase_tag}_best_model_oof_raw.csv")
    best_summary_csv = os.path.join(phase_dir, f"{phase_tag}_best_model_summary.csv")
    
    df_best_oof.to_csv(best_oof_csv, index=False)
    df_best_summary.to_csv(best_summary_csv, index=False)
    
    print(f"[INFO] Saved BEST OOF RAW  : {best_oof_csv}")
    print(f"[INFO] Saved BEST SUMMARY  : {best_summary_csv}")
# =========================
# 3) 전체 요약 저장
# =========================
summary_df = pd.DataFrame(all_phase_summary)
summary_csv = os.path.join(OUTDIR, f"phase_summary_best_{DATE_TAG}.csv")
summary_df.to_csv(summary_csv, index=False)

print("\n==============================")
print("✅ ALL DONE")
print(f"- Output dir : {OUTDIR}")
print(f"- Summary CSV: {summary_csv}")
if not summary_df.empty:
    print(summary_df.sort_values(["phase", "best_auc"], ascending=[True, False]).to_string(index=False))

