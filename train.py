#!/usr/bin/env python3
"""
Push-up Form Classifier — Training Script
==========================================
Dataset: 2191 samples  |  BAD=1304  GOOD=887  (1.47:1)

Strategy:
  - 10% stratified holdout test set (locked away until final eval)
  - 5-fold stratified CV on the remaining 90% for model selection
  - RandomizedSearchCV hyperparameter tuning for top candidates
  - Final evaluation on the untouched test set

Usage:
  python train.py
  python train.py --dataset my_data.csv
  python train.py --no-plots
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_validate,
    RandomizedSearchCV, train_test_split,
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

PLOTS_DIR    = "training_plots"
MODEL_FILE   = "model.pkl"
REPORT_FILE  = "training_report.json"
CV_FOLDS     = 5
TEST_SIZE    = 0.10
RANDOM_STATE = 42


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in ("frame_path", "label", "label_int")]

    X = df[feat_cols].copy().astype(float)
    y = df["label_int"].values.astype(int)

    # Replace sentinels with NaN
    X.replace([-1.0, -999.0], np.nan, inplace=True)

    n_good = int(y.sum())
    n_bad  = int((y == 0).sum())
    print(f"Dataset : {len(df)} rows  |  GOOD={n_good}  BAD={n_bad}  "
          f"(ratio {n_bad/n_good:.2f}:1)")

    # Report high-missing features
    miss_pct = X.isna().mean().sort_values(ascending=False)
    high = miss_pct[miss_pct > 0.05]
    if not high.empty:
        print("High-missing features (>5%):")
        for col, pct in high.items():
            print(f"  {col:<40} {pct*100:.1f}%")

    return X, y, feat_cols


# ─── Pipelines ────────────────────────────────────────────────────────────────

def skl_pipe(clf):
    """Median imputation → StandardScaler → classifier."""
    return Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf",   clf),
    ])


def get_models(n_bad: int, n_good: int) -> dict:
    ratio = n_bad / max(n_good, 1)
    models = {}

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    models["RandomForest"] = skl_pipe(
        RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    models["GradientBoosting"] = skl_pipe(
        GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )
    )

    models["SVM"] = skl_pipe(
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )
    )

    models["MLP"] = skl_pipe(
        MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            alpha=1e-3,              # L2 regularisation
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )
    )

    return models


# ─── Hyperparameter search grids ──────────────────────────────────────────────

SEARCH_GRIDS = {}

if HAS_XGB:
    SEARCH_GRIDS["XGBoost"] = {
        "n_estimators":   [300, 500, 700],
        "max_depth":      [3, 4, 5, 6],
        "learning_rate":  [0.01, 0.05, 0.1],
        "subsample":      [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

SEARCH_GRIDS["RandomForest"] = {
    "clf__n_estimators":   [200, 400, 600],
    "clf__max_depth":      [None, 10, 20],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features":   ["sqrt", "log2"],
}

SEARCH_GRIDS["SVM"] = {
    "clf__C":     [0.1, 1, 10, 50, 100],
    "clf__gamma": ["scale", "auto", 0.001, 0.01],
}


# ─── CV evaluation ────────────────────────────────────────────────────────────

def cv_evaluate(models: dict, X_train: pd.DataFrame, y_train: np.ndarray,
                feat_cols: list) -> dict:
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print(f"\n{'─'*65}")
    print(f"  {'Model':<18}  {'F1-macro':>9}  {'AUC-ROC':>8}  {'Accuracy':>9}")
    print(f"{'─'*65}")

    for name, model in models.items():
        scores = cross_validate(
            model, X_train[feat_cols], y_train,
            cv=cv,
            scoring=["f1_macro", "roc_auc", "accuracy"],
            n_jobs=1,
        )
        results[name] = {
            "f1_macro":     float(scores["test_f1_macro"].mean()),
            "f1_macro_std": float(scores["test_f1_macro"].std()),
            "auc":          float(scores["test_roc_auc"].mean()),
            "accuracy":     float(scores["test_accuracy"].mean()),
        }
        r = results[name]
        print(f"  {name:<18}  {r['f1_macro']:.4f}±{r['f1_macro_std']:.3f}"
              f"  {r['auc']:.4f}  {r['accuracy']:.4f}")

    print(f"{'─'*65}")
    best = max(results, key=lambda n: results[n]["f1_macro"])
    print(f"  Best (CV): {best}  (F1={results[best]['f1_macro']:.4f})\n")
    return results, best


# ─── Hyperparameter tuning ────────────────────────────────────────────────────

def tune(model_name: str, model, X_train: pd.DataFrame, y_train: np.ndarray,
         feat_cols: list):
    grid = SEARCH_GRIDS.get(model_name)
    if grid is None:
        print(f"  No search grid for {model_name}, skipping tuning.")
        return model

    print(f"  Tuning {model_name} (RandomizedSearchCV, 30 iterations) …")
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        model, grid,
        n_iter=30,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train[feat_cols], y_train)
    print(f"  Best params : {search.best_params_}")
    print(f"  Best F1-macro (CV): {search.best_score_:.4f}")
    return search.best_estimator_


# ─── Test-set evaluation ──────────────────────────────────────────────────────

def test_eval(model, X_test: pd.DataFrame, y_test: np.ndarray, feat_cols: list,
              model_name: str):
    y_pred  = model.predict(X_test[feat_cols])
    y_proba = model.predict_proba(X_test[feat_cols])[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    f1      = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'='*65}")
    print(f"  HELD-OUT TEST SET RESULTS  ({len(y_test)} samples, never seen during training)")
    print(f"{'='*65}")
    print(f"  Model    : {model_name}")
    print(f"  F1-macro : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Accuracy : {(y_pred == y_test).mean():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['BAD','GOOD'])}")

    return {"f1_macro": f1, "auc": auc, "accuracy": float((y_pred == y_test).mean()),
            "y_pred": y_pred, "y_proba": y_proba}


# ─── Plots ────────────────────────────────────────────────────────────────────

def make_plots(model, X_test: pd.DataFrame, y_test: np.ndarray, feat_cols: list,
               model_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    y_pred  = model.predict(X_test[feat_cols])
    y_proba = model.predict_proba(X_test[feat_cols])[:, 1]

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred), display_labels=["BAD", "GOOD"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name} — Test Set Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"{model_name} — ROC  (AUC={roc_auc_score(y_test, y_proba):.3f})  [Test Set]")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # Feature importances
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
        top = imp.head(20)
        fig, ax = plt.subplots(figsize=(9, 6))
        top.sort_values().plot(kind="barh", ax=ax, color="#89b4fa")
        ax.set_title(f"{model_name} — Top 20 Feature Importances")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "feature_importances.png"), dpi=150)
        plt.close(fig)
        print(f"\n  Top 10 features:")
        for feat, val in imp.head(10).items():
            print(f"    {feat:<42} {val:.4f}")

    print(f"  Plots saved → {out_dir}/")


# ─── Feedback rules ───────────────────────────────────────────────────────────

FEEDBACK_RULES = [
    ("angle_elbow_left",    "lt", 100,  "Left arm not fully extended at the top — push all the way up."),
    ("angle_elbow_right",   "lt", 100,  "Right arm not fully extended at the top — push all the way up."),
    ("angle_elbow_left",    "gt", 170,  "Left elbow hyperextending — keep a slight bend at the top."),
    ("angle_elbow_right",   "gt", 170,  "Right elbow hyperextending — keep a slight bend at the top."),
    ("angle_body_left",     "lt", 160,  "Body line is off — hips may be sagging or piking."),
    ("angle_body_right",    "lt", 160,  "Body line is off — hips may be sagging or piking."),
    ("hip_deviation_left",  "lt", -0.08,"Hips sagging below plank line — engage your core."),
    ("hip_deviation_right", "lt", -0.08,"Hips sagging below plank line — engage your core."),
    ("hip_deviation_left",  "gt",  0.12,"Hips piked upward — lower them into a straight line."),
    ("hip_deviation_right", "gt",  0.12,"Hips piked upward — lower them into a straight line."),
    ("head_drop_norm",      "gt",  0.4, "Head dropping too far forward — keep your neck neutral."),
]

def generate_feedback(features: dict) -> list:
    tips, seen = [], set()
    for feat, direction, threshold, msg in FEEDBACK_RULES:
        val = features.get(feat)
        if val is None or val < -0.5:
            continue
        if ((direction == "lt" and val < threshold) or
                (direction == "gt" and val > threshold)) and msg not in seen:
            tips.append(msg)
            seen.add(msg)
    return tips


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="dataset.csv")
    parser.add_argument("--model-out", default=MODEL_FILE)
    parser.add_argument("--no-tune",   action="store_true", help="Skip hyperparam search")
    parser.add_argument("--no-plots",  action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"ERROR: {args.dataset} not found. Run generate_dataset.py first.")
        sys.exit(1)

    # ── Load & split ──────────────────────────────────────────────────────────
    X, y, feat_cols = load_dataset(args.dataset)
    n_good = int(y.sum())
    n_bad  = int((y == 0).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nSplit  →  train: {len(y_train)}  |  test: {len(y_test)} (held-out, locked)")
    print(f"Train  :  GOOD={y_train.sum()}  BAD={(y_train==0).sum()}")
    print(f"Test   :  GOOD={y_test.sum()}  BAD={(y_test==0).sum()}")

    # ── CV model selection on TRAIN only ─────────────────────────────────────
    models = get_models(n_bad, n_good)
    print(f"\n[1/3] {CV_FOLDS}-fold CV on training set …")
    cv_results, best_name = cv_evaluate(models, X_train, y_train, feat_cols)

    # ── Hyperparameter tuning on TRAIN only ───────────────────────────────────
    best_model = models[best_name]
    if not args.no_tune:
        print(f"[2/3] Hyperparameter tuning for {best_name} …")
        best_model = tune(best_name, best_model, X_train, y_train, feat_cols)
    else:
        print(f"[2/3] Skipping tuning (--no-tune).  Fitting {best_name} on full train set …")
        best_model.fit(X_train[feat_cols], y_train)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("[3/3] Evaluating on held-out test set …")
    test_results = test_eval(best_model, X_test, y_test, feat_cols, best_name)

    # ── Refit on ALL data for the saved model ─────────────────────────────────
    print("Refitting on full dataset for deployment model …")
    best_model.fit(X[feat_cols], y)

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "model":        best_model,
        "feature_cols": feat_cols,
        "model_name":   best_name,
        "cv_results":   cv_results,
        "test_f1":      test_results["f1_macro"],
        "test_auc":     test_results["auc"],
        "test_accuracy":test_results["accuracy"],
    }
    with open(args.model_out, "wb") as f:
        pickle.dump(payload, f)

    report = {
        "best_model":    best_name,
        "n_train":       int(len(y_train)),
        "n_test":        int(len(y_test)),
        "n_good":        n_good,
        "n_bad":         n_bad,
        "cv_f1_macro":   cv_results[best_name]["f1_macro"],
        "cv_auc":        cv_results[best_name]["auc"],
        "test_f1_macro": test_results["f1_macro"],
        "test_auc":      test_results["auc"],
        "test_accuracy": test_results["accuracy"],
        "all_cv_models": cv_results,
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel  saved → {args.model_out}")
    print(f"Report saved → {REPORT_FILE}")

    # ── Plots on test set ─────────────────────────────────────────────────────
    if HAS_MPL and not args.no_plots:
        make_plots(best_model, X_test, y_test, feat_cols, best_name, PLOTS_DIR)

    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"  CV  F1-macro : {cv_results[best_name]['f1_macro']:.4f}")
    print(f"  CV  AUC-ROC  : {cv_results[best_name]['auc']:.4f}")
    print(f"  TEST F1-macro: {test_results['f1_macro']:.4f}  ← on unseen data")
    print(f"  TEST AUC-ROC : {test_results['auc']:.4f}  ← on unseen data")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
