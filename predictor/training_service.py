from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .model_service import FEATURES, FEATURE_LABELS, model, regressor, training_data


def _artifact(name):
    return Path(settings.BASE_DIR) / "model_artifacts" / name


def _backup(path: Path):
    if path.exists():
        backup_dir = Path(settings.BASE_DIR) / "model_artifacts" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, backup_dir / f"{path.stem}_{stamp}{path.suffix}")


def _validate(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))
    if "hit" not in df.columns and "future_streams" not in df.columns:
        raise ValueError("Your CSV must contain either 'hit' (0/1) or 'future_streams' so the training target can be known.")
    if len(df) < 30:
        raise ValueError("Please provide at least 30 labeled rows. 100+ rows is strongly recommended.")
    if df[FEATURES].isnull().any().any():
        raise ValueError("The 11 model features contain missing values. Fill them before training.")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[FEATURES].isnull().any().any():
        raise ValueError("All 11 model features must be numeric.")
    if "hit" in df.columns:
        df["hit"] = pd.to_numeric(df["hit"], errors="coerce")
        if df["hit"].isnull().any() or not set(df["hit"].astype(int).unique()).issubset({0, 1}):
            raise ValueError("'hit' must contain only 0 and 1.")
        y = df["hit"].astype(int)
        threshold = None
    else:
        df["future_streams"] = pd.to_numeric(df["future_streams"], errors="coerce")
        if df["future_streams"].isnull().any():
            raise ValueError("'future_streams' must be numeric.")
        threshold = float(df["future_streams"].quantile(0.80))
        y = (df["future_streams"] >= threshold).astype(int)
    if y.nunique() < 2:
        raise ValueError("The training target contains only one class. You need both hit=0 and hit=1 examples.")
    return df, y, threshold


def refresh_existing_predictions():
    """Re-score the website's existing song catalog with the newly trained model."""
    from .models import SongPrediction
    from .model_service import _momentum, _business_action, _future_streams

    qs = list(SongPrediction.objects.all())
    if not qs:
        return 0
    m = model()
    rows = []
    for song in qs:
        d = {f: getattr(song, f) for f in FEATURES}
        X = pd.DataFrame([d])
        p = float(m.predict_proba(X)[0, 1]) * 100
        momentum_score, category, _ = _momentum(d)
        action = _business_action(p / 100.0, momentum_score)
        future = _future_streams(X)
        song.hit_probability = round(p, 2)
        song.momentum_score = momentum_score
        song.momentum_category = category
        song.business_action = action
        if future is not None:
            song.future_streams = round(future)
        rows.append(song)
    SongPrediction.objects.bulk_update(rows, ["hit_probability", "momentum_score", "momentum_category", "business_action", "future_streams"], batch_size=500)
    return len(rows)


def train_uploaded_csv(uploaded_file, mode="combine"):
    df_new = pd.read_csv(uploaded_file)
    df_new, y_new, new_threshold = _validate(df_new)

    if mode == "combine":
        base = training_data().copy()
        # Existing project data has future_streams but no ground-truth hit column.
        base, y_base, base_threshold = _validate(base)
        combined = pd.concat([base[FEATURES + (["future_streams"] if "future_streams" in base.columns else [])], df_new[FEATURES + (["future_streams"] if "future_streams" in df_new.columns else [])]], ignore_index=True)
        if "hit" in df_new.columns:
            # If the uploaded data supplies labels, preserve them. Existing rows derive labels from future_streams.
            y_combined = pd.concat([y_base.reset_index(drop=True), y_new.reset_index(drop=True)], ignore_index=True)
        else:
            if "future_streams" not in combined.columns:
                raise ValueError("Combined training data needs a target.")
            threshold = float(combined["future_streams"].quantile(0.80))
            y_combined = (combined["future_streams"] >= threshold).astype(int)
        train_df = combined
    else:
        train_df = df_new
        y_combined = y_new
        threshold = new_threshold

    X = train_df[FEATURES].astype(float)
    y = pd.Series(y_combined).astype(int)
    if y.nunique() < 2:
        raise ValueError("Training target must contain both classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred).tolist()
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "confusion_matrix": cm,
    }

    reg_metrics = None
    reg = None
    if "future_streams" in train_df.columns:
        target = pd.to_numeric(train_df["future_streams"], errors="coerce")
        valid = target.notna() & (target >= 0)
        if valid.sum() >= 30:
            Xr = X.loc[valid]
            yr = np.log1p(target.loc[valid])
            Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.20, random_state=42)
            reg = RandomForestRegressor(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            )
            reg.fit(Xtr, ytr)
            rp = np.expm1(reg.predict(Xte))
            actual = np.expm1(yte)
            reg_metrics = {
                "mae": float(mean_absolute_error(actual, rp)),
                "rmse": float(np.sqrt(mean_squared_error(actual, rp))),
                "r2": float(r2_score(actual, rp)),
            }

    model_path = Path(settings.MODEL_PATH)
    _backup(model_path)
    joblib.dump(clf, model_path)
    if reg is not None:
        _backup(_artifact("future_streams_regressor.pkl"))
        joblib.dump(reg, _artifact("future_streams_regressor.pkl"))

    # Save the exact data used by the active model so explainability uses the new training distribution.
    active_cols = FEATURES + (["future_streams"] if "future_streams" in train_df.columns else [])
    active_snapshot = train_df[active_cols].copy()
    if "hit" in train_df.columns:
        active_snapshot["hit"] = y.values
    active_snapshot.to_csv(settings.ACTIVE_TRAINING_PATH, index=False)

    # Save a clean, labelled training snapshot for reproducibility.
    out_dir = Path(settings.BASE_DIR) / "data" / "training_uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uploaded_path = out_dir / f"training_{stamp}.csv"
    df_new.to_csv(uploaded_path, index=False)

    fi = sorted(
        [{"feature": FEATURE_LABELS.get(f, f), "importance": round(float(v), 6)} for f, v in zip(FEATURES, clf.feature_importances_)],
        key=lambda x: x["importance"], reverse=True
    )

    old_report = {}
    try:
        with open(settings.REPORT_PATH, "r", encoding="utf-8") as f:
            old_report = json.load(f)
    except Exception:
        pass

    report = {
        **old_report,
        "classifier": {**metrics, "training_rows": int(len(X)), "new_rows": int(len(df_new)), "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()}},
        "feature_importance": fi,
        "regression": reg_metrics or old_report.get("regression"),
        "training": {
            "status": "retrained",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_file": uploaded_file.name,
            "mode": mode,
            "rows_used": int(len(X)),
            "new_rows": int(len(df_new)),
            "target": "hit" if "hit" in df_new.columns else "future_streams -> top 20% hit",
            "hit_threshold": threshold if 'threshold' in locals() else new_threshold,
            "features": FEATURES,
            "note": "Retrained locally from the uploaded labeled dataset. The model does not learn from unlabeled prediction inputs.",
        },
    }
    Path(settings.REPORT_PATH).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Clear model_service caches so the running Django process uses the new artifacts.
    model.cache_clear()
    regressor.cache_clear()
    training_data.cache_clear()

    return report
