import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

FEATURES = [
    "current_streams",
    "current_listeners",
    "repeat_rate",
    "skip_rate",
    "completion_rate",
    "save_rate",
    "share_rate",
    "playlist_add_rate",
    "avg_stream_growth",
    "max_stream_growth",
    "growth_acceleration",
]

FEATURE_LABELS = {
    "current_streams": "Current streams",
    "current_listeners": "Current listeners",
    "repeat_rate": "Repeat rate",
    "skip_rate": "Skip rate",
    "completion_rate": "Completion rate",
    "save_rate": "Save rate",
    "share_rate": "Share rate",
    "playlist_add_rate": "Playlist add rate",
    "avg_stream_growth": "Average stream growth",
    "max_stream_growth": "Maximum stream growth",
    "growth_acceleration": "Growth acceleration",
}

MOMENTUM_FEATURES = [
    "avg_stream_growth",
    "repeat_rate",
    "save_rate",
    "share_rate",
    "playlist_add_rate",
    "completion_rate",
]

MOMENTUM_WEIGHTS = {
    "avg_stream_growth": 0.25,
    "repeat_rate": 0.20,
    "save_rate": 0.15,
    "share_rate": 0.15,
    "playlist_add_rate": 0.10,
    "completion_rate": 0.15,
}


def _path(name):
    return Path(settings.BASE_DIR) / name


@lru_cache(maxsize=1)
def model():
    return joblib.load(settings.MODEL_PATH)


@lru_cache(maxsize=1)
def regressor():
    path = _path("model_artifacts/future_streams_regressor.pkl")
    return joblib.load(path) if path.exists() else None


@lru_cache(maxsize=1)
def training_data():
    active = _path("data/active_training_data.csv")
    source = active if active.exists() else _path("data/music_momentum_predictions.csv")
    return pd.read_csv(source)


@lru_cache(maxsize=1)
def report():
    with open(settings.REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def medians():
    return training_data()[FEATURES].median()


def _momentum(d):
    values = []
    components = []
    train = training_data()
    for feature in MOMENTUM_FEATURES:
        mn = float(train[feature].min())
        mx = float(train[feature].max())
        raw = float(d[feature])
        score = 50.0 if mx == mn else (raw - mn) / (mx - mn) * 100.0
        score = max(0.0, min(100.0, score))
        values.append(score)
        components.append({
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "score": round(score, 2),
            "weight": MOMENTUM_WEIGHTS[feature],
            "weighted": round(score * MOMENTUM_WEIGHTS[feature], 2),
        })
    score = sum(values[i] * MOMENTUM_WEIGHTS[f] for i, f in enumerate(MOMENTUM_FEATURES))
    category = "BREAKOUT" if score >= 80 else "RISING" if score >= 60 else "STABLE" if score >= 40 else "DECLINING"
    return round(score, 2), category, components


def _business_action(probability, momentum_score):
    return (
        "PROMOTE" if (probability >= 0.80 and momentum_score < 60) or (probability >= 0.70 and momentum_score >= 60)
        else "MONITOR" if probability >= 0.50
        else "MAINTAIN" if momentum_score >= 40
        else "RECONSIDER"
    )


def _future_streams(X):
    r = regressor()
    if r is None:
        return None
    try:
        prediction = float(r.predict(X)[0])
        # The notebook trains the regressor on log1p(future_streams).
        return max(0.0, float(np.expm1(prediction)))
    except Exception:
        return None


def _local_impacts(X, base_probability):
    med = medians()
    impacts = []
    m = model()
    for feature in FEATURES:
        xx = X.copy()
        xx[feature] = med[feature]
        counterfactual = float(m.predict_proba(xx)[0, 1])
        delta = base_probability - counterfactual
        impacts.append({
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "delta": round(delta * 100, 3),
            "absolute": round(abs(delta * 100), 3),
            "direction": "supports" if delta > 0.000001 else "drags" if delta < -0.000001 else "neutral",
            "current": float(X.iloc[0][feature]),
            "median": float(med[feature]),
        })
    impacts.sort(key=lambda x: x["absolute"], reverse=True)
    return impacts


def _improvement_levers(X, base_probability):
    """One-at-a-time counterfactuals toward a favorable training percentile.

    This is decision support, not a causal estimate. It answers: 'If this signal
    moved toward a stronger observed region of the training data, how would the
    model's probability change?'
    """
    train = training_data()
    m = model()
    levers = []
    for feature in FEATURES:
        current = float(X.iloc[0][feature])
        if feature == "skip_rate":
            target = float(train[feature].quantile(0.25))
            direction = "lower"
        else:
            target = float(train[feature].quantile(0.75))
            direction = "higher"
        if np.isclose(current, target):
            continue
        xx = X.copy()
        xx[feature] = target
        new_probability = float(m.predict_proba(xx)[0, 1])
        delta = (new_probability - base_probability) * 100
        levers.append({
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "delta": round(delta, 3),
            "current": round(current, 5),
            "target": round(target, 5),
            "direction": direction,
        })
    levers.sort(key=lambda x: x["delta"], reverse=True)
    return levers[:7]


def predict(d):
    X = pd.DataFrame([{k: float(d[k]) for k in FEATURES}])
    m = model()
    base_probability = float(m.predict_proba(X)[0, 1])
    momentum_score, category, components = _momentum(d)
    action = _business_action(base_probability, momentum_score)
    impacts = _local_impacts(X, base_probability)
    levers = _improvement_levers(X, base_probability)
    future_streams = _future_streams(X)

    return {
        "hit_probability": round(base_probability * 100, 2),
        "momentum_score": momentum_score,
        "momentum_category": category,
        "business_action": action,
        "predicted_future_streams": round(future_streams) if future_streams is not None else None,
        "contributors": impacts,
        "momentum_components": components,
        "improvement_levers": levers,
        "features": [
            {
                "feature": f,
                "label": FEATURE_LABELS[f],
                "value": float(d[f]),
                "median": float(medians()[f]),
            }
            for f in FEATURES
        ],
    }
