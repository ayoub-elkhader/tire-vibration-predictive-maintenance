"""Supervised fault classification and unsupervised anomaly detection.

All evaluations split by vehicle (GroupKFold): a vehicle seen in training is
never seen in testing, which is what a fleet deployment would face.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import feature_columns
from .scenarios import CLASSES


def classifiers(seed=0):
    return {
        "svc_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def cross_validate(df, model_name="random_forest", split="vehicle", n_splits=5, seed=0,
                   columns=None):
    """Out-of-fold predictions. ``split``: 'vehicle' (GroupKFold) or 'random'."""
    cols = columns or feature_columns(df)
    X = df[cols].to_numpy()
    y = df["label"].to_numpy()
    if split == "vehicle":
        folds = GroupKFold(n_splits).split(X, y, df["vehicle_id"])
    else:
        folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        model = classifiers(seed)[model_name]
        pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
    return pred


def scores(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "recall_per_class": {c: float(np.mean(pred[y == c] == c)) for c in CLASSES if (y == c).any()},
    }


def confusion(y, pred):
    return confusion_matrix(y, pred, labels=list(CLASSES))


def importance(df, seed=0, n_repeats=5):
    """Permutation importance of the random forest on a held-out vehicle split
    (drop in macro-F1)."""
    cols = feature_columns(df)
    X = df[cols].to_numpy()
    y = df["label"].to_numpy()
    tr, te = next(GroupKFold(5).split(X, y, df["vehicle_id"]))
    model = classifiers(seed)["random_forest"].fit(X[tr], y[tr])
    r = permutation_importance(model, X[te], y[te], n_repeats=n_repeats, random_state=seed,
                               scoring="f1_macro", n_jobs=-1)
    return pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)


def _detector_scores(name, X_fit, X_score, seed):
    scaler = StandardScaler().fit(X_fit)
    A, B = scaler.transform(X_fit), scaler.transform(X_score)
    if name == "mahalanobis":
        return EmpiricalCovariance().fit(A).mahalanobis(B)
    return -IsolationForest(n_estimators=300, random_state=seed).fit(A).score_samples(B)


def anomaly_detection(df, n_splits=5, false_alarm=0.05, seed=0):
    """Train on healthy runs of the training vehicles only; score the test
    vehicles. Returns per-detector AUC and detection rate per class.

    The alarm threshold is the (1 - ``false_alarm``) quantile of healthy
    scores obtained out-of-sample inside the training vehicles (inner
    vehicle-wise folds): in-sample scores would be optimistic."""
    cols = feature_columns(df)
    X = df[cols].to_numpy()
    y = df["label"].to_numpy()
    groups = df["vehicle_id"].to_numpy()
    out = {}
    for name in ("mahalanobis", "isolation_forest"):
        score = np.empty(len(y))
        thr = np.empty(len(y))
        for tr, te in GroupKFold(n_splits).split(X, y, groups):
            healthy = tr[y[tr] == "healthy"]
            inner = np.empty(len(healthy))
            for itr, ite in GroupKFold(4).split(healthy, groups=groups[healthy]):
                inner[ite] = _detector_scores(name, X[healthy[itr]], X[healthy[ite]], seed)
            score[te] = _detector_scores(name, X[healthy], X[te], seed)
            thr[te] = np.quantile(inner, 1 - false_alarm)
        res = {"auc_vs_healthy": {}, "detection_rate": {}}
        h = y == "healthy"
        for c in CLASSES:
            m = y == c
            res["detection_rate"][c] = float(np.mean(score[m] > thr[m]))
            if c != "healthy":
                sel = h | m
                res["auc_vs_healthy"][c] = float(roc_auc_score(m[sel], score[sel]))
        out[name] = res
    return out
