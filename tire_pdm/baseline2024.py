"""Python port of the 2024 internship pipeline, and a check of its evaluation.

Method 1 (frequency response of the wheel for a normal, flat and punctured
tyre) and method 2 (acceleration, velocity and displacement of a good and a
degraded wheel) are reproduced line by line from the MATLAB listings of the
2024 report; the classifier is its linear SVC on (frequency, magnitude)
points. These functions regenerate the 2024 CSV files exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

M2, K_SUSP, KT_NORMAL, C_DAMP = 100.0, 50_000.0, 200_000.0, 1_000.0


def wheel_response_db(f, m2=M2, k=K_SUSP, kt=KT_NORMAL, c=C_DAMP):
    """Listing 4.6 of the report: 1-DOF wheel response, in dB."""
    w = 2 * np.pi * f
    w2 = np.sqrt((k + kt) / m2)
    z2 = c / (2 * np.sqrt((k + kt) * m2))
    H = (kt / (k + kt)) / (1 - (w / w2) ** 2 + 1j * (2 * z2 * w / w2))
    return 20 * np.log10(np.abs(H))


def method1(m2=M2, k=K_SUSP, kt=KT_NORMAL, c=C_DAMP):
    """Amplitude_Frequency_Data.csv: normal, flat (kt x 0.5), punctured (kt x 0.75)."""
    f = np.linspace(0, 10, 1000)
    return pd.DataFrame({
        "Frequency_Hz": f,
        "Magnitude_Normal_dB": wheel_response_db(f, m2, k, kt, c),
        "Magnitude_Flat_dB": wheel_response_db(f, m2, k, kt * 0.5, c),
        "Magnitude_Perforated_dB": wheel_response_db(f, m2, k, kt * 0.75, c),
    })


def method2():
    """Acceleration-vitesse-deplacement.csv (listing 4.9)."""
    f = np.logspace(np.log10(0.1), np.log10(1000), 1000)
    zeta = 0.1
    w0 = 2 * np.pi * f
    a_good = 0.1
    a_bad = a_good * (1 + 0.5 * np.exp(-((f - 100) ** 2) / (2 * 50 ** 2)))
    damp = np.exp(-zeta * w0)
    return pd.DataFrame({
        "Frequency_Hz": f,
        "Acceleration_Good_m_s2": np.full_like(f, a_good),
        "Acceleration_Bad_m_s2": a_bad,
        "Velocity_Good_m_s": a_good / w0 * damp,
        "Velocity_Bad_m_s": a_bad / w0 * damp,
        "Displacement_Good_m": a_good / w0 ** 2 * damp,
        "Displacement_Bad_m": a_bad / w0 ** 2 * damp,
    })


def training_table(curves: pd.DataFrame) -> pd.DataFrame:
    """The 2024 training file: the three curves stacked, Panne = 0 for the
    normal tyre and 1 for the flat and punctured ones."""
    f = curves["Frequency_Hz"]
    parts = [pd.DataFrame({"Frequence": f, "Magnitude": curves[col], "Panne": lab})
             for col, lab in (("Magnitude_Normal_dB", 0), ("Magnitude_Flat_dB", 1),
                              ("Magnitude_Perforated_dB", 1))]
    return pd.concat(parts, ignore_index=True)


def original_svc_accuracy(table: pd.DataFrame) -> dict:
    """Listing of section 4.2.2.1: linear SVC, random 80/20 split, seed 42.
    (The report does not scale the features; neither do we.)"""
    X = table[["Frequence", "Magnitude"]]
    y = table["Panne"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel="linear").fit(Xtr, ytr)
    return {"accuracy": float((model.predict(Xte) == yte).mean()),
            "always_predict_fault": float((yte == 1).mean()),
            "test_points": int(len(yte))}


def generalisation_check(n_vehicles=60, seed=0, step=20):
    """Same pipeline on many tyres instead of one.

    Each simulated vehicle gets its own wheel mass, suspension stiffness,
    tyre stiffness and damping (+-20 %), and contributes three curves.
    Every ``step``-th frequency point is kept (50 points per curve) to keep
    the SVC fast. Points are classified by an RBF SVC; we compare a random split of the
    points (as in 2024) with a split by vehicle, and also score whole curves
    by majority vote. Returns a dict of accuracies.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for vid in range(n_vehicles):
        m2 = M2 * rng.uniform(0.8, 1.2)
        k = K_SUSP * rng.uniform(0.8, 1.2)
        kt = KT_NORMAL * rng.uniform(0.8, 1.2)
        c = C_DAMP * rng.uniform(0.8, 1.2)
        curves = method1(m2, k, kt, c).iloc[::step]
        t = training_table(curves)
        t["vehicle_id"] = vid
        t["curve_id"] = vid * 3 + np.repeat([0, 1, 2], len(curves))
        rows.append(t)
    data = pd.concat(rows, ignore_index=True)
    X = data[["Frequence", "Magnitude"]].to_numpy()
    y = data["Panne"].to_numpy()
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    random_acc = float((model.fit(X[idx_tr], y[idx_tr]).predict(X[idx_te]) == y[idx_te]).mean())

    point_ok, curve_ok, curves = 0, 0, 0
    for tr, te in GroupKFold(5).split(X, y, data["vehicle_id"]):
        pred = model.fit(X[tr], y[tr]).predict(X[te])
        point_ok += int((pred == y[te]).sum())
        part = data.iloc[te].assign(pred=pred)
        votes = part.groupby("curve_id").agg(pred=("pred", "mean"), y=("Panne", "first"))
        curve_ok += int(((votes["pred"] > 0.5).astype(int) == votes["y"]).sum())
        curves += len(votes)
    return {
        "points_random_split": random_acc,
        "points_vehicle_split": point_ok / len(y),
        "curves_vehicle_split_majority_vote": curve_ok / curves,
        "n_curves": curves,
    }
