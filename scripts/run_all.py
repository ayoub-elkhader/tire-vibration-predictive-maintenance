"""Run every experiment, write results/results.json and figures/*.png.

    python scripts/run_all.py            # everything (about 17 minutes on a laptop CPU)
    python scripts/run_all.py --no-cnn   # skip the spectrogram CNN
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tire_pdm import baseline2024, itpms, models  # noqa: E402
from tire_pdm.dataset import feature_columns, generate  # noqa: E402
from tire_pdm.features import tsa  # noqa: E402
from tire_pdm.scenarios import CLASSES, sample_condition, sample_vehicle, simulate_run  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
DATA = ROOT / "data"
COLORS = dict(zip(CLASSES, ["#2b7a3d", "#c0392b", "#2c6fbb", "#8e44ad", "#d68910"]))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fig_examples(seed=5):
    rng = np.random.default_rng(seed)
    car = sample_vehicle(rng)
    fig, ax = plt.subplots(3, 2, figsize=(12, 9))
    for c in CLASSES:
        run = simulate_run(car, sample_condition(c, rng), rng, speed=22.0, road_class="B")
        t = np.arange(len(run.hub_acc)) / run.fs
        if c in ("healthy", "bulge"):
            ax[0, 0 if c == "healthy" else 1].plot(t[:2000], run.hub_acc[:2000], color=COLORS[c], lw=0.6)
            ax[0, 0 if c == "healthy" else 1].set_title(f"Hub acceleration, {c} (22 m/s, road B)")
        f, P = signal.welch(run.hub_acc, run.fs, nperseg=4096)
        m = (f > 2) & (f < 40)
        ax[1, 0].semilogy(f[m], P[m] / f[m] ** 2, color=COLORS[c], label=c)
        avg, _ = tsa(run.hub_acc, run.fs, run.wheel_rot_hz)
        ax[1, 1].plot(np.linspace(0, 360, len(avg), endpoint=False), avg - avg.mean(),
                      color=COLORS[c], label=c)
        fb, Pb = signal.csd(run.hub_acc, run.body_acc, run.fs, nperseg=8192)
        _, Ph = signal.welch(run.hub_acc, run.fs, nperseg=8192)
        mb = (fb > 0.4) & (fb < 8)
        ax[2, 0].plot(fb[mb], np.abs(Pb[mb] / Ph[mb]), color=COLORS[c], label=c)
    for a in ax[0]:
        a.set_xlabel("time (s)"); a.set_ylabel("m/s²")
    ax[1, 0].set(title="Whitened hub PSD (PSD / f²): wheel-hop peak", xlabel="Hz")
    ax[1, 1].set(title="Time-synchronous average over one wheel revolution",
                 xlabel="wheel angle (deg)", ylabel="m/s²")
    ax[2, 0].set(title="|body / hub| (H1 estimate): suspension resonance", xlabel="Hz")
    for a in (ax[1, 0], ax[1, 1], ax[2, 0]):
        a.legend(fontsize=8)
    ax[2, 1].axis("off")
    ax[2, 1].text(0, 0.5, "Same vehicle, five wheel conditions.\n"
                  "Underinflation lowers the wheel-hop peak;\n"
                  "a worn damper raises the suspension peak;\n"
                  "imbalance and bulge repeat every revolution.", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "signals_and_indicators.png", dpi=130)
    plt.close(fig)


def fig_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--vehicles", type=int, default=160)
    args = ap.parse_args()
    for d in (FIG, RES, DATA):
        d.mkdir(exist_ok=True)
    results = {}
    t0 = time.time()

    # 0. the 2024 pipeline
    log("2024 baseline")
    curves = baseline2024.method1()
    results["baseline_2024"] = {
        "natural_frequencies_hz": {
            name: float(np.sqrt((baseline2024.K_SUSP + kt) / baseline2024.M2) / (2 * np.pi))
            for name, kt in (("normal", baseline2024.KT_NORMAL), ("flat", baseline2024.KT_NORMAL * 0.5),
                             ("punctured", baseline2024.KT_NORMAL * 0.75))},
        "original_linear_svc_random_split": baseline2024.original_svc_accuracy(
            baseline2024.training_table(curves)),
        "generalisation_check": baseline2024.generalisation_check(),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, c in (("Magnitude_Normal_dB", "healthy"), ("Magnitude_Flat_dB", "underinflated"),
                   ("Magnitude_Perforated_dB", "bulge")):
        ax.plot(curves["Frequency_Hz"], curves[col], color=COLORS[c], label=col.split("_")[1])
    ax.set(xlabel="Hz", ylabel="dB", title="2024 method 1, regenerated in Python")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "baseline_2024_curves.png", dpi=130); plt.close(fig)

    # 1. dataset
    log(f"simulating {args.vehicles} vehicles x {len(CLASSES)} conditions x 3 runs")
    df, specs = generate(n_vehicles=args.vehicles, runs_per_class=3, seed=0)
    df.to_csv(DATA / "features.csv", index=False)
    np.save(DATA / "spectrograms.npy", specs.astype(np.float16))
    results["dataset"] = {"runs": len(df), "vehicles": int(df.vehicle_id.nunique()),
                          "features": len(feature_columns(df)),
                          "spectrogram_shape": list(specs.shape[1:])}
    fig_examples()

    # physical indicators vs ground truth
    results["indicators"] = {
        "susp_zeta_corr_with_true": float(np.corrcoef(df.susp_zeta, df.true_susp_zeta)[0, 1]),
        "susp_zeta_median_abs_rel_error": float(np.median(np.abs(df.susp_zeta / df.true_susp_zeta - 1))),
        "hop_freq_corr_with_true": float(np.corrcoef(df.hop_freq, df.true_hop_freq)[0, 1]),
    }

    # 2. supervised classification
    y = df["label"].to_numpy()
    results["classification"] = {}
    best = None
    for name in models.classifiers():
        log(f"classification {name}")
        pred = models.cross_validate(df, name, split="vehicle")
        results["classification"][name] = models.scores(y, pred)
        if best is None or results["classification"][name]["macro_f1"] > best[1]:
            best = (name, results["classification"][name]["macro_f1"], pred)
    pred_rand = models.cross_validate(df, best[0], split="random")
    results["classification"][f"{best[0]}_random_split"] = models.scores(y, pred_rand)
    fig_confusion(models.confusion(y, best[2]), f"{best[0]}, split by vehicle",
                  FIG / "confusion_features.png")
    imp = models.importance(df)
    results["permutation_importance_top10"] = {k: float(v) for k, v in imp.head(10).items()}
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.head(15)[::-1].plot.barh(ax=ax, color="#2c6fbb")
    ax.set(xlabel="drop in macro-F1 when shuffled", title="Permutation importance (random forest)")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=130); plt.close(fig)

    # baseline with the 2024 inputs only (no physics features): band energies + RMS
    simple = [c for c in feature_columns(df) if "band" in c or c.endswith("_rms")]
    pred = models.cross_validate(df, "svc_rbf", split="vehicle", columns=simple)
    results["classification"]["svc_rbf_spectral_bands_only"] = models.scores(y, pred)

    # 3. CNN on spectrograms
    if not args.no_cnn:
        log("CNN on spectrograms")
        from tire_pdm.cnn import train_eval
        pred = train_eval(specs, y, df["vehicle_id"].to_numpy())
        results["classification"]["cnn_spectrogram"] = models.scores(y, pred)
        fig_confusion(models.confusion(y, pred), "CNN on spectrograms, split by vehicle",
                      FIG / "confusion_cnn.png")

    # 4. anomaly detection trained on healthy data only
    log("anomaly detection")
    results["anomaly_detection"] = models.anomaly_detection(df)

    # 5. pressure estimation
    log("pressure estimation")
    p_true, p_est, _ = itpms.pressure_study()
    err = p_est - p_true
    under = p_true < itpms.ALARM
    flag = p_est < itpms.ALARM
    results["pressure_estimation"] = {
        "runs": int(len(p_true)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "p90_abs_error": float(np.quantile(np.abs(err), 0.9)),
        "single_drive_detection_rate_below_0.75": float(flag[under].mean()),
        "single_drive_false_alarm_rate_above_0.90": float(flag[p_true > 0.90].mean()),
    }
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(p_true, p_est, s=8, alpha=0.6)
    ax.plot([0.4, 1.05], [0.4, 1.05], "k--", lw=1)
    ax.axhline(itpms.ALARM, color="#c0392b", lw=0.8); ax.axvline(itpms.ALARM, color="#c0392b", lw=0.8)
    ax.set(xlabel="true pressure / nominal", ylabel="estimated (wheel-hop frequency)",
           title="Pressure from one 16 s drive")
    fig.tight_layout(); fig.savefig(FIG / "pressure_estimation.png", dpi=130); plt.close(fig)

    # 6. prognostics (slow leak)
    log("prognostics")
    scen = itpms.prognostics_study()
    delays = np.array([s["alarm"] - s["t_cross"] for s in scen])
    rul_err = {}
    for frac in (0.25, 0.5, 0.75):
        e = []
        for s in scen:
            k = int(np.searchsorted(s["times"], frac * s["t_cross"]))
            if k < len(s["times"]) and np.isfinite(s["rul_pred"][k]):
                e.append(s["rul_pred"][k] - s["rul_true"][k])
        rul_err[f"at_{int(frac * 100)}pct_of_life"] = {
            "n": len(e), "median_abs_error_days": float(np.median(np.abs(e))),
            "median_true_rul_days": float(np.median([(1 - frac) * s["t_cross"] for s in scen]))}
    results["prognostics"] = {
        "scenarios": len(scen),
        "alarm_raised": int(np.isfinite(delays).sum()),
        "alarm_before_true_crossing": int((delays < 0).sum()),
        "alarm_delay_days_median": float(np.median(delays[np.isfinite(delays)])),
        "alarm_delay_days_p10_p90": [float(np.quantile(delays[np.isfinite(delays)], q)) for q in (0.1, 0.9)],
        "rul_error": rul_err,
    }
    s = scen[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(s["times"], s["p_true"], "k", label="true pressure")
    ax[0].plot(s["times"], s["p_hat"], ".", color="#2c6fbb", label="estimate per drive")
    ax[0].axhline(itpms.ALARM, color="#c0392b", lw=0.8, label="alarm level")
    ax[0].axvline(s["t_cross"], color="k", ls=":", lw=0.8)
    if np.isfinite(s["alarm"]):
        ax[0].axvline(s["alarm"], color="#c0392b", ls="--", lw=0.8, label="alarm raised")
    ax[0].set(xlabel="days", ylabel="pressure / nominal", title="Slow leak, one vehicle"); ax[0].legend(fontsize=8)
    ax[1].plot(s["times"], s["rul_true"], "k", label="true RUL")
    ax[1].plot(s["times"], s["rul_pred"], color="#2c6fbb", label="predicted RUL")
    ax[1].set(xlabel="days", ylabel="days before p < 0.75", ylim=(-5, max(s["rul_true"]) * 1.5),
              title="Remaining useful life"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "prognostics_example.png", dpi=130); plt.close(fig)

    results["runtime_s"] = round(time.time() - t0, 1)
    (RES / "results.json").write_text(json.dumps(results, indent=2))
    log(f"done in {results['runtime_s']} s")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
