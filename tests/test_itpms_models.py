import numpy as np

from tire_pdm import itpms, models
from tire_pdm.dataset import feature_columns, generate


def test_pressure_ratio_inverse_is_consistent():
    assert abs(itpms.pressure_ratio(10.0, 10.0) - 1.0) < 1e-12
    p = itpms.pressure_ratio(np.array([9.0, 9.5, 10.0]), 10.0)
    assert np.all(np.diff(p) > 0)


def test_decay_fit_recovers_time_constant():
    t = np.linspace(0, 10, 21)
    p = 0.98 * np.exp(-t / 25)
    a, b = itpms.fit_decay(t, p)
    assert abs(-1 / b - 25) < 1e-6
    assert abs(itpms.predict_crossing(t, p) - 25 * np.log(0.98 / itpms.ALARM)) < 1e-6


def test_small_dataset_pipeline_never_uses_ground_truth():
    df, specs = generate(n_vehicles=6, runs_per_class=1, seed=7)
    cols = feature_columns(df)
    assert not any(c.startswith("true_") for c in cols)
    assert "label" not in cols and "vehicle_id" not in cols
    assert specs.shape[0] == len(df)
    pred = models.cross_validate(df, "random_forest", n_splits=3)
    assert len(pred) == len(df)
