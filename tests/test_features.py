import numpy as np
import pytest

from tire_pdm.features import extract, suspension_transfer, wheel_hop
from tire_pdm.physics import QuarterCar
from tire_pdm.scenarios import Condition, simulate_run


@pytest.fixture(scope="module")
def car():
    return QuarterCar()


def _run(car, seed, **cond):
    rng = np.random.default_rng(seed)
    return simulate_run(car, Condition(**cond), rng, speed=20.0, road_class="B")


def test_wheel_hop_and_suspension_damping_are_recovered(car):
    run = _run(car, 1)
    fh_true = run.car.natural_frequencies()[1]
    fh, _ = wheel_hop(run.hub_acc, run.fs, run.wheel_rot_hz)
    assert abs(fh / fh_true - 1) < 0.05
    zeta_true = run.car.cs / (2 * np.sqrt(run.car.ks * run.car.ms))
    _, zeta, _ = suspension_transfer(run.body_acc, run.hub_acc, run.fs)
    assert abs(zeta / zeta_true - 1) < 0.10


def test_underinflation_lowers_wheel_hop(car):
    ok = _run(car, 2)
    low = _run(car, 2, pressure_ratio=0.5)
    assert wheel_hop(low.hub_acc, low.fs, low.wheel_rot_hz)[0] < 0.9 * wheel_hop(ok.hub_acc, ok.fs, ok.wheel_rot_hz)[0]


def test_bulge_and_imbalance_show_in_synchronous_features(car):
    ok = extract(*(lambda r: (r.body_acc, r.hub_acc, r.fs, r.speed, r.wheel_rot_hz))(_run(car, 3)))
    bulge = extract(*(lambda r: (r.body_acc, r.hub_acc, r.fs, r.speed, r.wheel_rot_hz))(_run(car, 3, bulge_m=3e-3)))
    unb = extract(*(lambda r: (r.body_acc, r.hub_acc, r.fs, r.speed, r.wheel_rot_hz))(_run(car, 3, imbalance_kgm=0.02)))
    assert bulge["tsa_kurtosis"] > 2 * ok["tsa_kurtosis"]
    assert unb["order1_snr_db"] > ok["order1_snr_db"] + 6
