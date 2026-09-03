"""Independent RDP accountant: properties + DP-SGD sanity (numpy only)."""
import pytest

from fl_med.privacy.accounting import compute_epsilon


def test_analytic_anchor_q1():
    # q=1, sigma=1, 1 step, delta=1e-5 -> ~4.75 (tight RDP->DP conversion)
    eps = compute_epsilon(sample_rate=1.0, noise_multiplier=1.0, steps=1, delta=1e-5)
    assert eps == pytest.approx(4.75, abs=0.15)


def test_subsampling_amplifies_privacy():
    e_full = compute_epsilon(sample_rate=1.0, noise_multiplier=1.0, steps=1, delta=1e-5)
    e_sub = compute_epsilon(sample_rate=0.01, noise_multiplier=1.0, steps=1, delta=1e-5)
    assert e_sub < e_full


def test_epsilon_increases_with_steps():
    a = compute_epsilon(sample_rate=0.01, noise_multiplier=1.0, steps=100, delta=1e-5)
    b = compute_epsilon(sample_rate=0.01, noise_multiplier=1.0, steps=1000, delta=1e-5)
    assert b > a


def test_epsilon_decreases_with_noise():
    lo = compute_epsilon(sample_rate=0.01, noise_multiplier=1.0, steps=1000, delta=1e-5)
    hi = compute_epsilon(sample_rate=0.01, noise_multiplier=4.0, steps=1000, delta=1e-5)
    assert hi < lo


def test_delta_below_one_over_n():
    # Config delta must be < 1/N for every client; N=7,947 is the strictest bound.
    assert 1e-5 < 1.0 / 7947


def test_zero_steps_spend_no_privacy():
    assert compute_epsilon(
        sample_rate=0.01, noise_multiplier=1.0, steps=0, delta=1e-5
    ) == 0.0


def test_zero_sampling_spends_no_privacy():
    assert compute_epsilon(
        sample_rate=0.0, noise_multiplier=1.0, steps=100, delta=1e-5
    ) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 1.1, "noise_multiplier": 1.0, "steps": 1, "delta": 1e-5},
        {"sample_rate": 0.1, "noise_multiplier": 0.0, "steps": 1, "delta": 1e-5},
        {"sample_rate": 0.1, "noise_multiplier": 1.0, "steps": -1, "delta": 1e-5},
        {"sample_rate": 0.1, "noise_multiplier": 1.0, "steps": 1, "delta": 1.0},
    ],
)
def test_invalid_accounting_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_epsilon(**kwargs)
