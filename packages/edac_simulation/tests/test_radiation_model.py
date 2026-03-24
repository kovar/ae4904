"""
test_radiation_model.py — Unit tests for RadiationModel.

Covers: constructor attributes, generate_events (zero duration, address range,
dtype), Poisson mean convergence, and expected_seus_per_page_per_scrub formula.
"""

from __future__ import annotations

import math

import numpy as np

import config
from radiation_model import RadiationModel

_TOTAL_BITS = 10_000  # small for fast tests


def test_array_seu_rate_s_computed() -> None:
    rate = 1.5e-10
    rm = RadiationModel(total_bits=2000, seu_rate_bit_s=rate)
    assert math.isclose(rm.array_seu_rate_s, rate * 2000, rel_tol=1e-9)


def test_zero_duration_returns_empty() -> None:
    rng = np.random.default_rng(0)
    rm = RadiationModel(total_bits=_TOTAL_BITS, seu_rate_bit_s=1e-6, rng=rng)
    events = rm.generate_events(0.0)
    assert len(events) == 0


def test_events_dtype_is_integer() -> None:
    rng = np.random.default_rng(0)
    rm = RadiationModel(total_bits=_TOTAL_BITS, seu_rate_bit_s=1e-3, rng=rng)
    events = rm.generate_events(100.0)
    assert np.issubdtype(events.dtype, np.integer)


def test_addresses_in_range() -> None:
    rng = np.random.default_rng(1)
    rm = RadiationModel(total_bits=_TOTAL_BITS, seu_rate_bit_s=1e-3, rng=rng)
    events = rm.generate_events(1000.0)
    assert len(events) > 0, "Expected non-zero events at high rate"
    assert int(events.min()) >= 0
    assert int(events.max()) < _TOTAL_BITS


def test_poisson_mean_converges() -> None:
    """Mean event count converges to the Poisson parameter over many samples."""
    rng = np.random.default_rng(99)
    rate = 1e-3  # /bit/s  (elevated for fast convergence)
    duration = 200.0  # s
    total_bits = 10_000
    expected = rate * total_bits * duration  # = 2000
    rm = RadiationModel(total_bits=total_bits, seu_rate_bit_s=rate, rng=rng)
    n_samples = 300
    counts = np.array([len(rm.generate_events(duration)) for _ in range(n_samples)])
    # std of the sample mean ≈ sqrt(expected / n_samples)
    assert abs(counts.mean() - expected) < 5 * np.sqrt(expected / n_samples)


def test_expected_seus_per_page_per_scrub_formula() -> None:
    """Helper value equals SEU rate × page bits × period."""
    rm = RadiationModel()
    period_s = 3600.0
    result = rm.expected_seus_per_page_per_scrub(period_s)
    expected = config.SEU_RATE_BIT_S * config.PAGE_DATA_BYTES * 8 * period_s
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_different_seeds_differ() -> None:
    """Two RadiationModels with different seeds produce different event streams."""
    rm1 = RadiationModel(
        total_bits=_TOTAL_BITS, seu_rate_bit_s=1e-3, rng=np.random.default_rng(1)
    )
    rm2 = RadiationModel(
        total_bits=_TOTAL_BITS, seu_rate_bit_s=1e-3, rng=np.random.default_rng(2)
    )
    e1 = rm1.generate_events(100.0)
    e2 = rm2.generate_events(100.0)
    # Two independent Poisson samples are astronomically unlikely to be identical
    assert not np.array_equal(e1, e2)
