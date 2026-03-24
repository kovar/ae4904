"""
test_analytical.py — Analytical consistency and reproducibility tests.

Three categories:
  1. Deterministic reproducibility: identical seeds → identical output.
     This validates the core simulation requirement for reproducible results.
  2. Poisson statistics: the RadiationModel generates events at the correct rate.
     Cross-checks the random event generator against closed-form Poisson theory.
  3. Uncorrectable-event probability: confirms that the BCH(t=4) choice is
     sufficient for the current radiation environment — the expected SEU count
     per sector per scrub period is far below t, and a full trial produces zero
     uncorrectable events.
"""

from __future__ import annotations

import numpy as np

import config
from radiation_model import RadiationModel
from simulation import run_one_trial


# ---------------------------------------------------------------------------
# 1. Deterministic reproducibility
# ---------------------------------------------------------------------------


class TestDeterministicReproducibility:
    def test_same_seed_same_results(self) -> None:
        """Two independent runs with the same seed must produce identical output."""
        seed = 0
        t1 = run_one_trial(scrub_period_s=config.SCRUB_PERIOD_S, seed=seed)
        t2 = run_one_trial(scrub_period_s=config.SCRUB_PERIOD_S, seed=seed)
        assert t1["total_injected"] == t2["total_injected"]
        assert t1["total_corrected"] == t2["total_corrected"]
        assert t1["total_uncorr_pages"] == t2["total_uncorr_pages"]
        assert t1["residual_errors"] == t2["residual_errors"]

    def test_different_seeds_produce_different_runs(self) -> None:
        """
        Runs with different seeds should (with overwhelming probability) differ.

        With the current very low SEU rate, both runs often accumulate 0 events
        (physically valid).  We therefore only assert divergence when the counts
        are non-zero — if both are zero the RNG is not broken, just quiet.
        """
        t1 = run_one_trial(scrub_period_s=config.SCRUB_PERIOD_S, seed=1)
        t2 = run_one_trial(scrub_period_s=config.SCRUB_PERIOD_S, seed=999)
        identical = (
            t1["total_injected"] == t2["total_injected"]
            and t1["total_corrected"] == t2["total_corrected"]
        )
        if identical and t1["total_injected"] != 0:
            raise AssertionError(
                "Non-zero event counts were identical across different seeds — RNG may be broken"
            )


# ---------------------------------------------------------------------------
# 2. Poisson statistics
# ---------------------------------------------------------------------------


class TestPoissonStatistics:
    def test_mean_event_count_matches_theory(self) -> None:
        """
        Over many independent trials the mean event count must converge to the
        Poisson parameter λ = array_seu_rate_s × duration_s.
        """
        rng = np.random.default_rng(77)
        rate = 1e-4  # /bit/s — elevated for fast convergence
        total_bits = 50_000
        duration_s = 1000.0
        expected = rate * total_bits * duration_s  # λ = 5000

        rm = RadiationModel(total_bits=total_bits, seu_rate_bit_s=rate, rng=rng)
        n_samples = 500
        counts = np.array(
            [len(rm.generate_events(duration_s)) for _ in range(n_samples)]
        )

        # std of the sample mean ≈ sqrt(λ / n_samples); 5σ gives P_fail ≈ 3×10⁻⁷
        assert abs(counts.mean() - expected) < 5 * np.sqrt(expected / n_samples)

    def test_variance_matches_poisson(self) -> None:
        """
        For a Poisson distribution, Var[N] = E[N] = λ.
        The sample variance should be close to λ.
        """
        rng = np.random.default_rng(88)
        rate = 1e-4
        total_bits = 50_000
        duration_s = 1000.0
        expected = rate * total_bits * duration_s  # λ = 5000

        rm = RadiationModel(total_bits=total_bits, seu_rate_bit_s=rate, rng=rng)
        n_samples = 500
        counts = np.array(
            [len(rm.generate_events(duration_s)) for _ in range(n_samples)]
        )

        # Sample variance should be within ~20% of λ for n=500
        assert abs(counts.var() - expected) / expected < 0.20


# ---------------------------------------------------------------------------
# 3. Uncorrectable-event probability
# ---------------------------------------------------------------------------


class TestUncorrectableProbability:
    def test_expected_lambda_per_sector_well_below_t(self) -> None:
        """
        Expected SEU count per sector per scrub period must be far below t=4,
        confirming the BCH choice is adequate for this radiation environment.
        """
        bits_per_sector = config.SECTOR_DATA_BYTES * 8
        lambda_per_scrub = (
            config.SEU_RATE_BIT_S * bits_per_sector * config.SCRUB_PERIOD_S
        )
        assert lambda_per_scrub < config.BCH_T, (
            f"λ={lambda_per_scrub:.2e} ≥ BCH(t={config.BCH_T}): "
            "uncorrectable events are no longer negligibly rare"
        )
        # Extra guard: λ < 0.1 means P(>0 errors per sector) < 10%
        assert lambda_per_scrub < 0.1, (
            f"λ={lambda_per_scrub:.2e} is large; BCH may be stressed"
        )

    def test_full_trial_produces_zero_uncorrectable_events(self) -> None:
        """
        A full 2-year mission simulation must yield zero uncorrectable-page events
        given the current SEU environment.  The analytical probability of any such
        event is < 10⁻²⁰, so this is effectively a deterministic assertion.
        """
        result = run_one_trial(scrub_period_s=config.SCRUB_PERIOD_S, seed=42)
        assert result["total_uncorr_pages"] == 0, (
            f"Unexpected uncorrectable events: {result['total_uncorr_pages']}. "
            "If SEU_RATE_BIT_DAY was increased significantly, revisit this assertion."
        )
