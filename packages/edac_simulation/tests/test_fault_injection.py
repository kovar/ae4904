"""
test_fault_injection.py — Unit tests for FaultInjector.

Covers: inject_at_ber (zero, statistical count, high rate),
evaluate_single_pass (key structure, postconditions, clean baseline), and
orbital_injection_test (key structure, scrub-pass count, log length).
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from edac import BCHCodec
from fault_injection import FaultInjector
from nand_flash_model import NANDFlashModel
from scrubbing import Scrubber


# ---------------------------------------------------------------------------
# Local fixture: FaultInjector wrapping the shared filled_memory / scrubber
# ---------------------------------------------------------------------------


@pytest.fixture
def injector(
    filled_memory: NANDFlashModel,
    codec: BCHCodec,
    scrubber: Scrubber,
    rng: np.random.Generator,
) -> FaultInjector:
    return FaultInjector(filled_memory, codec, scrubber, rng=rng)


# ---------------------------------------------------------------------------
# inject_at_ber
# ---------------------------------------------------------------------------


class TestInjectAtBer:
    def test_zero_ber_injects_nothing(self, injector: FaultInjector) -> None:
        n = injector.inject_at_ber(0.0)
        assert n == 0
        assert injector.memory.total_error_count() == 0

    def test_count_within_statistical_bounds(self, injector: FaultInjector) -> None:
        """Injected count should be close to BER × total_bits (binomial)."""
        ber = 1e-4
        total_bits = injector.memory.total_bits
        n = injector.inject_at_ber(ber)
        expected = ber * total_bits
        std = (total_bits * ber * (1 - ber)) ** 0.5
        # 6-sigma bound is extremely conservative; test is deterministic with fixed seed
        assert abs(n - expected) < 6 * max(std, 1.0)

    def test_high_ber_injects_errors(self, injector: FaultInjector) -> None:
        """At BER=0.01, expect at least some bits to be flipped."""
        n = injector.inject_at_ber(0.01)
        assert n > 0


# ---------------------------------------------------------------------------
# evaluate_single_pass
# ---------------------------------------------------------------------------


class TestEvaluateSinglePass:
    def test_clean_pass_all_zeros(self, injector: FaultInjector) -> None:
        """No errors injected → scrub pass reports everything zero."""
        result = injector.evaluate_single_pass()
        assert result["errors_before"] == 0
        assert result["errors_corrected"] == 0
        assert result["uncorrectable_pages"] == 0
        assert result["residual_errors"] == 0

    def test_returns_expected_keys(
        self, injector: FaultInjector, filled_memory: NANDFlashModel
    ) -> None:
        injector.inject_at_ber(1e-5)
        result = injector.evaluate_single_pass()
        assert set(result.keys()) == {
            "errors_before",
            "errors_corrected",
            "uncorrectable_pages",
            "residual_errors",
        }

    def test_residual_matches_memory_error_count(
        self, injector: FaultInjector, filled_memory: NANDFlashModel
    ) -> None:
        injector.inject_at_ber(1e-5)
        result = injector.evaluate_single_pass()
        assert result["residual_errors"] == filled_memory.total_error_count()

    def test_errors_before_equals_injected_count(self, injector: FaultInjector) -> None:
        """errors_before must reflect errors present before the scrub pass."""
        filled_memory = injector.memory
        injector.inject_at_ber(1e-5)
        expected_before = filled_memory.total_error_count()
        result = injector.evaluate_single_pass()
        assert result["errors_before"] == expected_before


# ---------------------------------------------------------------------------
# orbital_injection_test
# ---------------------------------------------------------------------------


class TestOrbitalInjectionTest:
    def test_returns_expected_keys(self, injector: FaultInjector) -> None:
        result = injector.orbital_injection_test(
            duration_s=config.SCRUB_PERIOD_S * 2,
            scrub_period_s=config.SCRUB_PERIOD_S,
        )
        required = {
            "duration_s",
            "scrub_period_s",
            "n_scrub_passes",
            "total_injected",
            "total_corrected",
            "total_uncorr_pages",
            "residual_errors",
            "scrub_log",
            "uncorrectable_ber",
        }
        assert required.issubset(result.keys())

    def test_scrub_count_matches_duration(self, injector: FaultInjector) -> None:
        n_periods = 3
        result = injector.orbital_injection_test(
            duration_s=config.SCRUB_PERIOD_S * n_periods,
            scrub_period_s=config.SCRUB_PERIOD_S,
        )
        assert result["n_scrub_passes"] == n_periods

    def test_scrub_log_length_matches_passes(self, injector: FaultInjector) -> None:
        n_periods = 2
        result = injector.orbital_injection_test(
            duration_s=config.SCRUB_PERIOD_S * n_periods,
            scrub_period_s=config.SCRUB_PERIOD_S,
        )
        assert len(result["scrub_log"]) == n_periods

    def test_zero_uncorrectable_ber_when_no_uncorr_events(
        self, injector: FaultInjector
    ) -> None:
        """uncorrectable_ber must be 0.0 when no uncorrectable pages were seen."""
        result = injector.orbital_injection_test(
            duration_s=config.SCRUB_PERIOD_S * 2,
            scrub_period_s=config.SCRUB_PERIOD_S,
        )
        if result["total_uncorr_pages"] == 0:
            assert result["uncorrectable_ber"] == 0.0
