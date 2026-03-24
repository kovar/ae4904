"""
test_scrubbing.py — Unit tests for Scrubber.

Covers: clean pass, single-error and t-error correction, t+1 uncorrectable,
memory state after scrub, cross-page correction, cumulative stats, and the
static bandwidth-analysis helpers.
"""

from __future__ import annotations

import math

import config
from nand_flash_model import NANDFlashModel
from scrubbing import Scrubber

T = config.BCH_T  # = 4


# ---------------------------------------------------------------------------
# Scrub-pass behaviour
# ---------------------------------------------------------------------------


class TestScrubPass:
    def test_clean_pass_no_corrections(
        self, scrubber: Scrubber, filled_memory: NANDFlashModel
    ) -> None:
        result = scrubber.scrub_pass()
        assert result["errors_corrected"] == 0
        assert result["uncorrectable_pages"] == 0
        assert result["pages_scrubbed"] == filled_memory.n_pages

    def test_single_error_corrected(
        self, scrubber: Scrubber, filled_memory: NANDFlashModel
    ) -> None:
        filled_memory.inject_bit_flip(0)
        result = scrubber.scrub_pass()
        assert result["errors_corrected"] == 1
        assert result["uncorrectable_pages"] == 0

    def test_exactly_t_errors_corrected(
        self, scrubber: Scrubber, filled_memory: NANDFlashModel
    ) -> None:
        """Exactly T errors in one sector must all be corrected."""
        for i in range(T):
            filled_memory.inject_bit_flip(i)
        result = scrubber.scrub_pass()
        assert result["errors_corrected"] == T
        assert result["uncorrectable_pages"] == 0

    def test_t_plus_one_errors_uncorrectable(
        self, scrubber: Scrubber, filled_memory: NANDFlashModel
    ) -> None:
        """T+1 errors in one sector must be flagged uncorrectable."""
        for i in range(T + 1):
            filled_memory.inject_bit_flip(i)
        result = scrubber.scrub_pass()
        assert result["uncorrectable_pages"] == 1

    def test_corrected_data_matches_reference(
        self, scrubber: Scrubber, filled_memory: NANDFlashModel
    ) -> None:
        """After a successful scrub, stored data must equal ground-truth reference."""
        filled_memory.inject_bit_flip(0)
        scrubber.scrub_pass()
        assert filled_memory.total_error_count() == 0

    def test_errors_across_two_pages_both_corrected(
        self, scrubber: Scrubber, filled_memory: NANDFlashModel
    ) -> None:
        filled_memory.inject_bit_flip(0)  # page 0
        filled_memory.inject_bit_flip(filled_memory.page_bits)  # page 1
        result = scrubber.scrub_pass()
        assert result["errors_corrected"] == 2
        assert result["uncorrectable_pages"] == 0
        assert filled_memory.total_error_count() == 0

    def test_cumulative_stats_updated(
        self, scrubber: Scrubber, filled_memory: NANDFlashModel
    ) -> None:
        filled_memory.inject_bit_flip(0)
        scrubber.scrub_pass()
        assert scrubber.total_scrub_passes == 1
        assert scrubber.total_errors_corrected == 1
        assert scrubber.total_uncorrectable_pages == 0

    def test_second_pass_on_clean_memory_is_noop(
        self, scrubber: Scrubber, filled_memory: NANDFlashModel
    ) -> None:
        """After a correcting pass, a subsequent pass sees no errors."""
        filled_memory.inject_bit_flip(0)
        scrubber.scrub_pass()
        result2 = scrubber.scrub_pass()
        assert result2["errors_corrected"] == 0
        assert result2["uncorrectable_pages"] == 0


# ---------------------------------------------------------------------------
# Bandwidth helper methods (static — no fixture needed)
# ---------------------------------------------------------------------------


class TestScrubBandwidth:
    def test_scrub_bandwidth_bps(self) -> None:
        period_s = 3600.0
        expected = config.TOTAL_BITS / period_s
        assert math.isclose(
            Scrubber.scrub_bandwidth_bps(period_s), expected, rel_tol=1e-9
        )

    def test_max_scrub_period_s(self) -> None:
        allocated_bps = (
            config.INTERFACE_THROUGHPUT_BPS * config.SCRUB_BANDWIDTH_FRACTION
        )
        expected = config.TOTAL_BITS / allocated_bps
        assert math.isclose(Scrubber.max_scrub_period_s(), expected, rel_tol=1e-9)

    def test_scrub_overhead_fraction(self) -> None:
        period_s = config.SCRUB_PERIOD_S
        expected = config.TOTAL_BITS / period_s / config.INTERFACE_THROUGHPUT_BPS
        assert math.isclose(
            Scrubber.scrub_overhead_fraction(period_s), expected, rel_tol=1e-9
        )

    def test_configured_period_exceeds_allocated_bandwidth(self) -> None:
        """
        REQ-02 FAIL: the 24 h default scrub period uses more bandwidth than the
        20 % allocation allows.  Max compliant period is ~38.2 h.

        This test documents the known design issue tracked in TASK_TRACKER.md.
        A design change (longer scrub period or larger BW allocation) is needed
        before the requirement can be marked PASS.
        """
        frac = Scrubber.scrub_overhead_fraction(config.SCRUB_PERIOD_S)
        assert frac > config.SCRUB_BANDWIDTH_FRACTION, (
            "REQ-02 is now satisfied — update TASK_TRACKER.md and remove this assertion"
        )
