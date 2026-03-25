"""
fault_injection.py — SEU fault injection and stress testing of the EDAC algorithm.

Two modes:
  1. Orbital mode: inject SEU events at the orbital rate derived from SPENVIS
     (Poisson process parameterised by config values).
  2. Sweep mode: inject at controlled rates to find the algorithm's breaking
     point (REQ-07 / assignment task "Fault Injection Testing").

The breaking point is defined as the injected BER at which the fraction of
uncorrectable pages exceeds a threshold (e.g. 1 %).
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

import config
from bad_block_manager import BadBlockManager
from nand_flash_model import NANDFlashModel
from edac import BCHCodec
from radiation_model import RadiationModel
from scrubbing import Scrubber


# ---------------------------------------------------------------------------
# Core fault injector
# ---------------------------------------------------------------------------


class FaultInjector:
    """
    Injects bit-flip errors into a NANDFlashModel and evaluates EDAC recovery.

    Parameters
    ----------
    memory  : NANDFlashModel
    codec   : BCHCodec
    scrubber: Scrubber
    rng     : Generator
    """

    def __init__(
        self,
        memory: NANDFlashModel,
        codec: BCHCodec,
        scrubber: Scrubber,
        rng: Generator | None = None,
    ) -> None:
        self.memory = memory
        self.codec = codec
        self.scrubber = scrubber
        self.rng = rng if rng is not None else np.random.default_rng(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # Inject at a specific BER
    # ------------------------------------------------------------------

    def inject_at_ber(self, ber: float) -> int:
        """
        Inject bit flips into the entire memory at the given BER.

        Parameters
        ----------
        ber : float
            Target bit error rate (probability of each bit being flipped).

        Returns
        -------
        n_injected : int
            Actual number of bits flipped.
        """
        n_flips = self.rng.binomial(self.memory.total_bits, ber)
        if n_flips == 0:
            return 0
        addresses = self.rng.integers(0, self.memory.total_bits, size=n_flips)
        # Deduplicate: two flips at the same address cancel out
        unique_addrs, counts = np.unique(addresses, return_counts=True)
        net_flips = unique_addrs[counts % 2 == 1]
        self.memory.inject_bit_flips(net_flips)
        return len(net_flips)

    # ------------------------------------------------------------------
    # Single-trial evaluation
    # ------------------------------------------------------------------

    def evaluate_single_pass(self) -> dict:
        """
        Run one scrub pass and measure EDAC performance.

        Returns
        -------
        dict with:
            errors_before        : int   — errors in memory before scrubbing
            errors_corrected     : int   — errors successfully corrected
            uncorrectable_pages  : int   — pages the EDAC could not fix
            residual_errors      : int   — errors remaining after scrub
        """
        errors_before = self.memory.total_error_count()
        result = self.scrubber.scrub_pass()
        residual = self.memory.total_error_count()

        return {
            "errors_before": errors_before,
            "errors_corrected": result["errors_corrected"],
            "uncorrectable_pages": result["uncorrectable_pages"],
            "residual_errors": residual,
        }

    # ------------------------------------------------------------------
    # Breaking-point sweep
    # ------------------------------------------------------------------

    def breaking_point_sweep(
        self,
        scrub_periods_s: np.ndarray | None = None,
        seu_rate_bit_s: float = config.SEU_RATE_BIT_S,
        n_trials_per_period: int = 50,
        uncorr_threshold: float = 0.01,
    ) -> list[dict]:
        """
        Sweep scrub period at a fixed SEU rate and find the period at which EDAC fails.

        For each scrub period:
          1. Reset memory to clean state with random data.
          2. Inject errors accumulated over one scrub period:
             p_flip = seu_rate_bit_s * scrub_period_s
          3. Run one scrub pass.
          4. Record fraction of uncorrectable pages.

        The breaking point is the scrub period at which the mean fraction of
        uncorrectable pages exceeds uncorr_threshold.

        Parameters
        ----------
        scrub_periods_s   : array of scrub periods to sweep [s].
                            Defaults to log-space from 1 h up to 10x the
                            analytical BCH capacity limit.
        seu_rate_bit_s    : SEU rate [bit⁻¹ s⁻¹] (default: config value)
        n_trials_per_period : independent trials per period point
        uncorr_threshold  : uncorrectable page fraction that defines "failure"

        Returns
        -------
        list of dicts, one per period, with keys:
            scrub_period_s, p_flip, seu_rate_bit_s,
            mean_uncorr_fraction, std_uncorr_fraction,
            mean_corrected, breaking_point (bool)
        """
        if scrub_periods_s is None:
            # Auto-scale upper bound to 10x the analytical breaking-point period
            # (period at which expected errors/sector = BCH_T)
            sector_bits = config.SECTOR_DATA_BYTES * 8
            breaking_estimate_s = config.BCH_T / (sector_bits * seu_rate_bit_s)
            max_s = min(10 * breaking_estimate_s, 365 * 24 * 3600)
            scrub_periods_s = np.logspace(np.log10(3600), np.log10(max_s), 40)

        results = []
        breaking_found = False

        for period_s in scrub_periods_s:
            # Accumulated bit-flip probability over one scrub period
            p_flip = seu_rate_bit_s * period_s
            uncorr_fractions = []
            corrected_counts = []

            for _ in range(n_trials_per_period):
                self.memory.reset()
                self.memory.fill_random()
                self.scrubber._ecc_store = self.scrubber._initial_encode_all()

                self.inject_at_ber(p_flip)
                r = self.evaluate_single_pass()

                frac_uncorr = (
                    r["uncorrectable_pages"] / self.memory.n_pages
                    if self.memory.n_pages > 0
                    else 0.0
                )
                uncorr_fractions.append(frac_uncorr)
                corrected_counts.append(r["errors_corrected"])

            mean_uncorr = float(np.mean(uncorr_fractions))
            std_uncorr = float(np.std(uncorr_fractions))
            mean_corr = float(np.mean(corrected_counts))
            is_breaking = (mean_uncorr > uncorr_threshold) and not breaking_found
            if is_breaking:
                breaking_found = True

            results.append(
                {
                    "scrub_period_s": period_s,
                    "p_flip": p_flip,
                    "seu_rate_bit_s": seu_rate_bit_s,
                    "mean_uncorr_fraction": mean_uncorr,
                    "std_uncorr_fraction": std_uncorr,
                    "mean_corrected": mean_corr,
                    "breaking_point": is_breaking,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Orbital-rate simulation (time-stepped)
    # ------------------------------------------------------------------

    def orbital_injection_test(
        self,
        duration_s: float = config.MISSION_DURATION_S,
        scrub_period_s: float = config.SCRUB_PERIOD_S,
        seu_rate_bit_s: float = config.SEU_RATE_BIT_S,
        bbm: BadBlockManager | None = None,
    ) -> dict:
        """
        Simulate the full mission at the orbital SEU rate with periodic scrubbing.

        Parameters
        ----------
        duration_s     : mission duration in seconds
        scrub_period_s : scrubbing period in seconds

        Returns
        -------
        dict with mission-level statistics
        """
        rad_model = RadiationModel(
            total_bits=self.memory.total_bits,
            seu_rate_bit_s=seu_rate_bit_s,
            rng=self.rng,
        )

        self.memory.reset()
        self.memory.fill_random()
        self.scrubber._ecc_store = self.scrubber._initial_encode_all()

        t = 0.0
        total_injected = 0
        total_corrected = 0
        total_uncorr_pages = 0
        n_scrub_passes = 0
        scrub_log: list[dict] = []

        while t < duration_s:
            # Advance to next scrub event
            dt = min(scrub_period_s, duration_s - t)
            events = rad_model.generate_events(dt)
            self.memory.inject_bit_flips(events)
            total_injected += len(events)
            t += dt

            # Scrub
            result = self.scrubber.scrub_pass()
            total_corrected += result["errors_corrected"]
            total_uncorr_pages += result["uncorrectable_pages"]
            n_scrub_passes += 1

            # Notify BBM of any uncorrectable pages found this pass
            if bbm is not None and result["uncorrectable_pages"] > 0:
                scale = config.TOTAL_PAGES / self.memory.n_pages
                bbm.register_uncorrectable(result["uncorrectable_pages"], scale)

            scrub_log.append(
                {
                    "time_s": t,
                    "errors_corrected": result["errors_corrected"],
                    "uncorr_pages": result["uncorrectable_pages"],
                }
            )

        return {
            "duration_s": duration_s,
            "scrub_period_s": scrub_period_s,
            "n_scrub_passes": n_scrub_passes,
            "total_injected": total_injected,
            "total_corrected": total_corrected,
            "total_uncorr_pages": total_uncorr_pages,
            "residual_errors": self.memory.total_error_count(),
            "scrub_log": scrub_log,
            # BER [/bit/s] = uncorrectable bits / total bits / duration_s
            "uncorrectable_ber": (
                total_uncorr_pages
                * config.BCH_T
                * config.SECTORS_PER_PAGE
                / (self.memory.total_bits * duration_s)
            )
            if total_uncorr_pages > 0
            else 0.0,
            "bbm_summary": bbm.summary if bbm is not None else None,
        }
