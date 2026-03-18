"""
radiation_model.py — SEU event generator for the 600 km SSO radiation environment.

Models Single Event Upsets as a Poisson process with rate derived from
SPENVIS analysis (SEU_RATE_BIT_DAY in config.py).

References:
  - SPENVIS AP-8/AP-9 proton flux model (600 km, 97.8 deg inclination)
  - ECSS-E-ST-10-12C, On-board Data Handling
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

import config


class RadiationModel:
    """
    Generates SEU events for a memory array over a simulated time span.

    Parameters
    ----------
    total_bits : int
        Number of bits in the simulated memory slice.
    seu_rate_bit_s : float
        SEU rate per bit per second.  Defaults to config value.
    rng : numpy Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        total_bits: int = config.SIM_TOTAL_BITS,
        seu_rate_bit_s: float = config.SEU_RATE_BIT_S,
        rng: Generator | None = None,
    ) -> None:
        self.total_bits = total_bits
        self.seu_rate_bit_s = seu_rate_bit_s
        self.rng = rng if rng is not None else np.random.default_rng(config.RANDOM_SEED)

        # Total SEU rate for the whole array [events/s]
        self.array_seu_rate_s = seu_rate_bit_s * total_bits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_events(self, duration_s: float) -> np.ndarray:
        """
        Return an array of bit addresses that are flipped during *duration_s*.

        Uses a Poisson process: number of events is Poisson-distributed with
        mean = array_seu_rate_s * duration_s; event locations are uniform
        over [0, total_bits).

        Returns
        -------
        np.ndarray, shape (N,), dtype int64
            Bit addresses of SEU events.  May be empty if no events occur.
        """
        expected_events = self.array_seu_rate_s * duration_s
        n_events = self.rng.poisson(expected_events)
        if n_events == 0:
            return np.empty(0, dtype=np.int64)
        bit_addresses = self.rng.integers(0, self.total_bits, size=n_events)
        return bit_addresses

    def expected_seus_per_page_per_scrub(self, scrub_period_s: float) -> float:
        """
        Expected number of SEUs in a single NAND page between two scrub cycles.

        Useful for checking whether the chosen BCH(t) is sufficient.
        """
        bits_per_page = config.PAGE_DATA_BYTES * 8
        return self.seu_rate_bit_s * bits_per_page * scrub_period_s


if __name__ == "__main__":
    rm = RadiationModel()
    print("=" * 60)
    print("Radiation Model Summary")
    print("=" * 60)
    print(f"  Orbit:              {config.ORBIT_ALTITUDE_KM} km SSO")
    print(f"  Mission duration:   {config.MISSION_DURATION_YEARS} years")
    print(f"  SEU rate (per bit): {config.SEU_RATE_BIT_S:.2e} /bit/s")
    print(f"                      {config.SEU_RATE_BIT_DAY:.2e} /bit/day")
    print(f"  Total bits:         {config.TOTAL_BITS:.2e}")
    print(
        f"  Sim slice:          {config.SIM_MEMORY_BYTES / 1024} kB = {config.SIM_TOTAL_BITS:.2e} bits"
    )
    print()
    for period_h in [1, 6, 24, 72]:
        rate = rm.expected_seus_per_page_per_scrub(period_h * 3600)
        print(
            f"  SEUs/page per {period_h:3d}h scrub: {rate:.3e}  "
            f"({'OK' if rate < config.BCH_T else 'EXCEEDS BCH(t)!'}  t={config.BCH_T})"
        )
    print("=" * 60)
