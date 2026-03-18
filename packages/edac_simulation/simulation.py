"""
simulation.py — Monte Carlo mission simulation runner.

Runs N independent 2-year mission simulations and accumulates statistics on:
  - Corrected errors per scrub pass
  - Uncorrectable page events over the mission
  - Residual bit errors at end of mission

Results are used by analysis.py to produce BER curves and a verification matrix.

Usage:
    python simulation.py                     # run with default config
    python simulation.py --scrub-hours 12   # override scrub period
    python simulation.py --trials 200       # fewer Monte Carlo trials (faster)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from nand_flash_model import NANDFlashModel
from edac import BCHCodec
from scrubbing import Scrubber
from fault_injection import FaultInjector


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_one_trial(
    scrub_period_s: float,
    seed: int,
    seu_rate_bit_s: float = config.SEU_RATE_BIT_S,
) -> dict:
    """
    Run a single 2-year mission simulation.

    Parameters
    ----------
    scrub_period_s : scrubbing interval in seconds
    seed           : RNG seed for this trial
    seu_rate_bit_s : SEU rate per bit per second (defaults to config value)

    Returns
    -------
    dict of trial-level statistics
    """
    rng = np.random.default_rng(seed)
    memory = NANDFlashModel(rng=rng)
    codec = BCHCodec()
    memory.fill_random()

    scrubber = Scrubber(memory, codec)
    injector = FaultInjector(memory, codec, scrubber, rng=rng)

    result = injector.orbital_injection_test(
        duration_s=config.MISSION_DURATION_S,
        scrub_period_s=scrub_period_s,
        seu_rate_bit_s=seu_rate_bit_s,
    )

    # Scale up uncorrectable pages from sim slice to full memory array
    scale = config.TOTAL_BITS / memory.total_bits
    result["uncorr_pages_scaled"] = result["total_uncorr_pages"] * scale
    result["injected_scaled"] = result["total_injected"] * scale

    return result


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------


def run_monte_carlo(
    n_trials: int = config.N_MONTE_CARLO_TRIALS,
    scrub_period_s: float = config.SCRUB_PERIOD_S,
    seu_rate_bit_s: float = config.SEU_RATE_BIT_S,
    base_seed: int = config.RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """
    Run N Monte Carlo trials and aggregate results.

    Returns
    -------
    dict with summary statistics and raw per-trial data
    """
    seeds = base_seed + np.arange(n_trials)
    trials: list[dict] = []

    if verbose:
        print(
            f"\nRunning {n_trials} Monte Carlo trials "
            f"(scrub period = {scrub_period_s / 3600:.1f} h) ..."
        )

    t0 = time.perf_counter()
    bar = tqdm(
        seeds,
        disable=not verbose,
        desc="Trials",
        unit="trial",
        dynamic_ncols=True,
    )
    for seed in bar:
        trial = run_one_trial(
            scrub_period_s=scrub_period_s, seed=int(seed), seu_rate_bit_s=seu_rate_bit_s
        )
        trial.pop("scrub_log", None)
        trials.append(trial)

        # Update postfix with running stats
        n_done = len(trials)
        run_uncorr = sum(t["total_uncorr_pages"] for t in trials)
        run_corr = sum(t["total_corrected"] for t in trials)
        bar.set_postfix(
            corrected=f"{run_corr / n_done:.1e}",
            uncorr_pages=f"{run_uncorr / n_done:.2f}",
            refresh=False,
        )
    elapsed = time.perf_counter() - t0

    # Aggregate
    uncorr_pages = np.array([t["total_uncorr_pages"] for t in trials])
    total_corr = np.array([t["total_corrected"] for t in trials])
    total_inj = np.array([t["total_injected"] for t in trials])

    # Uncorrectable BER over full memory array and full mission
    # BER = uncorrectable bits / total bits
    # Worst-case: assume each uncorrectable page has BCH_T+1 unrecoverable errors
    uncorr_bits_sim = uncorr_pages * (config.BCH_T + 1) * config.SECTORS_PER_PAGE
    scale = config.TOTAL_BITS / config.SIM_TOTAL_BITS
    uncorr_bits_full = uncorr_bits_sim * scale
    ber_full = uncorr_bits_full / config.TOTAL_BITS

    summary = {
        "config": {
            "n_trials": n_trials,
            "scrub_period_h": scrub_period_s / 3600,
            "mission_years": config.MISSION_DURATION_YEARS,
            "bch_t": config.BCH_T,
            "seu_rate_bit_day": seu_rate_bit_s * 86400,
        },
        "elapsed_s": elapsed,
        # -- error statistics (over simulation slice) --
        "mean_injected_per_trial": float(np.mean(total_inj)),
        "mean_corrected_per_trial": float(np.mean(total_corr)),
        "mean_uncorr_pages": float(np.mean(uncorr_pages)),
        "max_uncorr_pages": int(np.max(uncorr_pages)),
        "trials_with_uncorr_event": int(np.sum(uncorr_pages > 0)),
        # -- BER at 64 GB scale --
        "mean_ber_full_memory": float(np.mean(ber_full)),
        "p95_ber_full_memory": float(np.percentile(ber_full, 95)),
        "max_ber_full_memory": float(np.max(ber_full)),
        # -- REQ-06 compliance --
        "req06_threshold": 1e-12,
        "req06_pass": bool(np.max(ber_full) < 1e-12),
        # -- raw arrays for plotting --
        "raw_uncorr_pages": uncorr_pages.tolist(),
        "raw_ber_full": ber_full.tolist(),
        "raw_corrected": total_corr.tolist(),
    }

    if verbose:
        _print_summary(summary)

    return summary


# ---------------------------------------------------------------------------
# Scrub-period sweep
# ---------------------------------------------------------------------------


def scrub_period_sweep(
    periods_hours: list[float] | None = None,
    n_trials: int = 100,
    seu_rate_bit_s: float = config.SEU_RATE_BIT_S,
    base_seed: int = config.RANDOM_SEED,
) -> list[dict]:
    """
    Run Monte Carlo simulations for multiple scrub periods.

    Returns a list of summary dicts, one per period.
    """
    if periods_hours is None:
        periods_hours = [1, 6, 12, 24, 48, 72]

    sweep_results = []
    for ph in periods_hours:
        summary = run_monte_carlo(
            n_trials=n_trials,
            scrub_period_s=ph * 3600,
            seu_rate_bit_s=seu_rate_bit_s,
            base_seed=base_seed,
            verbose=False,
        )
        summary["scrub_period_h"] = ph
        sweep_results.append(summary)
        print(
            f"  Period {ph:4.0f}h — mean BER: {summary['mean_ber_full_memory']:.2e} "
            f"— REQ-06: {'PASS' if summary['req06_pass'] else 'FAIL'}"
        )

    return sweep_results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 60)
    print("Monte Carlo Simulation Summary")
    print("=" * 60)
    c = s["config"]
    print(f"  Trials:           {c['n_trials']}")
    print(f"  Scrub period:     {c['scrub_period_h']:.1f} h")
    print(f"  Mission:          {c['mission_years']} years")
    print(f"  BCH(t):           {c['bch_t']}")
    print(f"  SEU rate:         {c['seu_rate_bit_day']:.2e} /bit/day")
    print(f"  Elapsed:          {s['elapsed_s']:.1f} s")
    print()
    print(f"  Mean injected SEUs/trial (slice):  {s['mean_injected_per_trial']:.2e}")
    print(f"  Mean corrected/trial (slice):      {s['mean_corrected_per_trial']:.2e}")
    print(f"  Mean uncorrectable pages/trial:    {s['mean_uncorr_pages']:.2e}")
    print(f"  Trials with any uncorr. event:     {s['trials_with_uncorr_event']}")
    print()
    total_gb = config.NAND_TOTAL_CAPACITY_GB
    print(f"  BER ({total_gb} GB, mean):   {s['mean_ber_full_memory']:.2e}")
    print(f"  BER ({total_gb} GB, P95):    {s['p95_ber_full_memory']:.2e}")
    print(f"  BER ({total_gb} GB, max):    {s['max_ber_full_memory']:.2e}")
    print()
    status = "PASS" if s["req06_pass"] else "FAIL"
    print(f"  REQ-06 (BER < 1e-12):  {status}")
    print("=" * 60)


def save_results(summary: dict, path: str | Path = "results/monte_carlo.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDAC Monte Carlo Simulation")
    parser.add_argument("--trials", type=int, default=config.N_MONTE_CARLO_TRIALS)
    parser.add_argument("--scrub-hours", type=float, default=config.SCRUB_PERIOD_HOURS)
    parser.add_argument(
        "--seu-rate-day",
        type=float,
        default=config.SEU_RATE_BIT_DAY,
        help="SEU rate per bit per day (overrides config)",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Sweep scrub periods instead of single run"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--output", type=str, default="results/monte_carlo.json")
    args = parser.parse_args()

    seu_rate_bit_s = args.seu_rate_day / 86400

    if args.sweep:
        print("Running scrub-period sweep ...")
        sweep = scrub_period_sweep(
            n_trials=args.trials, seu_rate_bit_s=seu_rate_bit_s, base_seed=args.seed
        )
        save_results({"sweep": sweep}, args.output)
    else:
        summary = run_monte_carlo(
            n_trials=args.trials,
            scrub_period_s=args.scrub_hours * 3600,
            seu_rate_bit_s=seu_rate_bit_s,
            base_seed=args.seed,
        )
        save_results(summary, args.output)
