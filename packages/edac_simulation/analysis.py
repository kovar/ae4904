"""
analysis.py — Results analysis, plotting, and verification matrix.

Produces all figures referenced in the final report:
  1. BER vs. scrubbing period (REQ-06 compliance curve)
  2. Error accumulation over mission lifetime (one trial, with scrub events)
  3. Fault injection breaking-point curve
  4. Verification matrix (console + CSV)

Usage:
    python analysis.py --results results/monte_carlo.json   # plot saved results
    python analysis.py --quick                              # run small sim + plot inline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

import config

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "figure.dpi": 120,
    }
)

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


# ---------------------------------------------------------------------------
# Figure 1: BER vs. scrub period
# ---------------------------------------------------------------------------


def plot_ber_vs_scrub_period(
    sweep_results: list[dict], save: bool = True
) -> plt.Figure:
    """
    Plot mean and P95 uncorrectable BER vs. scrubbing period.
    Adds a horizontal line at the REQ-06 limit (10^-12).
    """
    periods = [r["scrub_period_h"] for r in sweep_results]
    ber_mean = [r["mean_ber_full_memory"] for r in sweep_results]
    ber_p95 = [r["p95_ber_full_memory"] for r in sweep_results]
    req_pass = [r["req06_pass"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    total_gb = config.NAND_TOTAL_CAPACITY_GB
    ax.semilogy(
        periods, ber_mean, "o-", label=f"Mean BER ({total_gb} GB)", color="steelblue"
    )
    ax.semilogy(
        periods, ber_p95, "s--", label=f"P95 BER ({total_gb} GB)", color="darkorange"
    )
    ax.axhline(
        config.BER_REQUIREMENT_BIT_S,
        color="red",
        linewidth=1.5,
        linestyle=":",
        label=f"REQ-06 limit ({config.BER_REQUIREMENT_BIT_S:.0e} bit⁻¹ s⁻¹)",
    )

    # Shade the passing region
    ax.fill_between(
        periods,
        [1e-18] * len(periods),
        [1e-12] * len(periods),
        alpha=0.08,
        color="green",
        label="REQ-06 compliant region",
    )

    # Mark failing points
    for p, bm, ok in zip(periods, ber_mean, req_pass):
        marker = "x" if not ok else ""
        if marker:
            ax.plot(p, bm, "rx", markersize=10, markeredgewidth=2)

    ax.set_xlabel("Scrubbing Period [hours]")
    ax.set_ylabel("Uncorrectable BER [bit⁻¹ s⁻¹]")
    ax.set_title(
        f"Uncorrectable BER vs. Scrubbing Period\n"
        f"BCH(t={config.BCH_T}), {config.ORBIT_ALTITUDE_KM} km SSO, SEU rate = "
        f"{config.SEU_RATE_BIT_DAY:.1e} /bit/day"
    )
    ax.legend()
    ax.set_xlim(left=0)

    fig.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "ber_vs_scrub_period.pdf")
        fig.savefig(PLOTS_DIR / "ber_vs_scrub_period.png")
        print("Saved: ber_vs_scrub_period.{pdf,png}")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Error accumulation over mission (single trial timeline)
# ---------------------------------------------------------------------------


def plot_error_accumulation(scrub_log: list[dict], save: bool = True) -> plt.Figure:
    """
    Plot corrected errors and uncorrectable pages per scrub pass vs. mission time.

    Parameters
    ----------
    scrub_log : list of dicts with keys 'time_s', 'errors_corrected', 'uncorr_pages'
    """
    times_days = np.array([e["time_s"] for e in scrub_log]) / 86400
    corrected = np.array([e["errors_corrected"] for e in scrub_log])
    uncorr = np.array([e["uncorr_pages"] for e in scrub_log])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(times_days, corrected, linewidth=0.8, color="steelblue")
    ax1.set_ylabel("Errors Corrected per Scrub Pass")
    ax1.set_title("Error Accumulation Over Mission Lifetime (single trial)")

    ax2.step(times_days, uncorr, linewidth=0.8, color="crimson", where="post")
    ax2.set_ylabel("Uncorrectable Pages per Scrub Pass")
    ax2.set_xlabel("Mission Time [days]")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for ax in (ax1, ax2):
        ax.set_xlim(0, config.MISSION_DURATION_DAYS)

    fig.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "error_accumulation.pdf")
        fig.savefig(PLOTS_DIR / "error_accumulation.png")
        print("Saved: error_accumulation.{pdf,png}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Fault injection breaking-point curve
# ---------------------------------------------------------------------------


def plot_breaking_point(sweep_data: list[dict], save: bool = True) -> plt.Figure:
    """
    Plot fraction of uncorrectable pages vs. scrub period.

    Parameters
    ----------
    sweep_data : output of FaultInjector.breaking_point_sweep()
    """
    periods_h = np.array([d["scrub_period_s"] / 3600 for d in sweep_data])
    uncorr_frac = np.array([d["mean_uncorr_fraction"] for d in sweep_data])
    uncorr_std = np.array([d["std_uncorr_fraction"] for d in sweep_data])
    seu_rate = sweep_data[0]["seu_rate_bit_s"]

    # Bandwidth limit: max period at which a full scrub pass fits within the
    # allocated bandwidth fraction (same formula as REQ-02 check)
    bw_limit_h = (
        config.TOTAL_BITS
        / (config.INTERFACE_THROUGHPUT_BPS * config.SCRUB_BANDWIDTH_FRACTION)
        / 3600
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(
        periods_h,
        uncorr_frac * 100,
        "o-",
        color="steelblue",
        label="Mean uncorrectable page fraction",
    )
    ax.fill_between(
        periods_h,
        (uncorr_frac - uncorr_std) * 100,
        (uncorr_frac + uncorr_std) * 100,
        alpha=0.2,
        color="steelblue",
    )

    # Mark the breaking point
    for d in sweep_data:
        if d["breaking_point"]:
            bp_h = d["scrub_period_s"] / 3600
            ax.axvline(
                bp_h,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"EDAC breaking point ≈ {bp_h:.0f} h",
            )
            break

    # Design scrub period
    ax.axvline(
        config.SCRUB_PERIOD_HOURS,
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=f"Design scrub period ({config.SCRUB_PERIOD_HOURS:.0f} h)",
    )

    # Bandwidth limit
    ax.axvline(
        bw_limit_h,
        color="goldenrod",
        linestyle="-.",
        linewidth=1.5,
        label=f"Bandwidth limit ({bw_limit_h:.1f} h, REQ-02)",
    )

    ax.axhline(
        1.0, color="orange", linestyle=":", linewidth=1, label="1 % failure threshold"
    )
    ax.set_xlabel("Scrub Period [hours]")
    ax.set_ylabel("Uncorrectable Pages [%]")
    ax.set_title(
        f"EDAC Breaking-Point vs. Scrub Period\n"
        f"BCH(t={config.BCH_T}) / {config.SECTOR_DATA_BYTES}-byte sector, "
        f"SEU rate = {seu_rate:.1e} bit⁻¹ s⁻¹"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "breaking_point.pdf")
        fig.savefig(PLOTS_DIR / "breaking_point.png")
        print("Saved: breaking_point.{pdf,png}")
    return fig


# ---------------------------------------------------------------------------
# Verification matrix
# ---------------------------------------------------------------------------


def verification_matrix(monte_carlo_summary: dict, save: bool = True) -> pd.DataFrame:
    """
    Generate a requirements verification matrix and print/save it.

    REQ-05 and REQ-06 are verified against simulation results.
    REQ-07 is verified by the existence of the simulation scripts themselves.
    """
    s = monte_carlo_summary
    max_ber = s.get("max_ber_full_memory", float("nan"))

    scrub_bw_fraction = config.TOTAL_BITS / (
        config.SCRUB_PERIOD_S * config.INTERFACE_THROUGHPUT_BPS
    )
    rows = [
        {
            "Requirement": "REQ-01",
            "Description": "64 GB usable storage",
            "Verification Method": "Component selection + datasheet",
            "Result / Evidence": (
                f"{config.NAND_NUM_CHIPS}x Micron MT29F256G08 = "
                f"{config.NAND_TOTAL_CAPACITY_GB} GB"
            ),
            "Status": "PASS",
        },
        {
            "Requirement": "REQ-02",
            "Description": "20 Mbps net payload throughput after EDAC overhead",
            "Verification Method": "Bandwidth budget (simulation config)",
            "Result / Evidence": (
                f"Scrubbing uses {scrub_bw_fraction:.1%} of {config.INTERFACE_THROUGHPUT_BPS / 1e6:.0f} Mbps "
                f"at {config.SCRUB_PERIOD_HOURS:.0f}h scrub period "
                f"(allocated: {config.SCRUB_BANDWIDTH_FRACTION:.0%}; "
                f"max allowed period: {config.TOTAL_BITS / (config.INTERFACE_THROUGHPUT_BPS * config.SCRUB_BANDWIDTH_FRACTION) / 3600:.1f}h)"
            ),
            "Status": "PASS"
            if scrub_bw_fraction <= config.SCRUB_BANDWIDTH_FRACTION
            else "FAIL",
        },
        {
            "Requirement": "REQ-03",
            "Description": "PCB within PC-104 / <90x90x15 mm",
            "Verification Method": "KiCad PCB layout (external)",
            "Result / Evidence": "Verified in KiCad design files (outside simulation scope)",
            "Status": "TBD",
        },
        {
            "Requirement": "REQ-04",
            "Description": "SEL detection + power cut within 10 µs",
            "Verification Method": "LTSpice/NGSpice transient simulation (external)",
            "Result / Evidence": "Verified in circuit simulation files (outside simulation scope)",
            "Status": "TBD",
        },
        {
            "Requirement": "REQ-05",
            "Description": "EDAC corrects expected bit-flip rate for target orbit",
            "Verification Method": "Simulation (fault injection + scrubbing)",
            "Result / Evidence": (
                f"BCH(t={config.BCH_T}) corrected {s.get('mean_corrected_per_trial', 0):.1e} "
                f"errors/trial on average"
            ),
            "Status": "PASS" if s.get("mean_corrected_per_trial", 0) > 0 else "TBD",
        },
        {
            "Requirement": "REQ-06 (BBM)",
            "Description": "Bad Block Management table for damaged sectors",
            "Verification Method": f"Monte Carlo simulation ({s['config']['n_trials']} trials)",
            "Result / Evidence": (
                f"Factory bad ≈ {s.get('bbm_mean_factory_bad', 0):.0f} blocks "
                f"({s.get('bbm_mean_factory_bad', 0) / (config.BLOCKS_PER_CHIP * config.NAND_NUM_CHIPS) * 100:.1f}% "
                f"of {config.BLOCKS_PER_CHIP * config.NAND_NUM_CHIPS} total); "
                f"runtime bad ≤ {s.get('bbm_max_runtime_bad', 0)} blocks; "
                f"spare pool: {config.BBM_SPARE_BLOCKS_PER_CHIP} blocks/chip "
                f"({config.BBM_SPARE_FRACTION:.0%} reserved)"
            ),
            "Status": "PASS" if not s.get("bbm_any_exhausted", True) else "FAIL",
        },
        {
            "Requirement": "REQ-06 (BER)",
            "Description": f"Uncorrectable BER < {config.BER_REQUIREMENT_BIT_S:.0e} bit⁻¹ s⁻¹",
            "Verification Method": f"Monte Carlo simulation ({s['config']['n_trials']} trials)",
            "Result / Evidence": (
                f"Max BER = {max_ber:.2e} bit⁻¹ s⁻¹ "
                f"({config.MISSION_DURATION_YEARS}-year mission, "
                f"{config.NAND_TOTAL_CAPACITY_GB} GB scaled)"
            ),
            "Status": "PASS" if s.get("req06_pass", False) else "FAIL",
        },
        {
            "Requirement": "REQ-07",
            "Description": "Design verified via simulation reproducing SEU/SEL faults",
            "Verification Method": "Code review + reproducibility",
            "Result / Evidence": (
                "Python scripts in edac_simulation/; seed-locked RNG; "
                "SEU injection + scrubbing demonstrated"
            ),
            "Status": "PASS",
        },
    ]

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("Verification Matrix")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS_DIR / "verification_matrix.csv", index=False)
        print("Saved: results/verification_matrix.csv")

    return df


# ---------------------------------------------------------------------------
# Quick demo run (no pre-saved results needed)
# ---------------------------------------------------------------------------


def quick_demo() -> None:
    """Run a small simulation inline and produce all plots."""
    from simulation import run_monte_carlo, scrub_period_sweep
    from fault_injection import FaultInjector
    from nand_flash_model import NANDFlashModel
    from edac import BCHCodec
    from scrubbing import Scrubber

    print("Quick demo: 50-trial Monte Carlo with default scrub period ...")
    summary = run_monte_carlo(n_trials=50, scrub_period_s=config.SCRUB_PERIOD_S)
    verification_matrix(summary)
    if summary.get("scrub_log"):
        plot_error_accumulation(summary["scrub_log"])

    print("\nScrub-period sweep (3 periods, 20 trials each) ...")
    sweep = scrub_period_sweep(periods_hours=[6, 24, 72], n_trials=20)
    plot_ber_vs_scrub_period(sweep)

    print("\nFault injection breaking-point sweep ...")
    rng = np.random.default_rng(config.RANDOM_SEED)
    memory = NANDFlashModel(rng=rng)
    codec = BCHCodec()
    memory.fill_random()
    scrubber = Scrubber(memory, codec)
    injector = FaultInjector(memory, codec, scrubber, rng=rng)
    bp_data = injector.breaking_point_sweep(
        seu_rate_bit_s=config.SEU_RATE_BIT_S,
        n_trials_per_period=10,
    )
    plot_breaking_point(bp_data)

    plt.show()
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDAC Simulation Analysis & Plots")
    parser.add_argument(
        "--results", type=str, default=None, help="Path to saved monte_carlo.json"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run a quick inline demo simulation"
    )
    args = parser.parse_args()

    if args.quick:
        quick_demo()
    elif args.results:
        with open(args.results) as f:
            data = json.load(f)
        if "sweep" in data:
            plot_ber_vs_scrub_period(data["sweep"])
        else:
            verification_matrix(data)
            if data.get("scrub_log"):
                plot_error_accumulation(data["scrub_log"])
        plt.show()
    else:
        print("Use --quick for a demo run, or --results <file> to plot saved data.")
