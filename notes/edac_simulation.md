# EDAC Simulation — Implementation Notes

## What was built

A fully runnable Python simulation for verifying the EDAC design of a 64 GB
NAND Flash mass memory subsystem for a 600 km SSO CubeSat (AE4904 assignment).

Located in `packages/edac_simulation/`. Run with `uv run simulation.py` from
within the package directory.

---

## File overview

| File | Purpose |
|---|---|
| `config.py` | Single source of truth for all parameters (mission, memory, EDAC, sim) |
| `radiation_model.py` | Poisson SEU event generator |
| `nand_flash_model.py` | Page-level NAND memory model (numpy bool array, tracks ground truth) |
| `edac/bch.py` | BCH encoder/decoder — sim mode (fast) and galois mode (real polynomial) |
| `scrubbing.py` | Periodic scrub pass; scrub-period trade-off table |
| `fault_injection.py` | Orbital-rate injection and breaking-point BER sweep |
| `simulation.py` | Monte Carlo runner (tqdm progress bar, live stats postfix) |
| `analysis.py` | Plots: BER vs scrub period, error accumulation, breaking point; verification matrix |

---

## Memory chip — MT29F256G08AUCABH3

Confirmed from datasheet (Rev. H, p. 1):

| Parameter | Value | Source |
|---|---|---|
| Page data bytes | 8192 | Datasheet |
| OOB bytes per page | 448 | Datasheet |
| Pages per block | 128 | Datasheet |
| Blocks per chip (256Gb) | 32,768 | Datasheet |
| Chips for 64 GB | 2 | 64 / 32 GB |

OOB per sector = 448 / 16 = **28 bytes** available for ECC.

---

## BCH parameter space

With GF(2¹³) and 512-byte sectors, ECC overhead = ⌈13·t / 8⌉ bytes:

| t | ECC bytes/sector | Fits in 28 B OOB? |
|---|---|---|
| 4 | 7 | ✓ (current default; Micron minimum) |
| 8 | 13 | ✓ |
| 12 | 20 | ✓ |
| 16 | 26 | ✓ |
| 17 | 28 | ✓ (maximum) |
| 18 | 30 | ✗ |

For this SEU environment the EDAC constraint on scrub period is dominated by
bandwidth, not by t. The scrub period limit from EDAC alone is ~1,600 h at t=4
and effectively unbounded at t ≥ 8 — well above the bandwidth limit of ~38 h.
Choosing t > 4 provides margin, not a shorter required scrub period.

---

## Key design decisions

- **BCH(t=4) per 512-byte sector** — default; meets Micron's minimum ECC
  requirement. 16 sectors per 8192-byte page.
- **Flat page-level memory model** — blocks, planes, and chips are not
  simulated. Physical hierarchy has no bearing on SEU statistics,
  BCH correctability, or scrubbing outcome for this analysis.
  (A structured model would only be needed for MBU clustering, retention
  errors, or wear — all out of scope.)
- **SLC NAND** chosen over MLC/TLC for better radiation tolerance.
- **Scrubbing period default: 24 h** — binding constraint is bandwidth
  (max ~38 h at 20% of 20 Mbps interface). EDAC is not the binding constraint.

---

## Sim mode vs. galois mode

The `galois` BCH implementation over GF(2¹³) takes ~288 ms per call — too
slow for Monte Carlo. `BCHCodec` has two modes:

- **`sim` mode** (default `True`) — stores the reference sector as the "ECC"
  at encode time; decode is a numpy XOR comparison. **~8.6 µs per call**
  (33,000× faster). Correct for BER simulation: correctability depends only
  on error count vs. t.
- **`galois` mode** (`BCHCodec(sim_mode=False)`) — full polynomial arithmetic.
  Use to unit-test the real codec on specific injected patterns.

---

## Radiation environment

Two SPENVIS values are defined in `config.py`; select the active one:

```python
SEU_RATE_BIT_DAY_V1 = 1.4e-7      # initial estimate
SEU_RATE_BIT_DAY_V2 = 2.8775e-6   # revised value

SEU_RATE_BIT_DAY = SEU_RATE_BIT_DAY_V2   # <-- active
```

Both can also be passed at runtime via `--seu-rate-day` (see below).

---

## How to run

```bash
# Scrub period trade-off table (bandwidth + SEU accumulation)
uv run --directory packages/edac_simulation scrubbing.py
uv run --directory packages/edac_simulation scrubbing.py --seu-rate-day 1.4e-7

# Single Monte Carlo run (default config)
uv run --directory packages/edac_simulation simulation.py --trials 10

# Override scrub period or SEU rate
uv run --directory packages/edac_simulation simulation.py --trials 200 --scrub-hours 48
uv run --directory packages/edac_simulation simulation.py --trials 200 --seu-rate-day 1.4e-7

# Sweep scrub periods
uv run --directory packages/edac_simulation simulation.py --sweep --trials 100

# Plots from saved results / quick inline demo
uv run --directory packages/edac_simulation analysis.py --results results/monte_carlo.json
uv run --directory packages/edac_simulation analysis.py --quick
```

For the report, use at least a few hundred trials so BER percentiles are meaningful.

---

## Current status

- All placeholder values removed; config uses confirmed datasheet and SPENVIS values.
- No open TODOs in config.
- Active SEU rate: V2 = 2.8775×10⁻⁶ /bit/day.
- Default scrub period: 24 h, BCH(t=4).
- REQ-06 (BER < 10⁻¹²): **PASS** at default settings (both SEU rates).
