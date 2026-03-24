# Test Suite — `packages/edac_simulation/tests/`

Unit and analytical verification tests for the NAND Flash EDAC simulation.
All tests use fixed random seeds and tiny in-memory objects so the full suite
runs in a few seconds on any laptop.

## Running the tests

```bash
# Fast tests only (default for pre-commit)
uv run pytest -m "not slow" -v

# Full suite including slow cross-validation tests
uv run pytest -v

# Single module
uv run pytest packages/edac_simulation/tests/test_bch.py -v
```

## Test files

| File | Module under test | What is verified |
|------|-------------------|-----------------|
| `test_config.py` | `config.py` | All derived parameters match hand-calculated values; physical constraints satisfied (OOB capacity, mission arithmetic) |
| `test_nand_flash_model.py` | `nand_flash_model.py` | Initialisation, write/read round-trips, single and bulk bit-flip injection, duplicate-address cancellation, `fill_random`, `reset`, and error-query helpers |
| `test_bch.py` | `edac/bch.py` | Encode/decode round-trips; correctability boundary at t and t+1; page-level encode/decode; `@pytest.mark.slow` cross-validation against real GF polynomial arithmetic |
| `test_radiation_model.py` | `radiation_model.py` | Poisson event generation (zero duration, address bounds, dtype); mean convergence against closed-form λ; `expected_seus_per_page_per_scrub` formula |
| `test_scrubbing.py` | `scrubbing.py` | Clean pass, single-error and T-error correction, T+1 uncorrectable, post-scrub memory state, cross-page correction, cumulative statistics, and static bandwidth-analysis helpers |
| `test_fault_injection.py` | `fault_injection.py` | `inject_at_ber` (zero, statistical bound, high rate); `evaluate_single_pass` key structure and postconditions; `orbital_injection_test` structure and scrub-pass count |
| `test_analytical.py` | `simulation.py`, `radiation_model.py` | Deterministic reproducibility (same seed → identical output); Poisson mean and variance convergence; confirmation that BCH(t=4) is sufficient for the current radiation environment |

## Shared fixtures (`conftest.py`)

| Fixture | Type | Description |
|---------|------|-------------|
| `rng` | `np.random.Generator` | Fixed seed (42) for deterministic tests |
| `codec` | `BCHCodec` | BCH(t=4), 512-byte sector, **sim mode** (fast) |
| `galois_codec` | `BCHCodec` | Same parameters, **galois mode** (real polynomial arithmetic, slow) |
| `memory` | `NANDFlashModel` | 4 pages, all-zero, fresh for each test |
| `filled_memory` | `NANDFlashModel` | 4 pages pre-filled with random data (reference == stored, zero errors) |
| `scrubber` | `Scrubber` | Attached to `filled_memory` with pre-computed ECC for all pages |

## Test markers

| Marker | Usage |
|--------|-------|
| `slow` | BCH cross-validation tests that invoke real GF polynomial arithmetic (~300 ms each). Deselect with `-m "not slow"` for fast iteration. |

## Key design decisions

**Sim mode vs. galois mode (`test_bch.py`)**
The BCH codec has two modes.  Sim mode is the default for all non-`slow` tests:
it counts errors directly against the stored reference rather than running
polynomial arithmetic, which is semantically equivalent for error counts ≤ t.
For n > t, sim mode is *conservative* (always uncorrectable), while the real
galois decoder can sometimes recover beyond t if the error pattern is
favourable.  This is documented in `TestBCHCrossValidation` and means the
simulation gives a **pessimistic (safe) BER estimate**.

**galois deprecation warning suppression (`pyproject.toml`)**
`galois` 0.4.x internally calls its own deprecated `Field()` alias instead of
`GF()` when constructing BCH codes.  The warning originates inside the library
and cannot be fixed in our code.  It is suppressed via:
```toml
filterwarnings = ["ignore::DeprecationWarning:galois"]
```
The `galois` dependency is pinned to `>=0.4.10,<0.5` in
`packages/edac_simulation/pyproject.toml` so that a 0.5 release (which removes
`Field()` entirely and may break the `BCHCodec` galois path) does not slip in
silently.  When upgrading galois, re-check `edac/bch.py` and remove the filter
if the library has been fixed.

**REQ-02 bandwidth failure (`test_scrubbing.py`)**
`TestScrubBandwidth.test_configured_period_exceeds_allocated_bandwidth` is an
intentional FAIL assertion that documents a known design issue: the default
24 h scrub period consumes ~31.8% of the 20 Mbps interface, exceeding the
20% allocation.  The maximum compliant scrub period is ~38.2 h.  This is
tracked in `TASK_TRACKER.md`.

**Analytical reproducibility (`test_analytical.py`)**
`TestDeterministicReproducibility.test_same_seed_same_results` runs two full
2-year mission simulations with the same seed and asserts bit-identical output.
This verifies the reproducibility requirement.  With the current SEU rate
(3.09 × 10⁻⁹ /bit/day), the expected total SEU count over a 2-year simulation
of the 1 MB memory slice is ≈ 18 events, so each trial completes in < 1 s.
