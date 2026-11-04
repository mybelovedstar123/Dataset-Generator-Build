# Dataset-Generator-Build

A deterministic, multi-family **synthetic time-series generator** for pre-training
forecasting models. It is a single Python module (`generator.py`) driven by one
configuration file (`config.json`). The module implements the `DataGenerator`
interface of the Cascade (Bittensor SN91) evaluation harness and emits an unbounded
stream of fixed-length series: univariate rows, and two-channel rows where a second,
coupled channel is stacked under the first.

> **Status: experimental.** The code itself says so: *"Experimental training-data
> generator. No measured forecasting gain is implied."* Nothing in this repository
> claims a benchmark result.

---

## At a glance

| | |
|---|---|
| Language | Python 3 (single module, ~19k lines) |
| Interface | `cascade.interface.DataGenerator` → `Generator(config_dir, *, seed).generate(n_series)` |
| Output | `numpy.float64` arrays: shape `(4096,)` (univariate) or `(2, L)` (two channels) |
| Generative families | ~90 in the engine (74 active at the shipped weights), 48 in the partner program, 6 in the active teach stream |
| Determinism | Byte-identical output for a given `seed`, independent of thread count or scheduling |
| Hot paths | Numba-compiled kernels (`nogil=True`), NumPy/SciPy vector code |
| I/O | Reads `config.json` only; no network, no clock, no global RNG |

## Repository layout

| File | Purpose |
|---|---|
| `generator.py` | The whole generator: engine, vendored *chronoforge* family library, partner family program, teach stream and the public `Generator` |
| `config.json` | Family weights, length band, observation-layer rates, augmentation, partner-program settings, runtime controls |
| `requirements.txt` | Hash-pinned runtime dependencies |
| `SCAN.json` | Static scan report for the module (`high: 0, warn: 0`) |

## Technology stack

| Component | Version | Used for |
|---|---|---|
| **NumPy** | 2.4.6 | All array work; `SeedSequence`/`PCG64` for every random stream |
| **Numba** | 0.66.0 | ~60 JIT-compiled recurrences (AR/ARMA filters, RK4 integrators, Hawkes, renewal processes). Compiled with `nogil=True` so worker threads run in parallel |
| **SciPy** | 1.17.1 | `signal.lfilter` (linear recurrences), `ndimage` min/max filters, `special.gammaincinv` |
| **threadpoolctl** | via scikit-learn | Pins BLAS/OpenMP pools to one thread at import time |
| **scikit-learn** | 1.9.0 | Pinned, but not imported by `generator.py`; it supplies `threadpoolctl` |
| **cascade** | harness | Supplies the `DataGenerator` base class. Not on this repo's dependency list; provided by the evaluation sandbox |

Standard library: `threading`, `queue`, `functools`, `json`, `pathlib`.
`multiprocessing` is deliberately avoided because the target sandbox blocks it.

## Architecture

The public `Generator` is a stack of wrappers. Each layer consumes the stream of
the one below it and changes only the rows its own random stream selects:

```
Generator  ("k86-fork/b2")            fork: adds a second channel to a band of non-quiet univariate rows
  └─ Generator  ("w9qedrki")          stacks a partner / teach-stream channel under ~29.6% of rows
       ├─ _EngineGenerator            swaps 14.779% of rows for a "novel" family stream (I141 selector)
       │    └─ _BaseGenerator         the engine: family dispatch → build → observe → sanitise → post-process
       │         └─ chronoforge       vendored family library (32 families, observation layer, sanitiser)
       ├─ _pt_Generator               partner family program (names prefixed _pt_)
       └─ _TeachStream                batched "teach" families (IXP traffic, web counters, season onsets, ...)
```

The module is organised by banner comments (`# ═══ engine ═══`, `# ═══ chronoforge: cf_rng ═══`, ...),
so each subsystem can be found by searching for its banner.

## Operating principles

### 1. Determinism by stream isolation
Every random number comes from `np.random.SeedSequence` entropy keyed by the
position of the work, never by time or thread:

```
(base_seed, chunk_index, family, slot, tag)      # engine streams
(seed_hi, seed_lo, batch_index, stage_id)        # chronoforge streams
```

Consequences, which the design relies on:

- **Prefix stability.** Batch sizes depend only on the batch index (the chunk ramp is
  256 → 1024 → 2048 rows), so the first *k* series are the same whatever `n_series` is.
- **Paired experiments.** Each family draws its per-row parameters as a dense batch
  from its own stream, then indexes the result by row position. Changing one family's
  weight changes only *which* rows belong to that family; every other row stays
  byte-identical. Weight sweeps and A/B runs are therefore paired comparisons.
- **Minimal-mass dispatch.** Rows are first assigned against a fixed reference CDF,
  then moved to the active weights by a maximal coupling. A +2 pp weight edit
  reassigns about that much mass, instead of the ~18% a plain inverse-CDF re-draw
  would shuffle.

### 2. The per-chunk pipeline (engine)
For each chunk of rows, `_BaseGenerator.generate` runs:

1. **Assign:** draw lengths (`min_length`…`max_length`, fixed at 4096 in the shipped
   config) and a family for each row from the (optionally curriculum-scheduled) weights.
2. **Build:** each family group builds its rows, in parallel across groups.
3. **Observe:** the shared observation layer (below) is applied, gated per row by
   capability flags that each family returns (`integer`, `bounded`, `positive`, ...).
4. **Sanitise:** finiteness, degeneracy, cold-start and magnitude guards. Degenerate
   rows are regenerated from a fallback, never emitted flat.
5. **Post-process:** non-negative integer prior, tail rebasing, zero inflation,
   TSMixup and prefix padding.

### 3. Concurrency without nondeterminism
- A **producer thread** fills a bounded `Queue` (`prefetch_depth`), so generation
  overlaps with the trainer consuming rows.
- Within a chunk, **family groups are built on worker threads**
  (`_PRODUCER_WORKERS = 3`). Groups write only their own slots, so the output does
  not depend on worker count or completion order.
- The chronoforge `BatchProducer` uses a **strict-order reorder buffer**: workers may
  finish out of order, but batches are emitted by index, with a bounded in-flight
  window (`max_inflight_batches`) to keep memory flat.
- BLAS/OpenMP/Numba thread pools are pinned to 1 at import (`OMP_NUM_THREADS`, etc.,
  plus `threadpoolctl`). This avoids oversubscription, because parallelism comes
  from the GIL-free Numba kernels instead.

### 4. Throughput techniques
- **Hoisted draw tables.** `Generator.choice` rebuilds its CDF on every call
  (~24 µs vs ~2 µs for the draw itself). The tables are precomputed, and the code
  reproduces both NumPy's values and its stream position exactly.
- **Batched seeding.** Each row needs about seven isolated streams. Building
  `default_rng(SeedSequence(...))` each time costs ~40 µs, about 15% of runtime.
  The module instead replicates NumPy's SeedSequence → PCG64 seeding in vectorised
  form, producing identical 128-bit states, and falls back to NumPy for any unusual
  key layout.
- **Thread-local stream pools.** These reseed one bit generator in place per slot,
  rather than allocating a new one.

## Functionality

### Engine families
The engine dispatches over about 90 named families (`family_weights` in `config.json`),
grouped roughly as:

| Group | Examples |
|---|---|
| Classical stochastic | `ar2`, `arma_sarima`, `threshold_ar`, `integrated`, `long_memory`, `ou_stochastic_vol`, `garch_leverage`, `vol_regime_switch`, `spectral_gp` |
| Trend / seasonality | `trend_seasonal_ar`, `multiplicative`, `cycle_profile`, `envelope_mod`, `tidal_constituents`, `smooth_drift_extrapolable` |
| Regimes & steps | `step_level`, `regime_shift`, `regime_dwell_*`, `level_ladder_staircase`, `held_rate`, `settle_hold`, `conditional_stability`, `cs_*` |
| Counts & intermittency | `seasonal_counts`, `rate_counts`, `bounded_counts`, `overdispersed_counts`, `intermittent`, `hawkes_marked`, `pulse_outlier` |
| Physical systems | `rk4_flows`, `physical_sensors`, `flow_recession`, `gauge_rain`, `coastal_residual`, `met_*` |
| Energy & markets | `energy_load_price_solar`, `grid_flow`, `dispatch_price`, `spiky_price`, `price_shock`, `econ_release_staircase` |
| Operations / web | `ops_*`, `web_traffic_plateau`, `weekly_web_counts`, `heavy_traffic_counts`, `capacity_counts`, `storm_outage_counts` |
| Health | `epi_renewal`, `epi_decay`, `reported_epi_counts`, `admin_reporting_counts`, `clinical_bounded_vitals`, `physio_quasiperiodic` |

A second **"novel" stream** is an independently seeded `_BaseGenerator` restricted to
ten families (`epi_season_decay`, `intraday_spot_price`, `road_commute_counts`, ...).
A per-row selector draws from it with probability 0.14779; that value is pinned and
validated at construction.

### chronoforge library (vendored)
32 families in six blocks, each with a `params` and a `build` function, registered in
`cf_registry`:

| Block | Families |
|---|---|
| A: meteorology | `met_thermal`, `met_pressure_smooth`, `met_bounded_atom`, `met_wind_speed`, `wet_dry_intermittent` |
| B: operations | `ops_diurnal_traffic`, `ops_saturating_feedback`, `ops_counter_reset`, `ops_latency_queue`, `ops_deploy_transient`, `ops_rate_plateau` |
| C: health | `epi_renewal`, `admin_reporting_counts`, `physio_quasiperiodic`, `clinical_bounded_vitals` |
| D: regimes | five `regime_dwell_*` variants, `level_ladder_staircase`, `smooth_drift_extrapolable` |
| E: stochastic | `arma_sarima`, `garch_leverage`, `hawkes_marked`, `chaotic_delay`, `spectral_kernel_zoo`, `changepoint_composite` |
| F: domain | `energy_load_price_solar`, `transport_flow`, `retail_promo_intermittent`, `econ_release_staircase` |

It also provides calendars and cadences (`cf_calendars`, `cf_cadence`), spectral
helpers, and a config schema validator (`cf_config_schema`) that raises `ConfigError`.

### Observation layers
After generation, two layers reproduce how real feeds are recorded.

**Engine: `_measurement_artifacts`.** Its rates come from `config.json["observation"]`.
It applies censoring, quantisation (uniform and non-uniform), regular and irregular
holds, shock rows, kernel and periodic spikes, amplitude trends and drift, time,
gamma and global warps, and duty cycling. Integer, bounded and non-negative
supports are preserved where a family declares them.

**chronoforge: `apply_observation`.** This one serves the library families. It runs
eight stages in a fixed order, each gated per row by the capability flags that row's
family returned:

1. block aggregation (rolling-window statistics of a finer signal)
2. quantisation to a round human tick
3. one-sided censoring at a round value
4. staleness (bursty last-observation-carried-forward)
5. missing-as-zero
6. isolated outliers
7. instrument drift with abrupt recalibration
8. scale-guarded rounding: only when the series spans enough ticks, so continuous
   families are not collapsed into a 3–5-level staircase

Three things are left out on purpose, as the code documents: no global time
reversal, no tail-concentrated regime breaks, and no blind rounding.

### Second channels
- **Partner family program** (`_pt_*`): an independent generator with its own length
  distribution, seasonal-period table and 48 configured domain families
  (`integer_staircase`, `periodic_profile`, `rki_epidemic_weekly_sums`,
  `electricity_spot_price`, `hydrology_flow_stage`, ...).
- **Teach stream**: batches of 256 rows from six active families: `ixp_traffic`,
  `signed_flow`, `web_counter`, `season_onset`, `sparse_counts` and
  `ed_visit_share`. An earlier ten-family set is still in the module, but it has
  been superseded. The teach stream supplies 43.75% of partner rows.
- **Stacking:** a per-row coin (`_Q = 0.296`) decides whether a partner row is stacked
  under the engine row, aligned on the right-hand edge.
- **Coupling:** the second channel gets `0.2 × sd(Δy)` of the first channel's
  standardised increments added, so the channels are correlated rather than
  independent. Families whose integer floors would be blurred by this skip it.
- **Fork band:** a final layer gives a further 32% of the remaining univariate,
  non-quiet rows a second channel from a differently weighted teach stream.
- Every stacked row is kept only if both channels, and their difference, have
  non-trivial variance.

## Configuration

`config.json` is read from `config_dir` by the engine. The partner program and the
chronoforge defaults also read the copy next to `generator.py`, so keep the two
identical. Main keys:

| Key | Meaning |
|---|---|
| `min_length`, `max_length` | Row length band (the shipped config fixes both at 4096) |
| `family_weights` | Engine dispatch weights (normalised; must be finite, non-negative, not all zero) |
| `curriculum` | Optional schedule that interpolates weights and parameters over training progress (disabled) |
| `observation` | Rates for the observation-layer stages |
| `augment` | `tsmixup` (0.18), `pad_prefix` (0.05) |
| `tail_rebase_rate`, `nonneg_integer_frac` | Post-processing rates |
| `chronoforge` | Library schema version, batch schedule, thread limits, its own weights and cadences |
| `renewal_persistence`, `support_contract`, `temporal_integrity` | Engine guards and repair options |
| `isolated_scalar_fraction` | Share of rows from the novel stream (pinned to 0.14779) |
| `partner` | Partner-program row length, padding, rounding, family weights and augmentation |

Invalid values fail fast with `ValueError` / `ConfigError` at construction.

## Usage

```bash
pip install numpy==2.4.6 scipy==1.17.1 numba==0.66.0 scikit-learn==1.9.0
```

```python
from generator import Generator      # requires the harness's `cascade` package

gen = Generator(config_dir=".", seed=1234)
for x in gen.generate(1_000):
    # x.shape == (4096,)  or  (2, L)
    ...
```

`generate` is a generator. Closing it early (`close()`, or leaving a `for` loop) stops
the producer threads cleanly.

## Known limitations

- **`requirements.txt` is not installable as written with pip.** It uses `--hash`
  pins, which switch pip into hash-checking mode. That mode also requires every
  transitive dependency (`llvmlite`, `joblib`, `threadpoolctl`, ...) to be pinned
  and hashed. Until those are added, install the four packages by version, as shown
  above.
- **`threadpoolctl` is imported unconditionally** but only arrives transitively
  through scikit-learn.
- **`cascade.interface` is external.** The module cannot be imported outside the
  Cascade harness without a stub that provides `DataGenerator`.
- **Single large module.** Earlier layers are kept and wrapped (`Generator` is
  redefined twice; `_EngineGenerator` once) so that every published row stream stays
  reproducible. That makes the file long, and its layering is easiest to follow from
  the bottom up.
