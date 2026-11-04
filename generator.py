"""SN91 cascade generator — single native module (s1 of c313k105c303epi_fast).

One generator, no packed sources: the engine, its vendored chronoforge family
library and the partner family program are ordinary top-level code of this
module.  Output rows are byte-identical to the c313 exec-wrapper for every seed
and ``n``: the engine produces each row; with probability ``_Q`` (decided by a
per-row SeedSequence draw) the next partner-family row is stacked under it as a
second channel.

Name layout (to keep every former module's globals separate):
  * engine names are unchanged, except its public ``Generator`` which is
    ``_EngineGenerator`` here;
  * chronoforge names are unchanged unless they would collide, in which case
    they are prefixed ``_<cf_module>_``;
  * partner-family names are prefixed ``_pt_``; its config lives in
    ``config.json["partner"]``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator

# ═══════════════ engine: runtime setup and imports ═══════════════
# Cascade V145 — native support integrity over the V144 persistence candidate.
#
# Experimental training-data generator. No measured forecasting gain is implied.


# H393: production-sandbox thread control before numerical imports.
# These must be set before the first numerical import to take effect.
import os as _h393_os
for _h393_var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS"):
    _h393_os.environ[_h393_var] = "1"
from threadpoolctl import threadpool_limits as _h393_threadpool_limits
_H393_THREAD_LIMITER = _h393_threadpool_limits(limits=1)
del _h393_var

import json
import os
from functools import lru_cache, partial
from queue import Full, Queue
from threading import Event, Lock, Thread, local
from typing import Any

from numba import njit
from scipy.signal import lfilter


# --- graft: chronoforge, inlined (logic unchanged) ---


# ═══════════════ chronoforge: cf_rng ═══════════════
# Deterministic RNG stream derivation.
#
# Every random number in the corpus comes from ``np.random.SeedSequence`` entropy
# built from four 32-bit words::
#
#     (seed_hi, seed_lo, batch_index, stage_id)
#
# ``batch_index`` is the unit of reproducibility: batch sizes follow a schedule
# that depends on the batch index alone (never on ``n_series``), so the corpus is
# prefix-stable by construction.  ``stage_id`` isolates the pipeline stages from
# each other, which is what makes weight sweeps *paired*: changing family ``f``'s
# weight leaves every other family's per-row parameter draws byte-identical.
#
# Nothing here reads the clock, the environment, ``os.urandom``, or a global RNG.


MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# ── stage identifiers ──────────────────────────────────────────────────────
STAGE_ASSIGN = 0        # length + cadence + family assignment
STAGE_CALENDAR = 1      # calendar bundle (start day-of-week / day-of-year)
STAGE_FAMILY_PARAM = 1000   # + family index: dense per-row parameter draws
STAGE_FAMILY_BULK = 5000    # + family index: group-sized innovation draws
STAGE_OBSERVE = 9000    # observation layer
STAGE_SCALE = 9001      # scale / offset
STAGE_AGGREGATE = 9002  # structured hierarchical aggregation
STAGE_SANITISE = 9003   # fallback regeneration + degeneracy repair
STAGE_COUNT = 9004      # count prior: nonnegative floor + level quantisation


def seed_words(seed: int) -> tuple[int, int]:
    """Split an arbitrary Python int seed into two 32-bit words."""
    s = int(seed) & MASK64
    return (s >> 32) & MASK32, s & MASK32


def stream(seed_hi: int, seed_lo: int, batch: int, stage: int) -> np.random.Generator:
    """Independent PCG64 stream for ``(seed, batch, stage)``."""
    ss = np.random.SeedSequence(
        [int(seed_hi) & MASK32, int(seed_lo) & MASK32,
         int(batch) & MASK32, int(stage) & MASK32]
    )
    return np.random.Generator(np.random.PCG64(ss))


def _self_test() -> None:
    """Fail loudly at import if stream derivation is not reproducible.

    We deliberately do not hard-code NumPy's internal bit-generator state (that
    would pin us to one NumPy build for no benefit).  What we *do* assert is the
    three properties the corpus relies on: same coordinates -> same draws,
    different coordinates -> different draws, and the derivation is insensitive
    to how the Python int seed is spelled.
    """
    a = stream(1234, 5678, 7, 1003).random(8)
    b = stream(1234, 5678, 7, 1003).random(8)
    if not np.array_equal(a, b):
        raise RuntimeError("chronoforge: RNG stream derivation is not reproducible")
    for coords in ((1234, 5678, 8, 1003), (1234, 5678, 7, 1004),
                   (1235, 5678, 7, 1003), (1234, 5679, 7, 1003)):
        if np.array_equal(a, stream(*coords).random(8)):
            raise RuntimeError("chronoforge: RNG streams collide across coordinates")
    hi, lo = seed_words(-1)
    if (hi, lo) != (MASK32, MASK32):
        raise RuntimeError("chronoforge: seed word split is wrong")
    hi, lo = seed_words((1 << 63) + 12345)
    if (hi << 32 | lo) != ((1 << 63) + 12345):
        raise RuntimeError("chronoforge: seed word split loses bits")
    # zero seed must still produce a usable stream
    if not np.isfinite(stream(0, 0, 0, 0).random(4)).all():
        raise RuntimeError("chronoforge: zero seed produced non-finite draws")


_self_test()
