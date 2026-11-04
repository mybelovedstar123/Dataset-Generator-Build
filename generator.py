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
