"""Execution modes shared by the controller and Python workers."""
import os
import re
from contracts import require

LEGACY_MODE = "legacy-capped"
PRODUCTION_MODE = "production-default"
MODES = (LEGACY_MODE, PRODUCTION_MODE)
LEGACY_THREADS = {
    "VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "POLARS_MAX_THREADS": "1",
}
# Remove scheduling overrides as well as the historical caps before imports.
THREAD_VARIABLES = tuple(sorted(set(LEGACY_THREADS) | {
    "OPENBLAS_DEFAULT_NUM_THREADS", "GOTO_NUM_THREADS", "BLIS_NUM_THREADS",
    "OMP_THREAD_LIMIT", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
    "OMP_MAX_ACTIVE_LEVELS", "MKL_DYNAMIC", "MKL_DOMAIN_NUM_THREADS",
    "NUMEXPR_MAX_THREADS", "RAYON_NUM_THREADS", "POLARS_ASYNC_THREAD_COUNT",
    "VECLIB_NUM_THREADS",
}))


def policy(mode):
    require(mode in MODES, "Unknown execution mode: " + str(mode))
    return dict(schema_version=1, mode=mode,
                thread_environment=LEGACY_THREADS.copy() if mode == LEGACY_MODE else {},
                cleared_environment=list(THREAD_VARIABLES),
                cleared_prefixes=["OMP_", "MKL_", "OPENBLAS_", "VECLIB_", "NUMEXPR_"],
                duckdb_threads=1 if mode == LEGACY_MODE else "engine-default",
                minimum_process_rounds=5 if mode == PRODUCTION_MODE else 1,
                order="rotating-engines-v1", active_threads_measured=False)


def worker_environment(mode, inherited=None):
    result = dict(os.environ if inherited is None else inherited)
    for key in list(result):
        if key in THREAD_VARIABLES or re.match(r"^(OMP|MKL|OPENBLAS|VECLIB|NUMEXPR)_", key):
            result.pop(key, None)
    result.update(policy(mode)["thread_environment"])
    result["SWIFTSCI_BENCHMARK_MODE"] = mode
    return result


def engine_order(engines, batch):
    offset = batch % len(engines)
    return engines[offset:] + engines[:offset]


def numerical_backend():
    import numpy as np
    dependencies = np.show_config(mode="dicts").get("Build Dependencies", {})
    return {name: {k: v for k, v in value.items() if k in ("name", "version", "openblas configuration")}
            for name, value in dependencies.items() if name in ("blas", "lapack")}


def runtime_record(pools=None):
    mode = os.environ.get("SWIFTSCI_BENCHMARK_MODE", LEGACY_MODE)
    return dict(mode=mode,
                thread_environment={k: v for k, v in os.environ.items() if k in THREAD_VARIABLES or re.match(r"^(OMP|MKL|OPENBLAS|VECLIB|NUMEXPR)_", k)},
                configured_pool_sizes=pools or {}, active_threads_measured=False)


def validate_runtime(record, mode, engine=None):
    require(isinstance(record, dict), "Missing worker execution record")
    require(record.get("mode") == mode, "Worker execution mode mismatch")
    require(record.get("thread_environment") == policy(mode)["thread_environment"],
            "Worker thread environment differs from execution policy")
    require(record.get("active_threads_measured") is False, "Pool size is not active thread use")
    pools = record.get("configured_pool_sizes")
    require(isinstance(pools, dict) and all(type(v) is int and v > 0 for v in pools.values()),
            "Invalid configured pool sizes")
    if engine in ("polars", "duckdb"):
        require(engine in pools, "Missing engine pool size")
        if mode == LEGACY_MODE:
            require(pools[engine] == 1, "Legacy engine pool must be one")
