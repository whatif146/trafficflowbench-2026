"""Environment self-check for the TrafficFlowBench workspace.

Run:  python work/setup/verify_env.py
"""
from __future__ import annotations

import glob
import os
import sys

ROOT = r"C:\Users\lenovo\Desktop\kaggle"
REL = os.path.join(ROOT, "kaggle_public")
CACHE = os.path.join(ROOT, "work", "cache")

ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f"[ok]   {name}: {msg}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def versions():
    import lightgbm
    import numpy
    import pandas
    import pyarrow
    import psutil
    import scipy
    import sklearn
    return (f"numpy {numpy.__version__}, pandas {pandas.__version__}, "
            f"pyarrow {pyarrow.__version__}, scipy {scipy.__version__}, "
            f"sklearn {sklearn.__version__}, lightgbm {lightgbm.__version__}, "
            f"psutil {psutil.__version__}, python {sys.version.split()[0]}")


def parquet_read():
    import pandas as pd
    f = sorted(glob.glob(os.path.join(REL, "corridors", "*", "train", "mainline_states", "*", "*.parquet")))[0]
    d = pd.read_parquet(f, columns=["timestamp", "station_id", "link_id", "speed_kmh"])
    return f"{len(d):,} rows from {os.path.basename(f)}"


def release():
    files = glob.glob(os.path.join(REL, "**", "*.parquet"), recursive=True)
    return f"{len(files):,} parquet files under {REL}"


def cache():
    missing = []
    for panel in os.listdir(CACHE):
        pdir = os.path.join(CACHE, panel)
        if not os.path.isdir(pdir) or panel == "task2_windows.parquet":
            continue
        have = set(os.listdir(pdir))
        need = {"train.npz", "train_masked.npz", "validation_masked.npz", "private_masked.npz",
                "targets_train.npz", "targets_validation.npz", "targets_private.npz",
                "ramp_train.npz", "ramp_validation.npz", "ramp_private.npz"}
        lack = need - have
        if lack:
            missing.append(f"{panel}:{sorted(lack)}")
    if missing:
        raise RuntimeError("; ".join(missing))
    return f"{len(os.listdir(CACHE))} panel cache dirs complete"


def kaggle_auth():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    limits = api.competition_get_submission_limits("2026-ieee-big-data-traffic-flow-bench")
    return f"authenticated, {getattr(limits, 'num_allowed_now', '?')} submissions allowed now"


def memory():
    import psutil
    vm = psutil.virtual_memory()
    return f"{vm.available/2**30:.2f} GB free of {vm.total/2**30:.1f} GB"


def main():
    check("package versions", versions)
    check("release data", release)
    check("parquet read", parquet_read)
    check("panel caches", cache)
    check("kaggle api", kaggle_auth)
    check("memory", memory)
    print("\nENV OK" if ok else "\nENV HAS PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
