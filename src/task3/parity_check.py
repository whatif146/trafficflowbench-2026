"""Build a small staging release root (3 days per regime, one panel) and check the
numpy replica in local_eval.py against the official src/task3/score_task3.py.

The official evaluator needs the whole release layout; a staging root lets it run
on a 9-day slice in seconds.  `--flux fallback` in local_eval reproduces the
official topology fallback (no organizer boundary flux), so the two must agree
exactly on S_FD / S_LWR / S_physics for every regime.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
REL = os.path.join(ROOT, "kaggle_public")
REPO = os.path.join(ROOT, "repo", "trafficflowbench-public-main")
WORK3 = os.path.join(ROOT, "work", "task3")
STAGE = os.path.join(WORK3, "staging")
sys.path.insert(0, os.path.join(ROOT, "libs"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(WORK3))
PANEL = "D7_I10_E"
PY = sys.executable


def build_stage(days_per_regime: int = 3):
    panel_dir = os.path.join(REL, "corridors", PANEL)
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(os.path.join(STAGE, "corridors", PANEL, "network"))
    os.makedirs(os.path.join(STAGE, "corridors", PANEL, "train", "mainline_states"))
    os.makedirs(os.path.join(STAGE, "corridors", PANEL, "train", "ramp_states"))
    os.makedirs(os.path.join(STAGE, "task1", PANEL, "train"))
    for f in os.listdir(os.path.join(panel_dir, "network")):
        if f.endswith(".csv") or f.endswith(".json"):
            shutil.copy2(os.path.join(panel_dir, "network", f),
                         os.path.join(STAGE, "corridors", PANEL, "network", f))
    # 3 days per regime, taken from the published masked views so each regime has days
    chosen = []
    for r in ("R1", "R2", "R3"):
        mdir = os.path.join(panel_dir, "train", "mainline_states_masked", f"mask_regime={r}")
        files = sorted(os.listdir(mdir))[:days_per_regime]
        chosen += [(r, f) for f in files]
    print("chosen:", chosen)
    plain_dir = os.path.join(panel_dir, "train", "mainline_states")
    plain_by_name = {f: os.path.join(dp, f) for dp, _, fs in os.walk(plain_dir) for f in fs}
    ramp_dir = os.path.join(panel_dir, "train", "ramp_states")
    ramp_by_name = {f: os.path.join(dp, f) for dp, _, fs in os.walk(ramp_dir) for f in fs}
    dates = []
    for r, f in chosen:
        if f not in plain_by_name:
            raise FileNotFoundError(f"plain partition missing for {f}")
        shutil.copy2(plain_by_name[f], os.path.join(STAGE, "corridors", PANEL, "train",
                                                    "mainline_states", f))
        rf = f.replace("mainline", "ramp")
        if rf in ramp_by_name:
            shutil.copy2(ramp_by_name[rf], os.path.join(STAGE, "corridors", PANEL, "train",
                                                        "ramp_states", rf))
        else:
            print(f"WARNING: ramp partition missing for {rf}")
        base = f.split(".")[0]
        dates.append("-".join(base.split("_")[-3:]))
    return set(dates), [f for _, f in chosen]


def build_template_and_submission(dates: set[str]):
    # copy only the template rows of the staged days (read in chunks: the train
    # template is ~170 MB)
    src = os.path.join(REL, "task1", PANEL, "train", "sample_submission_state.csv")
    keep = []
    for chunk in pd.read_csv(src, dtype=str, chunksize=1_500_000):
        day = chunk.timestamp.str[:10]
        keep.append(chunk[day.isin(dates)])
    tpl = pd.concat(keep, ignore_index=True)
    tpl.to_csv(os.path.join(STAGE, "task1", PANEL, "train", "sample_submission_state.csv"),
               index=False)
    print(f"template rows staged: {len(tpl):,}")
    # truth-field submission for those rows, from the cached train arrays
    import traffic
    z = np.load(traffic.split_cache_path(PANEL, "train"))
    plain = {k: z[k] for k in z.files}
    z.close()
    times = pd.to_datetime(plain["times"].astype("int64"), unit="s", utc=True)
    t_index = pd.Series(np.arange(len(times)), index=times)
    stations = pd.Index(plain["stations"].astype(str))
    links = pd.Index(plain["links"].astype(str))
    ti = pd.to_datetime(tpl.timestamp, utc=True).map(t_index).to_numpy(np.int64)
    si = stations.get_indexer(tpl.station_id.astype(str))
    sub = tpl[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
    sub["speed_kmh"] = plain["speed"][ti, si]
    sub["flow_vph"] = plain["flow"][ti, si]
    out = os.path.join(WORK3, "staging", "truth_submission.csv")
    sub.to_csv(out, index=False)
    print(f"truth submission: {len(sub):,} rows -> {out}")
    return out


def run_official(sub_csv: str, tag: str):
    out = os.path.join(WORK3, "reports", f"official_{tag}.csv")
    cmd = [PY, os.path.join(REPO, "src", "task3", "score_task3.py"),
           "--release-root", STAGE, "--split", "train", "--panel", PANEL,
           "--state-submission", sub_csv, "--output", out]
    print("RUN", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-4000:])
    print(r.stderr[-4000:])
    if r.returncode != 0:
        raise SystemExit(f"official scorer failed: {r.returncode}")
    return pd.read_csv(out)


if __name__ == "__main__":
    dates, files = build_stage(3)
    print("dates:", sorted(dates))
    sub = build_template_and_submission(dates)
    off = run_official(sub, "stage")
    # run the local replica on the same days, same fallback flux
    import local_eval
    out = subprocess.run([PY, os.path.join(WORK3, "local_eval.py"),
                          "--candidate", "observed", "--flux", "fallback", "--panel", PANEL,
                          "--dates", ",".join(sorted(dates)),
                          "--output", os.path.join(WORK3, "reports", "local_stage.csv")],
                         capture_output=True, text=True)
    print(out.stdout[-3000:])
    print(out.stderr[-3000:])
    loc = pd.read_csv(os.path.join(WORK3, "reports", "local_stage.csv"))
    loc = loc[loc.level == "panel_regime"][["regime", "S_FD", "S_LWR", "S_physics", "n_rows", "n_trans"]]
    offr = off[off.level == "panel_regime"][["regime", "S_FD", "S_LWR", "S_physics", "n_rows"]] if "S_FD" in off.columns else off
    print("\n=== official staged ===")
    print(offr.to_string(index=False))
    print("\n=== local replica ===")
    print(loc.to_string(index=False))
