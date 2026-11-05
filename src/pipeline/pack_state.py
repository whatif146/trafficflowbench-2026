"""Pack Task 1 state CSVs into dense per-panel arrays for fast merging.

Key-based (not order-based) packing: each row is located by
(timestamp -> time index, station_id -> station index) from the split cache, so
files with missing or reordered rows are handled safely; a coverage mask records
which target cells were actually provided.

Input CSV(s): panel,timestamp,station_id,link_id,mask_regime,speed_kmh,flow_vph
covering the template rows of validation (2031-03) and/or private (2031-04).

Output: work/submissions/<out-name>/<panel>_<split>.npz  speed, flow, mask
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
import traffic

BASE_OUT = os.path.join(traffic.WORK, "submissions")
CHUNK = 500_000


def main(paths, out_name="dense", panels=None):
    panels = panels or traffic.PANELS
    out_dir = os.path.join(BASE_OUT, out_name)
    os.makedirs(out_dir, exist_ok=True)
    meta = {}
    for panel in panels:
        for split in ("validation", "private"):
            masked = traffic.load_mainline(panel, split)["masked"]
            tg = traffic.load_targets(panel, split)
            shape = (len(masked["times"]), len(masked["stations"]))
            iso = traffic.epoch_to_iso(masked["times"])
            meta[(panel, split)] = {
                "shape": shape,
                "times": {s: i for i, s in enumerate(iso)},
                "stations": {s: i for i, s in enumerate(masked["stations"].astype(str))},
                "speed": np.zeros(shape, dtype=np.float32),
                "flow": np.zeros(shape, dtype=np.float32),
                "mask": np.zeros(shape, dtype=bool),
                "n_targets": len(tg["t_idx"]),
                "seen": 0,
            }
    for path in paths:
        print("reading", path, flush=True)
        for chunk in pd.read_csv(path, chunksize=CHUNK,
                                 dtype={"panel": str, "timestamp": str, "station_id": str,
                                        "link_id": str, "mask_regime": str}):
            month = chunk.timestamp.str[5:7].to_numpy()
            split_arr = np.where(month == "03", "validation", "private")
            panel_arr = chunk.panel.to_numpy()
            for (panel, sp), m in meta.items():
                sel = np.flatnonzero((panel_arr == panel) & (split_arr == sp))
                if not len(sel):
                    continue
                ts = chunk.timestamp.to_numpy()[sel]
                st = chunk.station_id.to_numpy()[sel]
                ti = np.fromiter((m["times"].get(t, -1) for t in ts), np.int64, len(sel))
                si = np.fromiter((m["stations"].get(s, -1) for s in st), np.int64, len(sel))
                good = (ti >= 0) & (si >= 0)
                if not good.all():
                    print(f"  WARN {panel} {sp}: {int((~good).sum())} rows not in cache grid", flush=True)
                ti, si = ti[good], si[good]
                m["speed"][ti, si] = chunk.speed_kmh.to_numpy(np.float32)[sel][good]
                m["flow"][ti, si] = chunk.flow_vph.to_numpy(np.float32)[sel][good]
                m["mask"][ti, si] = True
                m["seen"] += len(sel)
    for (panel, sp), m in meta.items():
        provided = int(m["mask"].sum())
        status = "ok" if provided == m["n_targets"] else f"INCOMPLETE ({provided}/{m['n_targets']})"
        print(f"{panel:12s} {sp:10s} {provided:8d} cells {status}", flush=True)
        np.savez_compressed(os.path.join(out_dir, f"{panel}_{sp}.npz"),
                            speed=m["speed"], flow=m["flow"], mask=m["mask"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out-name", default="dense")
    args = ap.parse_args()
    main(args.paths, args.out_name)
