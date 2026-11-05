"""Onset set-size distribution per threshold (for the LB operating-point probe)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import traffic

wi = traffic.load_window_index()
cond = dict(zip(wi.window_id.astype(str), wi.condition))

print("== predicted onset links/window (last step only) ==")
files = {
    0.30: "queue_v5_{split}.csv",
    0.40: "queue_v5blend_on040_{split}.csv",
    0.45: "queue_v5blend_on045_{split}.csv",
    0.50: "queue_v5blend_on050_{split}.csv",
    0.55: "queue_v5blend_on055_{split}.csv",
}
for split in ("validation", "private"):
    print(f"-- {split}")
    for th, pat in files.items():
        d = pd.read_csv(os.path.join(LH.OUT, pat.format(split=split)),
                        dtype={"window_id": str, "link_id": str})
        d["condition"] = d.window_id.map(cond)
        on = d[d.condition == "queue_onset"]
        g = on.groupby("window_id").queue_pred.sum()
        print(f"  th {th:.2f}: mean {g.mean():.2f} links/window | min {int(g.min())} "
              f"max {int(g.max())} | zero-pred windows {int((g == 0).sum())}/{len(g)}")

# truth set size on the 80 released train windows (3-step smoothed plain truth)
tot = []
for p in LH.PANELS:
    z = np.load(os.path.join(LH.CACHE, f"rel_{p}.npz"), allow_pickle=False)
    sel = (z["split"] == "train") & (z["condition"] == "queue_onset")
    tr = z["truth"][sel]
    tot.extend(tr[:, LH.HORIZON - 1].sum(axis=1).tolist())
    z.close()
tot = np.asarray(tot)
print(f"\nreleased-train onset truth (T+30, smoothed): mean {tot.mean():.2f} links/window "
      f"| min {tot.min()} max {tot.max()} | n {len(tot)}")
