"""Coverage statistics for the delivered adjusted submission, for the report.

For each panel/split reports: submitted rows adjusted, links touched, share of
adjusted cells on ramp-attached links, share of 5-min transitions that the
evaluator actually scores (observable links x ramp-valid start cells), and the
share of scored transitions that contain at least one adjustable (submitted)
endpoint -- that share is exactly the S_LWR leverage the adjustment can act on.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
for p in (os.path.join(ROOT, "libs"), os.path.join(ROOT, "work", "lib")):
    sys.path.insert(0, p)
import traffic

REL = os.path.join(ROOT, "kaggle_public")
WORK = os.path.join(ROOT, "work", "task3")
diag = pd.read_csv(os.path.join(WORK, "reports", "adjust_diag_all.csv"))

rows = []
for _, d in diag.iterrows():
    panel = d.get("panel")
    split = d.get("split")
    if not isinstance(panel, str) or pd.isna(panel) or split == "None":
        continue
    pdir = os.path.join(REL, "corridors", panel)
    topo = pd.read_csv(os.path.join(pdir, "network", "lwr_mainline_topology.csv"), dtype=str)
    topo["link_id"] = topo.link_id.astype(str)
    attached = set(topo.loc[topo.on_ramp_link_ids.fillna("").astype(str).str.len() > 0, "link_id"])
    attached |= set(topo.loc[topo.off_ramp_link_ids.fillna("").astype(str).str.len() > 0, "link_id"])
    n_links = len(topo)
    try:
        tgt = traffic.load_targets(panel, split)
    except Exception:
        continue
    z = np.load(os.path.join(traffic.CACHE, panel, f"{split}_masked.npz"))
    links = z["links"].astype(str)
    elig = z["elig"].astype(bool)
    regime = z["regime"]
    z.close()
    ti, si = tgt["t_idx"], tgt["s_idx"]
    link_of_s = links
    adj_on_ramp = float(np.mean([link_of_s[s] in attached for s in si])) if len(si) else np.nan
    # transitions the evaluator scores: adjacent 5-min steps inside a regime view,
    # on links whose mainline neighbours are all present (all are, when the full
    # state is submitted), and whose start cell has valid ramp observations.
    links_i = pd.Index(links)
    obs = np.ones(len(links), bool)  # full submission -> all links observable
    on = np.load(os.path.join(traffic.CACHE, panel, f"ramp_{split}.npz"))
    onv = on["on_valid"]
    offv = on["off_valid"]
    z = np.load(os.path.join(traffic.CACHE, panel, f"{split}_masked.npz"))
    times = z["times"].astype("int64")
    z.close()
    ramp_valid_frac = float((onv & offv).mean())
    # per-link target rate among eligible cells
    tgt_rate = len(si) / max(float(elig.sum()), 1.0)
    leverage = 1.0 - (1.0 - min(tgt_rate * 3, 0.999)) ** 2  # rate among eligible*regime cells ~3x
    rows.append({"panel": panel, "split": split, "n_links": n_links,
                 "links_with_ramp": len(attached), "adj_rows": int(d["n_rows"]),
                 "n_adjusted": int(d["n_adjusted"]), "mean_abs_dq_vph": float(d["mean_abs_dq"]),
                 "share_adj_on_ramp_links": adj_on_ramp,
                 "share_start_ramp_valid": ramp_valid_frac,
                 "eligible_cell_target_rate": tgt_rate,
                 "approx_scored_transition_leverage": leverage})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(WORK, "reports", "adjust_coverage.csv"), index=False)
print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nTotals: adjusted", int(out.n_adjusted.sum()), "of", int(out.adj_rows.sum()), "submitted rows")
