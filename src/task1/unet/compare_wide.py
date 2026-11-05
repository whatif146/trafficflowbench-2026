import pandas as pd, numpy as np
cur = pd.read_csv("holdout_physics_u80ens.csv")      # 1.0M, seeds 43+44
new = pd.read_csv("holdout_physics_w4mens.csv")      # 2.23M, seeds 42+43
cols = ["S_state","S_physics","est_total"]
print("variant  1.0M-2seed(sp43,44)           2.23M-2seed(sp42,43)")
for v in ["gbm","unet","w60","w70"]:
    a = cur[cur.variant==v]
    b = new[new.variant==v]
    if len(a)==0: continue
    sa = {c: a[c].mean() for c in cols}
    sb = {c: b[c].mean() for c in cols}
    print(f"{v:5s}  S_state {sa['S_state']:.4f}  S_phys {sa['S_physics']:.4f}  est {sa['est_total']:.4f}"
          f"   |   S_state {sb['S_state']:.4f}  S_phys {sb['S_physics']:.4f}  est {sb['est_total']:.4f}"
          f"   d_est {sb['est_total']-sa['est_total']:+.4f}")
print()
print("per-panel w60 est_total: 1.0M vs 2.23M")
a = cur[cur.variant=="w60"].set_index("panel")["est_total"]
b = new[new.variant=="w60"].set_index("panel")["est_total"]
for p in a.index:
    print(f"  {p:12s} {a[p]:.4f}  {b[p]:.4f}  d={b[p]-a[p]:+.4f}")
print()
print("w60 d_est_total vs GBM:  1.0M %.4f   2.23M %.4f" % (
    (cur[cur.variant=="w60"].est_total - cur[cur.variant=="gbm"].est_total).mean(),
    (new[new.variant=="w60"].est_total - new[new.variant=="gbm"].est_total).mean()))
