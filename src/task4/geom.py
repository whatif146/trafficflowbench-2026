"""Task-4 corridor geometry: zone order, segments, cut flows, ramp mapping.

Hypothesis: counted mainline link flows are cut flows C_k through the corridor;
C_k - C_{k-1} = on-ramp inflow - off-ramp outflow at the zone boundary. If so the
released ramp_states give the origin and destination marginals of the true OD.
"""
import sys, os, glob
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, pandas as pd
import odme

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)

panel = sys.argv[1] if len(sys.argv) > 1 else "D12_I405_N"
split = sys.argv[2] if len(sys.argv) > 2 else "validation"

paths = odme.load_paths(panel)
inc = odme.load_incidence(panel)
nd = odme.net_dir(panel)
ramp_map = pd.read_csv(os.path.join(nd, "ramp_attachment_map.csv"), dtype=str)

zone_num = lambda z: int(str(z)[1:])
paths["oi"] = paths.origin_zone.map(zone_num)
paths["di"] = paths.destination_zone.map(zone_num)
first_zone, last_zone = int(paths.oi.min()), int(max(paths.di.max(), paths.oi.max()))
print("panel", panel, split, "paths", len(paths), "zones", first_zone, "..", last_zone)

cand = paths[paths.oi == first_zone].copy()
cand["n"] = cand.link_seq.str.count(";") + 1
full = cand.sort_values("n").iloc[-1]
chain = full.link_seq.split(";")
pref = {r.di: r.link_seq.split(";") for r in cand.itertuples()}
seglen, prev_n = {}, 0
for z in sorted(pref):
    seglen[z - 1] = len(pref[z]) - prev_n
    prev_n = len(pref[z])
seg_links, seg_of, cum = {}, {}, 0
for k in sorted(seglen):
    seg_links[k] = chain[cum:cum + seglen[k]]
    for l in seg_links[k]:
        seg_of[l] = k
    cum += seglen[k]
print("chain", len(chain), "segments", len(seglen), "lengths", list(seglen.values()))

pref_start = {z: sum(seglen[j] for j in range(z)) for z in range(0, last_zone + 1)}
bad = sum(chain[pref_start[r.oi]:pref_start[r.di]] != r.link_seq.split(";") for r in paths.itertuples())
print("paths not equal to chain slice:", bad, "/", len(paths))

sets = inc.groupby("path_id").link_id.apply(set)
ron_z, roff_z = [], []
for p in sets.index:
    for l in sets[p]:
        if l.startswith("RON"):
            ron_z.append(int(l.split("-Z")[1]))
        elif l.startswith("ROFF"):
            roff_z.append(int(l.split("-Z")[1]))
ron_z = sorted(set(ron_z)); roff_z = sorted(set(roff_z))
print("RON zones", ron_z, "\nROFF zones", roff_z)

cdf = odme.load_counts(panel, split)
cnt = cdf.set_index("link_id")["count"]
cut, cutn = {}, {}
for k, ls in seg_links.items():
    vals = [cnt[l] for l in ls if l in cnt.index]
    if vals:
        cut[k] = float(np.mean(vals)); cutn[k] = float(np.std(vals))
print("\nsegments with counts:", len(cut), "of", len(seg_links), " max within-seg std", max(cutn.values()))
sc = sorted(cut)
C = np.array([cut[k] for k in sc])
print("cut profile C:", np.round(C, 0).astype(int).tolist())


def ramp_pm(panel, split):
    base = os.path.join(odme.REL, "corridors", panel, split, "ramp_states")
    fs = sorted(glob.glob(os.path.join(base, "**", "*.parquet"), recursive=True))
    parts = []
    for f in fs:
        d = pd.read_parquet(f, columns=["timestamp", "ramp_link_id", "flow_vph", "is_score_eligible"])
        ts = pd.to_datetime(d.timestamp, utc=True)
        d = d[(ts.dt.hour >= 15) & (ts.dt.hour < 19)]
        parts.append(d[["ramp_link_id", "flow_vph", "is_score_eligible"]])
    a = pd.concat(parts, ignore_index=True)
    n_all = len(a)
    a = a[a.is_score_eligible == 1]
    return a.groupby("ramp_link_id").flow_vph.mean(), n_all, len(a)


rf, n_all, n_ok = ramp_pm(panel, split)
print("\nramp obs rows PM:", n_all, "eligible:", n_ok, "ramps:", len(rf), "map rows:", len(ramp_map))
rm = ramp_map.drop_duplicates("ramp_link_id").set_index("ramp_link_id")
rm = rm[rm.index.isin(rf.index)].copy()
rm["flow"] = rf.reindex(rm.index).to_numpy()
rm["seg"] = rm.nearest_mainline_link_id.map(seg_of)
print("ramps mapped to segments:", int(rm.seg.notna().sum()), "/", len(rm))
print(rm[["ramp_type", "nearest_mainline_link_id", "seg", "flow"]].round(2).to_string())

net = rm.groupby(["seg", "ramp_type"]).flow.sum().unstack(fill_value=0.0)
net["net"] = net.get("OR", 0) - net.get("FR", 0)
print("\nper-segment ramp net flow:")
print(net.round(1).to_string())

dC = np.diff(C)  # dC[i] = C[sc[i+1]]-C[sc[i]]
for off in (-1, 0, 1):
    vals, ref = [], []
    for i in range(1, len(sc)):
        seg = sc[i] + off
        vals.append(net.net.loc[seg] if seg in net.index else 0.0)
        ref.append(dC[i - 1])
    vals = np.array(vals, float); ref = np.array(ref, float)
    print(f"offset {off:+d}: corr(dC, rampnet)={np.corrcoef(ref, vals)[0,1]:.3f}  "
          f"sum|dC|={np.abs(ref).sum():.1f} sum|ramp|={np.abs(vals).sum():.1f} "
          f"L1={np.abs(ref-vals).sum():.1f}")
print("C[0]=", C[0], " sum|dC|=", np.abs(dC).sum(),
      " sum OR=", net.get("OR", pd.Series(dtype=float)).sum(),
      " sum FR=", net.get("FR", pd.Series(dtype=float)).sum())
