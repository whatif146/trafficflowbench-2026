"""Discover the Task-4 corridor geometry: zone order, segments, cut flows, ramp mapping.

Key hypothesis to test: link counts are cut flows through the corridor; consecutive
cut differences equal on-ramp minus off-ramp flow at each zone. If so, ramp_states
release the origin and destination marginals of the true OD matrix.
"""
import sys, os, glob
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, pandas as pd
import odme

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)

panel = sys.argv[1] if len(sys.argv) > 1 else "D12_I405_N"
split = sys.argv[2] if len(sys.argv) > 2 else "validation"

paths = odme.load_paths(panel)
inc = odme.load_incidence(panel)
nd = odme.net_dir(panel)
ramp_map = pd.read_csv(os.path.join(nd, "ramp_attachment_map.csv"), dtype=str)
topo = pd.read_csv(os.path.join(nd, "lwr_mainline_topology.csv"), dtype=str)

# ---- mainline link order from topology order_index ----
col = "order_index" if "order_index" in topo.columns else None
t = topo[["mainline_link_id", col]].copy(); t[col] = pd.to_numeric(t[col]); t = t.sort_values(col)
order = t.mainline_link_id.tolist()
pos = {l: i for i, l in enumerate(order)}
print("mainline links", len(order), "topo cols", list(topo.columns))

# ---- path link sets ----
sets = inc.groupby("path_id").link_id.apply(lambda s: set(s))
paths = paths.set_index("path_id")
orig = paths.origin_zone.to_dict(); dest = paths.destination_zone.to_dict()
zone_num = lambda z: int(z[1:])
zones = sorted({zone_num(z) for z in orig.values()} | {zone_num(z) for z in dest.values()})
print("zones", zones[0], "..", zones[-1], "count", len(zones))

mainl = [l for l in inc.link_id.unique() if l.startswith("L")]
conn = [l for l in inc.link_id.unique() if not l.startswith("L")]
print("n mainline in inc", len(mainl), "connectors", len(conn), sorted(conn)[:6])

# ---- derive mainline sequence from origin-min paths (nested prefixes) ----
o0 = min(orig.values(), key=zone_num)
p0 = [p for p in paths.index if orig[p] == o0]
p0.sort(key=lambda p: zone_num(dest[p]))
seq = []
for p in p0:
    s = sets[p]
    m = [l for l in s if l.startswith("L")]
    assert set(m) <= set(order), "unknown mainline link"
    m_sorted = sorted(m, key=lambda l: pos[l])
    if len(m_sorted) > len(seq):
        seq = m_sorted
print("mainline seq from origin", o0, "len", len(seq))
print("first 6", seq[:6], "last 3", seq[-3:])

# ---- segment of each mainline link: segment k is between zone k and k+1 ----
# from nested prefixes: segment k = seq links added by path to dest k+1
seg_of = {}
prefixes = {}
for p in p0:
    m = sorted([l for l in sets[p] if l.startswith("L")], key=lambda l: pos[l])
    prefixes[zone_num(dest[p])] = m
seg_links = {}
prev = []
for k in sorted(prefixes):
    m = prefixes[k]
    new = [l for l in m if l not in prev]
    seg_links[k - 1] = new
    prev = m
print("segments:", {k: len(v) for k, v in seg_links.items()})
for k in sorted(seg_links)[:5]:
    print("  seg", k, seg_links[k])

# ---- verify every path = union of segments i..j-1 + connectors ----
bad = 0
for p in paths.index:
    i, j = zone_num(orig[p]), zone_num(dest[p])
    m = sorted([l for l in sets[p] if l.startswith("L")], key=lambda l: pos[l])
    exp = [l for k in range(i, j) for l in seg_links[k]]
    exp = sorted(exp, key=lambda l: pos[l])
    if m != exp:
        bad += 1
        if bad < 4:
            print("MISMATCH", p, i, j, "got", len(m), "exp", len(exp), m[:5], exp[:5])
print("paths mismatching segment-union:", bad, "/", len(paths))

# connectors per path
ron = {p: [l for l in sets[p] if l.startswith("RON")] for p in paths.index}
roff = {p: [l for l in sets[p] if l.startswith("ROFF")] for p in paths.index}
print("paths with RON", sum(len(v) > 0 for v in ron.values()), "with ROFF", sum(len(v) > 0 for v in roff.values()))
ron_z = sorted({int(v[0].split("-Z")[1]) for v in ron.values() if v})
roff_z = sorted({int(v[0].split("-Z")[1]) for v in roff.values() if v})
print("RON zones", ron_z)
print("ROFF zones", roff_z)
# check RON consistency: RON present iff origin in ron_z
print("RON present exactly for origin in list:",
      all((int(orig[p]) in ron_z) == (len(ron[p]) > 0) for p in paths.index))
print("ROFF present exactly for dest in list:",
      all((int(dest[p]) in roff_z) == (len(roff[p]) > 0) for p in paths.index))

# ---- counted links: cut flow profile ----
cdf = odme.load_counts(panel, split)
cnt = cdf.set_index("link_id")["count"]
cut = {}
for k, ls in seg_links.items():
    vals = [cnt[l] for l in ls if l in cnt.index]
    if vals:
        cut[k] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
print("\nsegments with counts:", len(cut), "of", len(seg_links))
prof = np.array([cut[k][0] for k in sorted(cut)])
print("cut profile head", np.round(prof[:6], 1), "tail", np.round(prof[-4:], 1))
print("max within-segment std", max(v[1] for v in cut.values()))
seg_c = sorted(cut)
dc = np.diff(np.array([cut[k][0] for k in seg_c]))
print("delta cut per counted segment:")
print(pd.DataFrame({"seg": seg_c, "C": [cut[k][0] for k in seg_c]}).to_string(index=False))

# ---- ramp flows PM mean ----
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
    a = a[a.is_score_eligible == 1]
    return a.groupby("ramp_link_id").flow_vph.mean()

rf = ramp_pm(panel, split)
rm = ramp_map.drop_duplicates("ramp_link_id").set_index("ramp_link_id")
rf = rf[rf.index.isin(rm.index)]
rm = rm.loc[rf.index]
rm["flow"] = rf
rm["seg"] = rm.nearest_mainline_link_id.map(lambda l: next((k for k, ls in seg_links.items() if l in ls), None))
print("\nramps mapped to segments:", rm.seg.notna().sum(), "/", len(rm))
print(rm[["ramp_link_id", "ramp_type", "nearest_mainline_link_id", "seg", "flow"]].to_string(index=False))
net = rm.groupby(["seg", "ramp_type"]).flow.sum().unstack(fill_value=0.0)
net["increments"] = net.get("OR", 0) - net.get("FR", 0)
print("\nper-segment ramp net flow:")
print(net.round(2).to_string())
print("\ncount increments vs ramp net: correlation =",
      float(np.corrcoef(dc, net.increments.reindex(seg_c).fillna(0).to_numpy())[0, 1]))
print("sum OR", net.get("OR", pd.Series(0, index=net.index)).sum(),
      "sum FR", net.get("FR", pd.Series(0, index=net.index)).sum())
