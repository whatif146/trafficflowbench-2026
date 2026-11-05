import sys, os
sys.path.insert(0, r'C:\Users\lenovo\Desktop\kaggle\work\lib')
import pandas as pd, numpy as np, traffic
REL = traffic.REL
p="D12_I405_N"
nd = os.path.join(REL,"corridors",p,"network")
paths = pd.read_csv(os.path.join(nd,"path_set.csv"), dtype={"path_id":str})
inc = pd.read_csv(os.path.join(nd,"path_link_incidence.csv"), dtype={"path_id":str,"link_id":str})
c = pd.read_csv(os.path.join(REL,"task4",p,"validation","synthetic_link_counts.csv"), dtype={"link_id":str})
links = pd.read_csv(os.path.join(nd,"links.csv"), dtype={"link_id":str})
print("link prefixes:", pd.Series([l.split("-")[0] for l in links.link_id]).value_counts().to_dict())
print("inc link prefixes:", pd.Series([l.split("-")[0] for l in inc.link_id]).value_counts().to_dict())
print("counted link prefixes:", pd.Series([l.split("-")[0] for l in c.link_id]).value_counts().to_dict())
print("\ncounted links:", sorted(c.link_id.tolist()))
print("\nnonzero col sums for RON links:")
RON=[l for l in sorted(inc.link_id.unique()) if l.startswith("RON")]
print(RON)
for l in RON[:5]:
    print(l, (inc.link_id==l).sum(), inc[inc.link_id==l].path_id.head(3).tolist())
# example path
for pid in ["P000_001","P000_002","P020_005"]:
    r=paths[paths.path_id==pid]
    print(pid, r.origin_zone.iloc[0], r.destination_zone.iloc[0], r.link_seq.iloc[0] if "link_seq" in paths.columns else "")
    print("   inc:", inc[inc.path_id==pid].link_id.tolist())
# which links are NOT in inc
print("\nlinks not in inc:", sorted(set(links.link_id)-set(inc.link_id)))
print("num mainline in links", links.link_id.str.startswith("L").sum())
# Check OD structure
print("\nunique origins", sorted(paths.origin_zone.unique()))
print("unique dests", sorted(paths.destination_zone.unique()))
# OD pair count check
print("n OD", paths[["origin_zone","destination_zone"]].drop_duplicates().shape)
# do all pairs i<j exist?
oz = [int(z[1:]) for z in paths.origin_zone]
dz = [int(z[1:]) for z in paths.destination_zone]
print("any i>=j?", sum(np.array(oz)>=np.array(dz)))
print("zone range", min(oz), max(oz), min(dz), max(dz))
