import sys, os
sys.path.insert(0, r'C:\Users\lenovo\Desktop\kaggle\work\lib')
import pandas as pd, numpy as np, traffic
REL = traffic.REL
p = "D12_I405_N"
nd = os.path.join(REL,"corridors",p,"network")
paths = pd.read_csv(os.path.join(nd,"path_set.csv"), dtype={"path_id":str})
inc = pd.read_csv(os.path.join(nd,"path_link_incidence.csv"), dtype={"path_id":str,"link_id":str})
print("paths cols", list(paths.columns), paths.shape)
print(paths.head(10).to_string())
print("n origin zones", paths.origin_zone.nunique(), "dest", paths.destination_zone.nunique())
print("origin dtype", paths.origin_zone.dtype)
print(sorted(paths.origin_zone.unique())[:10])
print(sorted(paths.destination_zone.unique())[:10])
print("inc cols", list(inc.columns), inc.shape)
print("n unique links in inc", inc.link_id.nunique())
print("link id samples", sorted(inc.link_id.unique())[:8], "...", sorted(inc.link_id.unique())[-4:])
# paths per OD pair
g = paths.groupby(["origin_zone","destination_zone"]).size()
print("OD pairs", len(g), "paths/OD", g.describe().to_dict())
links = pd.read_csv(os.path.join(nd,"links.csv"), dtype={"link_id":str})
print("links", links.shape, list(links.columns))
print(links.head(4).to_string())
# incidence sizes
sz = inc.groupby("path_id").size()
print("links per path", sz.describe().to_dict())
# count links in counts file
for split in ["train","validation","private"]:
    c = pd.read_csv(os.path.join(REL,"task4",p,split,"synthetic_link_counts.csv"), dtype={"link_id":str})
    pr = pd.read_csv(os.path.join(REL,"task4",p,split,"synthetic_weak_prior.csv"), dtype={"path_id":str})
    tpl = pd.read_csv(os.path.join(REL,"task4",p,split,"sample_submission_path_flow.csv"), dtype={"path_id":str})
    print(split,"counts",c.shape,"prior",pr.shape, list(pr.columns), "tpl",tpl.shape, list(tpl.columns))
    print(" c head:", c.head(3).to_dict("records"))
    print(" pr head:", pr.head(3).to_dict("records"))
    print(" tpl head:", tpl.head(2).to_dict("records"))
    print(" c in inc?", c.link_id.isin(inc.link_id).mean(), "prior ids in paths?", pr.path_id.isin(paths.path_id).mean(), "tpl ids in paths?", tpl.path_id.isin(paths.path_id).mean())
