import sys
sys.path.insert(0, r'C:\Users\lenovo\Desktop\kaggle\work\lib')
import pandas as pd, numpy as np, traffic, os
print("REL", traffic.REL)
print("PANELS", traffic.PANELS)
p="D12_I405_N"
for split in ["train","validation","private"]:
    t4=traffic.load_task4(p,split)
    pr,c,tpl,paths,inc = t4["prior"],t4["counts"],t4["template"],t4["paths"],t4["incidence"]
    print("="*60)
    print(split, "prior",pr.shape, list(pr.columns), "counts",c.shape, "tpl",tpl.shape, list(tpl.columns))
    print(pr.head(3).to_string())
    print(c.head(3).to_string())
    print(tpl.head(3).to_string())
print("paths", paths.shape, list(paths.columns))
print(paths.head(5).to_string())
print("inc", inc.shape, list(inc.columns))
print("n unique path ids in paths", paths.path_id.nunique(), "in inc", inc.path_id.nunique(), "in tpl", tpl.path_id.nunique())
print("zones:", paths.origin_zone.nunique(), paths.destination_zone.nunique())
