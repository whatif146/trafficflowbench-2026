import sys, os
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import pandas as pd, numpy as np, odme
O=os.path.join(odme.WORK,"out")
key=pd.read_csv(os.path.join(odme.REL,"submission_key.csv"),dtype=str)
kr=key[key.task=="odme"][["panel","departure_time","path_id"]]
for name in ["cand_x_ent_blend35","cand_x_ent_blend70"]:
    d=pd.read_csv(os.path.join(O,name+"_validation_plus_private.csv"),dtype={"panel":str,"path_id":str,"departure_time":str})
    m=d.merge(kr,on=["panel","departure_time","path_id"],how="outer",indicator=True)
    f=d.path_flow.to_numpy(float)
    print(name,"rows",len(d),"missing",int((m._merge=='right_only').sum()),"extra",int((m._merge=='left_only').sum()),
          "finite",np.isfinite(f).all(),"min",round(float(f.min()),4),"dups",int(d.duplicated(["panel","departure_time","path_id"]).sum()))
