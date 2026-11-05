import sys, os, pandas as pd, numpy as np
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import odme
O=os.path.join(odme.WORK,"out")
key=pd.read_csv(os.path.join(odme.REL,"submission_key.csv"),dtype=str)
kr=key[key.task=="odme"][["panel","departure_time","path_id"]]
d=pd.read_csv(os.path.join(O,"final_recommended_validation_plus_private.csv"),dtype={"panel":str,"path_id":str,"departure_time":str})
m=d.merge(kr,on=["panel","departure_time","path_id"],how="outer",indicator=True)
f=d.path_flow.to_numpy(float)
print("final_recommended rows",len(d),"missing",int((m._merge=='right_only').sum()),"extra",int((m._merge=='left_only').sum()),
      "finite",np.isfinite(f).all(),"min",float(f.min()),"dups",int(d.duplicated(["panel","departure_time","path_id"]).sum()),
      "total",round(float(f.sum()),1))
