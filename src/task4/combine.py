import pandas as pd, os
base=r"C:\Users\lenovo\Desktop\kaggle\work\task4\out"
v=pd.read_csv(os.path.join(base,"path_flow_validation.csv"),dtype={"path_id":str,"panel":str})
p=pd.read_csv(os.path.join(base,"path_flow_private.csv"),dtype={"path_id":str,"panel":str})
t=pd.read_csv(os.path.join(base,"path_flow_train.csv"),dtype={"path_id":str,"panel":str})
print("validation rows",len(v),"private",len(p),"train",len(t))
allp=pd.concat([v,p],ignore_index=True)
allp.to_csv(os.path.join(base,"path_flow_validation_plus_private.csv"),index=False)
print("combined",len(allp),"departure times",allp.departure_time.value_counts().to_dict())
print("finite",allp.path_flow.notna().all(),"min",allp.path_flow.min(),"dups",allp.duplicated(["panel","departure_time","path_id"]).sum())
