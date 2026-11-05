import pandas as pd, numpy as np
O=r"C:\Users\lenovo\Desktop\kaggle\work\task4\out"
for split in ["validation","private"]:
    a=pd.read_csv(O+f"\\_check_b_{split}.csv",dtype={"path_id":str,"panel":str}).set_index(["panel","path_id"]).path_flow
    b=pd.read_csv(O+f"\\cand_b_{split}.csv",dtype={"path_id":str,"panel":str}).set_index(["panel","path_id"]).path_flow
    j=a.align(b,join="inner")
    print(split,"n",len(j[0]),"max abs diff",float(np.abs(j[0]-j[1]).max()),"sum a",round(a.sum(),1),"sum b",round(b.sum(),1))
