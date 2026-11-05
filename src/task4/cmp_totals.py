import pandas as pd
R=r"C:\Users\lenovo\Desktop\kaggle\work\task4\reports"
e=pd.read_csv(R+r"\entropy_diagnostics.csv")
b=pd.read_csv(R+r"\candidate_diagnostics.csv")
ev=e[(e.tag=="ent_own")&(e.split=="validation")].set_index("panel")
bv=b[(b.method=="baseline")&(b.split=="validation")].set_index("panel")
out=pd.DataFrame({"ent_own_sum":ev.sum_f,"baseline_sum":bv.sum_f,"ratio":ev.sum_f/bv.sum_f,
                  "ent_own_slink":ev.s_link,"base_slink":bv.s_link})
print(out.round(4).to_string())
print("\nmean ratio", round(float((ev.sum_f/bv.sum_f).mean()),4))
