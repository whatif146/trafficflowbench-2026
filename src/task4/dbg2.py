import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, odme, cand
p="D12_I405_N"
sd=odme.split_arrays(p,"validation")
t=time.time(); f=cand.solve_ridge(sd["A_meas"],sd["counts"],sd["b"],0.05); print("ridge %.1fs"%(time.time()-t), "S_link",odme.s_link_panel(f,sd), "sum", round(f.sum()), flush=True)
for m in ["scale:0.5","pool","shrink:0.5","denoise"]:
    t=time.time(); f,sd2=cand.make(p,"validation",m); print(m,"%.1fs"%(time.time()-t),"S_link",round(odme.s_link_panel(f,sd2),5),"sum",round(f.sum()), flush=True)
t=time.time(); f,sd2=cand.make(p,"validation","margcorr"); print("margcorr %.1fs"%(time.time()-t),"S_link",round(odme.s_link_panel(f,sd2),5),"sum",round(f.sum()), flush=True)
t=time.time(); f,sd2=cand.make(p,"validation","denoise_margcorr"); print("denoise_margcorr %.1fs"%(time.time()-t),"S_link",round(odme.s_link_panel(f,sd2),5),"sum",round(f.sum()), flush=True)
