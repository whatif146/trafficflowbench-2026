import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, odme, cand
p="D12_I405_N"
t=time.time(); sd=odme.split_arrays(p,"validation"); print("load %.1fs"%(time.time()-t), flush=True)
t=time.time(); f=odme.nnls_dual(sd["A_meas"], sd["counts"], sd["b"], 0.05); print("dual %.1fs"%(time.time()-t), "S_link",odme.s_link_panel(f,sd), flush=True)
t=time.time(); f,res=cand.solve_ridge(sd["A_meas"],sd["counts"],sd["b"],0.05); print("ridge %.1fs"%(time.time()-t), "S_link",odme.s_link_panel(f,sd), "nit",res.nit, flush=True)
for m in ["scale:0.5","pool","denoise"]:
    t=time.time(); f,sd2=cand.make(p,"validation",m); print(m,"%.1fs"%(time.time()-t),"S_link",odme.s_link_panel(f,sd2),"sum",f.sum(), flush=True)
t=time.time(); f,sd2=cand.make(p,"validation","margcorr"); print("margcorr %.1fs"%(time.time()-t),"S_link",odme.s_link_panel(f,sd2),"sum",f.sum(), flush=True)
