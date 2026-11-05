import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, odme, cand
for p in ["D12_I405_N","D7_I10_W"]:
    sd=odme.split_arrays(p,"validation"); oi,di=cand.path_meta(p)
    t=time.time(); f,th=cand.margcorr(sd["A_meas"],sd["counts"],sd["b"],oi,di,touch=False,verbose=True)
    print(p,"margcorr %.2fs"%(time.time()-t),"S_link",round(odme.s_link_panel(f,sd),7),"sum",round(f.sum()),"theta_max",round(np.abs(th).max(),2), flush=True)
    t=time.time(); f2,_=cand.margcorr(sd["A_meas"],sd["counts"],sd["b"],oi,di,touch=True)
    print(p,"margcorr+touch %.2fs"%(time.time()-t),"S_link",round(odme.s_link_panel(f2,sd),7),"sum",round(f2.sum()), flush=True)
    t=time.time(); f3,sd3=cand.make(p,"validation","denoise_margcorr")
    print(p,"denoise_margcorr %.2fs"%(time.time()-t),"S_link",round(odme.s_link_panel(f3,sd3),6),"sum",round(f3.sum()), flush=True)
