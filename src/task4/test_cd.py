import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, odme
from scipy.optimize import nnls

for p in ["D12_I405_S","D7_I10_W"]:
    for split in ["validation"]:
        sd = odme.split_arrays(p, split)
        A=sd["A_meas"]; c=sd["counts"]; b=sd["b"]
        lam=0.05
        print("="*70); print(p, split, "A", A.shape, "sum b", b.sum(), "sum c", c.sum())
        # reference: exact nnls on small panel only
        if A.shape[1] <= 1500:
            t=time.time()
            A2=np.vstack([A, np.sqrt(lam)*np.eye(A.shape[1])]); bb=np.concatenate([c, np.sqrt(lam)*b])
            f_ref,_=nnls(A2,bb); print("ref nnls time %.1fs"%(time.time()-t))
            def obj(f): return np.sum((A@f-c)**2)+lam*np.sum((f-b)**2)
            print("ref obj", obj(f_ref), "S_link", odme.s_link_panel(f_ref, sd), "sum", f_ref.sum(), "nnz",(f_ref>0).sum())
            f_cd = odme.nnls_cd(A,c,b,lam,verbose=False)
            print("cd  obj", obj(f_cd), "S_link", odme.s_link_panel(f_cd, sd), "sum", f_cd.sum(), "nnz",(f_cd>0).sum())
            print("max |f_cd-f_ref|", np.abs(f_cd-f_ref).max(), " rel obj diff", (obj(f_cd)-obj(f_ref))/obj(f_ref))
        else:
            def obj(f): return np.sum((A@f-c)**2)+lam*np.sum((f-b)**2)
            t=time.time(); f_cd = odme.nnls_cd(A,c,b,lam,verbose=True); print("cd time %.1fs"%(time.time()-t))
            print("cd obj", obj(f_cd), "S_link", odme.s_link_panel(f_cd, sd), "sum", f_cd.sum(), "nnz",(f_cd>0).sum())
            f_d = odme.nnls_dual(A,c,b,lam); print("dual obj", obj(f_d), "S_link", odme.s_link_panel(f_d, sd), "sum", f_d.sum())
