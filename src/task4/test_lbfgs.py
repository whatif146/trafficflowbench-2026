import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, odme
from scipy.optimize import nnls, minimize

def solve_lbfgs(A, c, b, lam, x0=None):
    Afun = lambda f: 0.5*np.sum((A@f-c)**2) + 0.5*lam*np.sum((f-b)**2)
    gfun = lambda f: A.T@(A@f-c) + lam*(f-b)
    x0 = np.clip(b,1e-9,None) if x0 is None else x0
    r = minimize(Afun, x0, jac=gfun, method="L-BFGS-B", bounds=[(0,None)]*len(b),
                 options=dict(maxiter=2000, maxfun=5000, ftol=1e-14, gtol=1e-10, maxcor=30))
    return r.x, r

for p in ["D12_I405_S","D7_I10_W"]:
    sd = odme.split_arrays(p, "validation")
    A=sd["A_meas"]; c=sd["counts"]; b=sd["b"]; lam=0.05
    obj=lambda f: np.sum((A@f-c)**2)+lam*np.sum((f-b)**2)
    print("="*60, p, A.shape)
    t=time.time(); f,r = solve_lbfgs(A,c,b,lam); print("lbfgs %.1fs iters %d status %s"%(time.time()-t, r.nit, r.message))
    print(" obj", obj(f), "S_link", odme.s_link_panel(f, sd), "sum", f.sum(), "nnz",(f>0).sum())
    if A.shape[1]<=1500:
        A2=np.vstack([A, np.sqrt(lam)*np.eye(A.shape[1])]); bb=np.concatenate([c, np.sqrt(lam)*b])
        fr,_=nnls(A2,bb); print(" ref obj", obj(fr), "S_link", odme.s_link_panel(fr,sd), "maxdiff", np.abs(f-fr).max())
    fd=odme.nnls_dual(A,c,b,lam); print(" dual obj", obj(fd), "S_link", odme.s_link_panel(fd,sd), "sum", fd.sum())
