import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, odme
import scipy.sparse as sp
from scipy.optimize import lsq_linear

p="D7_I10_W"; split="validation"
sd = odme.split_arrays(p, split)
A_meas=sd["A_meas"]; c=sd["counts"]; b=sd["b"]
n=A_meas.shape[1]; lam=0.05
A2 = sp.vstack([sp.csr_matrix(A_meas), np.sqrt(lam)*sp.eye(n, format="csr")]).tocsr()
bb = np.concatenate([c, np.sqrt(lam)*b])
t=time.time(); r = lsq_linear(A2, bb, bounds=(0,np.inf), method="trf", lsq_solver="lsmr", max_iter=200); 
print("lsq_linear sparse time", time.time()-t, "status", r.status, r.message if hasattr(r,'message') else '')
print("sum", r.x.sum(), "nnz", (r.x>0).sum(), "S_link", odme.s_link_panel(r.x, sd), "cost", r.cost)
# compare to dual ridge
r0 = c - A_meas@b
M = A_meas@A_meas.T + lam*np.eye(A_meas.shape[0])
u = A_meas.T@np.linalg.solve(M, r0)
f_dual = b+u
print("dual clip S_link", odme.s_link_panel(np.clip(f_dual,0,None), sd), "sum", np.clip(f_dual,0,None).sum())
np.save(r"C:\Users\lenovo\Desktop\kaggle\work\task4\reports\_tmp_fbase_D7_I10_W.npy", r.x)
