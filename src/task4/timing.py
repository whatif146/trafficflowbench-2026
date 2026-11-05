import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, odme
from scipy.optimize import nnls, lsq_linear
p="D7_I10_W"; split="validation"
sd = odme.split_arrays(p, split)
A=sd["A"]; m=sd["measured"]; A_meas=A[m]; c=sd["counts"]; b=sd["b"]
print("shape", A_meas.shape, "sum c", c.sum(), "sum b", b.sum())
lam=0.05
A2=np.vstack([A_meas, np.sqrt(lam)*np.eye(A_meas.shape[1])]); bb=np.concatenate([c, np.sqrt(lam)*b])
t=time.time(); f,_=nnls(A2,bb); print("nnls time", time.time()-t, "sum f", f.sum(), "nnz", (f>0).sum())
t=time.time(); r=lsq_linear(A2, bb, bounds=(0,np.inf)); print("lsq_linear time", time.time()-t, "sum", r.x.sum(), "nnz",(r.x>0).sum())
print("S_link nnls", odme.s_link_panel(f, sd), "lsq", odme.s_link_panel(r.x, sd))
