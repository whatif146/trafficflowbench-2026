import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, odme
from scipy.optimize import nnls, lsq_linear

def ridge_dual(A_meas, c, b, lam):
    r = c - A_meas @ b
    M = A_meas @ A_meas.T + lam*np.eye(A_meas.shape[0])
    x = np.linalg.solve(M, r)
    u = A_meas.T @ x
    return b + u

p="D7_I10_W"; split="validation"
sd = odme.split_arrays(p, split)
A=sd["A"]; m=sd["measured"]; A_meas=A[m]; c=sd["counts"]; b=sd["b"]
print("shape", A_meas.shape, "sum c", c.sum(), "sum b", b.sum())
for lam in [0.05]:
    t=time.time(); f=ridge_dual(A_meas,c,b,lam); print("dual time", time.time()-t)
    print("neg count", (f<0).sum(), "min", f.min(), "sum", f.sum())
    print("S_link dual raw", odme.s_link_panel(f, sd), "clipped", odme.s_link_panel(np.clip(f,0,None), sd))
