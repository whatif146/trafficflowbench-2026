import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, odme, cand
for p in ["D12_I405_N","D7_I210_E"]:
    sd=odme.split_arrays(p,"validation")
    print("="*80); print(p,"n=",len(sd["b"]),"sum b",round(sd["b"].sum(),0),"sum c",round(sd["counts"].sum(),0),"rank A",np.linalg.matrix_rank(sd["A_meas"]))
    for m in ["baseline","scale:0.5","scale:1.5","pool","shrink:0.5","denoise","margcorr","denoise_margcorr","margcorr_pool"]:
        t=time.time()
        try:
            f,sd2=cand.make(p,"validation",m)
            sl=odme.s_link_panel(f,sd2); dhat=float(np.abs(f-sd2["b"]).sum())
            # cut-profile rel L1
            print(f"  {m:18s} sum={f.sum():9.1f} S_link={sl:.5f} Dhat={dhat:9.1f} Dhat/sumb={dhat/sd2['b'].sum():.3f} nnz={int((f>0).sum()):5d} t={time.time()-t:.1f}s")
        except Exception as e:
            print("  ",m,"FAILED",type(e).__name__,e)
