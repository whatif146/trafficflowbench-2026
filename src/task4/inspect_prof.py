import sys, os
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, pandas as pd, odme

def geom(panel):
    paths = odme.load_paths(panel); paths["oi"]=paths.origin_zone.str[1:].astype(int); paths["di"]=paths.destination_zone.str[1:].astype(int)
    fz=paths.oi.min(); cand=paths[paths.oi==fz].copy(); cand["n"]=cand.link_seq.str.count(";")+1
    full=cand.sort_values("n").iloc[-1]; chain=full.link_seq.split(";")
    pref={r.di: r.link_seq.split(";") for r in cand.itertuples()}
    seglen={}; prev=0
    for z in sorted(pref): seglen[z-1]=len(pref[z])-prev; prev=len(pref[z])
    seg_of={}; cum=0
    for k in sorted(seglen):
        for l in chain[cum:cum+seglen[k]]: seg_of[l]=k
        cum+=seglen[k]
    return paths, seg_of, seglen

for panel in ["D7_I210_E","D12_I405_N"]:
    paths, seg_of, seglen = geom(panel)
    sd = odme.split_arrays(panel,"validation")
    segs=np.array([seg_of[l] for l in sd["measured_link_ids"]])
    prof_c=pd.Series(sd["counts"]).groupby(segs).mean()
    prof_b=pd.Series(sd["A_meas"]@sd["b"]).groupby(segs).mean()
    idx=sorted(set(prof_c.index)&set(prof_b.index))
    C=prof_c.loc[idx].to_numpy(float); B=prof_b.loc[idx].to_numpy(float)
    s=(C@B)/(B@B)
    print("="*80); print(panel,"nseg",len(idx),"C mean",C.mean().round(1),"C std",C.std().round(1),"B mean",B.mean().round(1),"B std",B.std().round(1),"scale",round(s,3))
    print("C:", np.round(C[:20],0).astype(int))
    print("sB:", np.round(s*B[:20],0).astype(int))
    print("C range", C.min().round(0), C.max().round(0), " sB range", (s*B).min().round(0),(s*B).max().round(0))
    # regress C on sB including intercept
    A1=np.c_[s*B, np.ones(len(B))]
    coef=np.linalg.lstsq(A1,C,rcond=None)[0]
    pred=A1@coef
    print("regression C = a*sB + b :", np.round(coef,3), "R2", round(1-np.sum((C-pred)**2)/np.sum((C-C.mean())**2),3), "relL1", round(np.abs(C-pred).sum()/C.sum(),3))
    # increments
    dC=np.diff(C); dB=np.diff(B)
    A2=np.c_[dB, np.ones(len(dB))]
    c2=np.linalg.lstsq(A2,dC,rcond=None)[0]; pred2=A2@c2
    print("dC = a*dB + b:", np.round(c2,3), "corr", round(np.corrcoef(dC,dB)[0,1],3), "R2", round(1-np.sum((dC-pred2)**2)/np.sum((dC-dC.mean())**2),3))
