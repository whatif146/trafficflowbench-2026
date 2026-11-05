import sys, os
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, pandas as pd, odme
pd.set_option("display.width",250); pd.set_option("display.max_columns",60)

# geometry: derive segment index for each mainline link, via chain from origin Z0
def geometry(p):
    paths = odme.load_paths(p); paths["oi"]=paths.origin_zone.str[1:].astype(int); paths["di"]=paths.destination_zone.str[1:].astype(int)
    fz=paths.oi.min(); cand=paths[paths.oi==fz].copy(); cand["n"]=cand.link_seq.str.count(";")+1
    full=cand.sort_values("n").iloc[-1]; chain=full.link_seq.split(";")
    pref={r.di: r.link_seq.split(";") for r in cand.itertuples()}
    seglen={}; prev=0
    for z in sorted(pref): seglen[z-1]=len(pref[z])-prev; prev=len(pref[z])
    seg_of={}; cum=0
    for k in sorted(seglen):
        for l in chain[cum:cum+seglen[k]]: seg_of[l]=k
        cum+=seglen[k]
    return chain, seg_of, paths

rows=[]
for p in odme.PANELS:
    chain, seg_of, paths = geometry(p)
    bs={}
    for split in ["train","validation","private"]:
        sd = odme.split_arrays(p, split)
        # cut profile of prior b: per segment, A b on counted links (=rows of A_meas)
        Ab = sd["A_meas"] @ sd["b"]
        segs = np.array([seg_of[l] for l in sd["measured_link_ids"]])
        prof_b = pd.Series(Ab).groupby(segs).mean()
        C = pd.Series(sd["counts"]).groupby(segs).mean()
        # align
        idx = sorted(set(prof_b.index)&set(C.index))
        x=C.loc[idx].to_numpy(float); y=prof_b.loc[idx].to_numpy(float)
        s=float(x@y/max(y@y,1e-9))  # best scale x~s*y
        corr=float(np.corrcoef(x,y)[0,1])
        rows.append(dict(panel=p, split=split, nseg=len(idx), corr_cut=corr,
                         scale_Ab_over_c = s,
                         relL1_scaled=float(np.abs(x-s*y).sum()/x.sum()),
                         sum_c=x.sum(), sum_Ab=y.sum(),
                         ratio=float(y.sum()/x.sum()),
                         sum_b=sd["b"].sum(), mean_b=sorted(sd["b"])[len(sd["b"])//2]))
        bs[split]=sd["b"]
    b_tr,b_va,b_pr = bs["train"],bs["validation"],bs["private"]
    for name,(u,v) in {"tr-va":(b_tr,b_va),"tr-pr":(b_tr,b_pr),"va-pr":(b_va,b_pr)}.items():
        rows.append(dict(panel=p, split=name, corr_b=float(np.corrcoef(u,v)[0,1]),
                         corr_logb=float(np.corrcoef(np.log(u),np.log(v))[0,1]),
                         median_ratio=float(np.median(u/v))))
df=pd.DataFrame(rows)
print(df[df.split.isin(["train","validation","private"])].to_string(index=False))
print("\n--- cross-split prior correlations ---")
print(df[df.split.isin(["tr-va","tr-pr","va-pr"])][["panel","split","corr_b","corr_logb","median_ratio"]].to_string(index=False))
df.to_csv(os.path.join(odme.REPORTS,"prior_cut_diag.csv"), index=False)
