import sys, os
sys.path.insert(0, r'C:\Users\lenovo\Desktop\kaggle\work\lib')
import pandas as pd, numpy as np, traffic
REL = traffic.REL

def load_net(p):
    nd = os.path.join(REL,"corridors",p,"network")
    paths = pd.read_csv(os.path.join(nd,"path_set.csv"), dtype={"path_id":str})
    inc = pd.read_csv(os.path.join(nd,"path_link_incidence.csv"), dtype={"path_id":str,"link_id":str})
    return paths, inc

def A_of(paths, inc):
    pid = paths.path_id.tolist(); pidx={x:i for i,x in enumerate(pid)}
    links = sorted(inc.link_id.unique().tolist()); lidx={x:i for i,x in enumerate(links)}
    A = np.zeros((len(links), len(pid)))
    A[inc.link_id.map(lidx).to_numpy(), inc.path_id.map(pidx).to_numpy()] = 1.0
    return A, pid, links

rows=[]
for p in traffic.PANELS:
    paths, inc = load_net(p)
    A, pid, links = A_of(paths, inc)
    for split in ["train","validation","private"]:
        c = pd.read_csv(os.path.join(REL,"task4",p,split,"synthetic_link_counts.csv"), dtype={"link_id":str})
        pr = pd.read_csv(os.path.join(REL,"task4",p,split,"synthetic_weak_prior.csv"), dtype={"path_id":str})
        b = pr.set_index("path_id").reindex(pid).path_flow.to_numpy(float)
        # measured links = counted links
        cmask = np.isin(links, c.link_id.values)
        cc = c.set_index("link_id").reindex(np.array(links)[cmask])["count"].to_numpy(float)
        Ab = A[cmask] @ b
        denom = max(cc.sum(),1e-9)
        rows.append(dict(panel=p, split=split, npaths=len(pid), nlinks=len(links), ncounted=int(cmask.sum()),
            rank=np.linalg.matrix_rank(A[cmask]),
            sumb=b.sum(), sumc=cc.sum(), sumAb=Ab.sum(),
            nzero_b=int((b==0).sum()), nneg_b=int((b<0).sum()),
            link_rel_L1=float(np.abs(Ab-cc).sum()/denom),
            link_int=float(np.abs((Ab-cc)).sum()),
            rel_Ab_sum=float(Ab.sum()/denom-1)))
df=pd.DataFrame(rows)
pd.set_option("display.width",250); pd.set_option("display.max_columns",50)
print(df.to_string(index=False))
print("\nmean link_rel_L1 by split", df.groupby("split").link_rel_L1.mean().to_dict())
df.to_csv(r"C:\Users\lenovo\Desktop\kaggle\work\task4\reports\structure_diag.csv", index=False)

