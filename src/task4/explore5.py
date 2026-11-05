import pyarrow.parquet as pq, os, glob
p="D12_I405_N"
base=os.path.join(r"C:\Users\lenovo\Desktop\kaggle\kaggle_public\corridors",p)
for sub in ["train/mainline_states","train/ramp_states","validation/ramp_states","validation/mainline_states_masked/mask_regime=R1"]:
    fs=sorted(glob.glob(os.path.join(base,sub,"**","*.parquet"),recursive=True))
    print(sub, "nfiles",len(fs), "first", os.path.basename(fs[0]) if fs else None)
    if fs:
        t=pq.read_table(fs[0])
        print("  rows",t.num_rows,"cols",t.column_names)
        print("  sample", t.slice(0,2).to_pylist())
