import pandas as pd, collections
KEY=r"C:\Users\lenovo\Desktop\kaggle\kaggle_public\submission_key.csv"
cnt=collections.Counter(); dep=collections.Counter()
for ch in pd.read_csv(KEY, chunksize=1_000_000, dtype=str):
    t=ch.task
    cnt.update(t.value_counts().to_dict())
    if "departure_time" in ch.columns:
        d=ch.loc[t=="odme","departure_time"]
        dep.update(d.value_counts().to_dict())
print("rows by task:", dict(cnt))
print("odme departure_time counts:", dict(dep))
print("columns:", list(ch.columns))
