"""Extract readable text from the downloaded public notebooks for review."""
import glob
import json
import os

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\work\research"


def extract(nb_path, out_path):
    data = json.load(open(nb_path, encoding="utf-8"))
    parts = []
    for i, c in enumerate(data.get("cells", [])):
        src = "".join(c.get("source", []))
        kind = c.get("cell_type")
        parts.append(f"\n\n########## {kind} cell {i} ##########\n{src}")
    text = "".join(parts)
    open(out_path, "w", encoding="utf-8").write(text)
    return len(text)


for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p):
        continue
    nbs = glob.glob(os.path.join(p, "*.ipynb"))
    for nb in nbs:
        out = os.path.join(p, "cells.txt")
        n = extract(nb, out)
        print(f"{d}: {n} chars -> {out}")
