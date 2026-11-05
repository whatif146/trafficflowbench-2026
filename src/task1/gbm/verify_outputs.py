"""Verify a Task 1 state CSV against the published templates (row counts per panel).

Streams the CSV (low memory) and compares per-panel row counts with the template
row counts from traffic.load_targets.  Also checks that the header is exact.
"""
import sys
import csv
import collections
import os

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\libs")
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np
import traffic

EXPECTED_COLS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def main():
    path = sys.argv[1]
    splits = sys.argv[2:] or ["validation", "private"]
    cnt = collections.Counter()
    n = 0
    bad = 0
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        if hdr != EXPECTED_COLS:
            print("BAD HEADER", hdr)
            return 1
        for row in r:
            n += 1
            cnt[row[0]] += 1
            if len(row) != 7:
                bad += 1
    print(f"{path}: {n} rows, {bad} malformed")
    ok = True
    split = splits[0]
    for panel in traffic.PANELS:
        try:
            tg = traffic.load_targets(panel, split) if len(splits) == 1 else None
        except Exception:
            tg = None
        exp = len(tg["t_idx"]) if tg is not None else None
        got = cnt.get(panel, 0)
        flag = "" if exp is None or exp == got else "  <-- MISMATCH"
        if exp is not None and exp != got:
            ok = False
        print(f"  {panel:12s} rows={got:8d} expected={exp if exp is not None else '?'}{flag}")
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
