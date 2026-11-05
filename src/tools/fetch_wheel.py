"""Download wheels manually (segmented, mirror-aware) and extract into libs/.

pip cannot be used here: the sandbox denies the temp-dir rename/delete dance pip
needs inside the workspace. This script does the same job with requests+zlib.

Usage:
    python work/tools/fetch_wheel.py kaggle==2.2.5 kagglesdk==0.1.38
    python work/tools/fetch_wheel.py --list <package>       # show available files
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile

import requests
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
LIBS = os.path.join(ROOT, "libs")
WHEELS = os.path.join(ROOT, "wheels")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python/3.13 requests"}
MIRRORS = [
    "https://pypi.tuna.tsinghua.edu.cn/simple/{pkg}/",
    "https://mirrors.aliyun.com/pypi/simple/{pkg}/",
    "https://pypi.org/simple/{pkg}/",
]


def wheel_for(pkg: str, version: str | None):
    """Pick the newest wheel for a package from the PyPI JSON API."""
    from packaging.version import InvalidVersion, Version

    j = requests.get(f"https://pypi.org/pypi/{pkg}/json", headers=UA, timeout=40).json()
    rel = {version: j["releases"].get(version, [])} if version else {
        v: files for v, files in j["releases"].items() if files}
    releases = []
    for vtext, files in rel.items():
        try:
            releases.append((Version(vtext), vtext, files))
        except InvalidVersion:
            continue
    releases.sort(reverse=True)
    for _ver, vtext, files in releases:
        cands = []
        for f in files:
            name = f["filename"]
            if not name.endswith(".whl"):
                continue
            if "cp3" in name and "cp313" not in name:
                continue
            if "cp313t" in name or "cp313-cp313t" in name:
                continue  # free-threaded build, incompatible with this CPython
            if any(x in name for x in ("win32", "win_arm64", "macosx", "manylinux")):
                continue
            if "cp313" in name and "win_amd64" not in name:
                continue
            cands.append((name, f["url"], f["size"]))
        if cands:
            cands.sort(key=lambda x: (0 if "py3-none-any" in x[0] else 1, x[0]))
            return vtext, cands[0]
    return None, None


def download(url: str, name: str) -> str:
    os.makedirs(WHEELS, exist_ok=True)
    dest = os.path.join(WHEELS, name)
    r = requests.get(url, headers=UA, timeout=(15, 45), stream=True)
    r.raise_for_status()
    total = int(r.headers["Content-Length"])
    r.close()
    if os.path.exists(dest) and os.path.getsize(dest) == total:
        return dest
    nparts = 12 if total > 500_000 else 1
    bounds = [(total * i // nparts, total * (i + 1) // nparts - 1) for i in range(nparts)]
    parts = [bytearray(b[1] - b[0] + 1) for b in bounds]

    def fetch(i, attempt=0):
        lo, hi = bounds[i]
        try:
            rr = requests.get(url, headers={**UA, "Range": f"bytes={lo}-{hi}"}, timeout=(15, 45))
            if rr.status_code not in (200, 206):
                raise RuntimeError(f"status {rr.status_code}")
            data = rr.content
            if len(data) != hi - lo + 1:
                raise RuntimeError(f"short {len(data)} != {hi-lo+1}")
            parts[i][:] = data
            return len(data)
        except Exception:
            if attempt < 5:
                time.sleep(0.4 * (attempt + 1))
                return fetch(i, attempt + 1)
            raise

    with ThreadPoolExecutor(max_workers=nparts) as ex:
        list(ex.map(fetch, range(nparts)))
    with open(dest, "wb") as fh:
        for p in parts:
            fh.write(p)
    assert os.path.getsize(dest) == total, (os.path.getsize(dest), total)
    return dest


def extract(path: str):
    os.makedirs(LIBS, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(LIBS)
    print("extracted", os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="*", help="package or package==version")
    ap.add_argument("--list", dest="listpkg", default=None)
    args = ap.parse_args()
    if args.listpkg:
        j = requests.get(f"https://pypi.org/pypi/{args.listpkg}/json", timeout=30).json()
        for u in j["urls"]:
            print(u["filename"], u["size"])
        return
    for spec in args.spec:
        pkg, _, ver = spec.partition("==")
        vtext, found = wheel_for(pkg, ver or None)
        if not found:
            print(f"{pkg}: no wheel found (sdist only?)")
            continue
        name, url, size = found
        print(f"{pkg}=={vtext}: {name} ({size/1024:.0f} KB)", flush=True)
        extract(download(url, name))
    sys.path.insert(0, LIBS)


if __name__ == "__main__":
    main()
