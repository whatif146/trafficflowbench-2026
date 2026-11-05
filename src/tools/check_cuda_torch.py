"""Check whether a CUDA 12.8 torch wheel exists for this Python/Windows."""
import re
import sys

import requests

url = "https://download.pytorch.org/whl/cu128/torch/"
r = requests.get(url, timeout=90)
print("status", r.status_code, "len", len(r.text))
cands = re.findall(r'torch-[0-9][^"<]*cp313[^"<]*win_amd64[^"<]*\.whl', r.text)
print("cp313 win_amd64 wheels:", len(cands))
for c in sorted(set(cands))[-10:]:
    print(c)
cands2 = re.findall(r'torch-[0-9][^"<]*cp312[^"<]*win_amd64[^"<]*\.whl', r.text)
print("cp312 win_amd64 wheels:", len(cands2))
for c in sorted(set(cands2))[-3:]:
    print(c)
