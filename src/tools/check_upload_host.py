"""Diagnose which Google frontend IPs are reachable for the kaggle upload URL."""
import socket
import sys
import time

import requests

host = "www.googleapis.com"
try:
    infos = socket.getaddrinfo(host, 443, family=socket.AF_INET, proto=socket.IPPROTO_TCP)
except Exception as e:
    print("dns A failed", e)
    infos = []
ips = sorted({i[4][0] for i in infos})
print("A records:", ips)
good = []
for ip in ips:
    t = time.time()
    try:
        s = socket.create_connection((ip, 443), timeout=5)
        s.close()
        print(f"  {ip}: TCP OK {time.time()-t:.2f}s")
        good.append(ip)
    except Exception as e:
        print(f"  {ip}: FAIL {type(e).__name__} {time.time()-t:.2f}s")

for ip in good:
    t = time.time()
    try:
        r = requests.get(f"https://{ip}/", headers={"Host": host}, timeout=8, verify=False)
        print(f"  {ip} HTTPS {r.status_code} {time.time()-t:.2f}s")
    except Exception as e:
        print(f"  {ip} HTTPS FAIL {type(e).__name__} {time.time()-t:.2f}s")
