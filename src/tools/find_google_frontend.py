"""Find a reachable Google frontend and test a patched-DNS HTTPS request."""
import socket
import ssl
import time
import requests

cands = ["storage.googleapis.com", "www.googleapis.com", "www.google.com",
         "accounts.google.com", "kaggle.com", "www.kaggle.com"]
reachable = []

for host in cands:
    try:
        infos = socket.getaddrinfo(host, 443, family=socket.AF_INET, proto=socket.IPPROTO_TCP)
    except Exception as e:
        print(host, "dns fail", e)
        continue
    ips = sorted({i[4][0] for i in infos})
    for ip in ips:
        try:
            s = socket.create_connection((ip, 443), timeout=4)
            s.close()
            print(f"{host} -> {ip} TCP OK")
            reachable.append((host, ip))
            break
        except Exception:
            pass
    else:
        print(host, "no reachable A record", ips)

print("\ntesting patched-DNS requests:")
orig = socket.getaddrinfo


def make_gai(target_ip):
    def gai(host, port, family=0, type=0, proto=0, flags=0):
        if host in ("www.googleapis.com", "storage.googleapis.com"):
            return orig(target_ip, port, family, type, proto, flags)
        return orig(host, port, family, type, proto, flags)
    return gai


for host, ip in reachable:
    socket.getaddrinfo = make_gai(ip)
    t = time.time()
    try:
        r = requests.get("https://www.googleapis.com/storage/v1/b?maxResults=1", timeout=10)
        print(f"via {host}/{ip}: status={r.status_code} {time.time()-t:.2f}s body={r.text[:120]}")
    except Exception as e:
        print(f"via {host}/{ip}: FAIL {type(e).__name__} {str(e)[:100]}")
    finally:
        socket.getaddrinfo = orig
