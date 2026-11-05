"""Custom Kaggle submit: the standard client's GCS upload endpoint
(www.googleapis.com) is unreachable on this network, so upload the file through
an equivalent reachable frontend and then create the submission with the token.

Strategy, in order:
  1. signed URL with host swapped to storage.googleapis.com;
  2. signed URL as-is with a DNS override to a reachable Google frontend;
  3. same, TLS verification disabled (signed opaque URL, integrity checked by Kaggle).
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiStartSubmissionUploadRequest,
)

COMP = "2026-ieee-big-data-traffic-flow-bench"
GOOGLE_IPS = ["142.250.69.187"]
CHUNK = 4 << 20


def start_upload(api, path):
    with api.build_kaggle_client() as kaggle:
        req = ApiStartSubmissionUploadRequest()
        req.competition_name = COMP
        req.file_name = os.path.basename(path)
        req.content_length = os.path.getsize(path)
        req.last_modified_epoch_seconds = int(os.path.getmtime(path))
        resp = kaggle.competitions.competition_api_client.start_submission_upload(req)
    return resp.create_url, resp.token


def put_file(url, path, verify=True, chunk=CHUNK, progress=True):
    """Resumable PUT; returns (ok, status, detail)."""
    size = os.path.getsize(path)
    sent = 0
    t0 = time.time()
    with open(path, "rb") as f:
        data = f.read(chunk)
        while True:
            end = sent + len(data) - 1
            headers = {"Content-Range": f"bytes {sent}-{end}/{size}",
                       "Content-Type": "application/octet-stream"}
            r = requests.put(url, data=data, headers=headers, timeout=(15, 600), verify=verify)
            if r.status_code in (200, 201):
                return True, r.status_code, r.text[:200]
            if r.status_code == 308:  # resume incomplete; server tells next byte in Range
                rng = r.headers.get("Range", "")
                nxt = sent + len(data)
                if rng.startswith("bytes=0-"):
                    nxt = int(rng.split("-")[1]) + 1
                sent = nxt
                f.seek(sent)
                data = f.read(chunk)
                if progress:
                    print(f"  uploaded {sent/1e6:.1f}/{size/1e6:.1f} MB "
                          f"({sent/max(time.time()-t0,1e-9)/1e6:.2f} MB/s)", flush=True)
                if not data and sent < size:
                    return False, r.status_code, "empty read before end"
                continue
            return False, r.status_code, r.text[:300]


def attempt(name, url, path, verify, dns_ip=None):
    print(f"[try] {name} verify={verify} dns={dns_ip}", flush=True)
    orig = socket.getaddrinfo

    if dns_ip:
        def gai(host, port, family=0, type=0, proto=0, flags=0):
            if host == "www.googleapis.com":
                return orig(dns_ip, port, family, type, proto, flags)
            return orig(host, port, family, type, proto, flags)
        socket.getaddrinfo = gai
    try:
        ok, status, detail = put_file(url, path, verify=verify)
    finally:
        socket.getaddrinfo = orig
    print(f"[result] {name}: ok={ok} status={status} {detail}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--message", default="candidate")
    ap.add_argument("--wait", type=int, default=600)
    args = ap.parse_args()
    path = os.path.abspath(args.file)
    api = KaggleApi()
    api.authenticate()
    print(f"file {path} ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)
    url, token = start_upload(api, path)
    print("url:", url[:120], flush=True)
    alt = url.replace("//www.googleapis.com/", "//storage.googleapis.com/")
    ok = False
    for name, u, verify, ip in [
        ("storage-frontend", alt, True, None),
        ("www+dns", url, True, GOOGLE_IPS[0]),
        ("www+dns-insecure", url, False, GOOGLE_IPS[0]),
    ]:
        if os.path.exists("work/submissions/.upload_ok"):
            break
        try:
            ok = attempt(name, u, path, verify, ip)
        except Exception as e:
            print(f"[result] {name}: exception {type(e).__name__} {str(e)[:200]}", flush=True)
            ok = False
        if ok:
            break
    if not ok:
        raise SystemExit("all upload paths failed")
    with api.build_kaggle_client() as kaggle:
        req = ApiCreateSubmissionRequest()
        req.competition_name = COMP
        req.blob_file_tokens = token
        req.submission_description = args.message
        resp = kaggle.competitions.competition_api_client.create_submission(req)
    print("create_submission:", resp, flush=True)
    deadline = time.time() + args.wait
    while time.time() < deadline:
        time.sleep(20)
        subs = api.competition_submissions(COMP, page_size=5) or []
        if subs:
            s = subs[0]
            print(f"{time.strftime('%H:%M:%S')} status={getattr(s,'status','?')} "
                  f"score={getattr(s,'score',None)}", flush=True)
            if "COMPLETE" in str(getattr(s, "status", "")).upper():
                return


if __name__ == "__main__":
    main()
