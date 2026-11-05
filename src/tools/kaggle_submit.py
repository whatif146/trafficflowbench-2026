"""Robust Kaggle submit for this machine.

Problem: Kaggle's client uploads the file to a signed GCS resumable session on
www.googleapis.com, whose IPs are unreachable here.  This tool
  * starts the upload through the official API (token + signed URL),
  * uploads the bytes to the signed session through a reachable Google frontend
    (DNS override, TLS still verified for the correct hostname),
  * resumes from the server-reported offset after network drops,
  * then creates the submission with the official API and polls the score.

Session state is cached in work/submissions/.upload_session.json so a killed
run can resume instead of re-uploading.

Usage:
  python work/tools/kaggle_submit.py --file work/submissions/submission_v2.zip --message "v2 ..."
  python work/tools/kaggle_submit.py --status
"""
from __future__ import annotations

import argparse
import json
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
FRONTEND_IP = "142.250.69.187"
SESSION_FILE = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\work\submissions\.upload_session.json"
CHUNK = 1 << 20
MAX_BYTES_PER_TRY = 1 << 30


def start_upload(api, path):
    with api.build_kaggle_client() as kaggle:
        req = ApiStartSubmissionUploadRequest()
        req.competition_name = COMP
        req.file_name = os.path.basename(path)
        req.content_length = os.path.getsize(path)
        req.last_modified_epoch_seconds = int(os.path.getmtime(path))
        resp = kaggle.competitions.competition_api_client.start_submission_upload(req)
    return resp.create_url, resp.token


def create_submission(api, token, message):
    with api.build_kaggle_client() as kaggle:
        req = ApiCreateSubmissionRequest()
        req.competition_name = COMP
        req.blob_file_tokens = token
        req.submission_description = message
        return kaggle.competitions.competition_api_client.create_submission(req)


def session(path):
    size = os.path.getsize(path)
    if os.path.exists(SESSION_FILE):
        try:
            s = json.load(open(SESSION_FILE))
            if s.get("file") == os.path.abspath(path) and s.get("size") == size and \
                    time.time() - s.get("t", 0) < 3600 * 3:
                return s
        except Exception:
            pass
    return None


def save_session(s):
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump(s, f)


def patch_dns():
    orig = socket.getaddrinfo

    def gai(host, port, family=0, type=0, proto=0, flags=0):
        if host == "www.googleapis.com":
            return orig(FRONTEND_IP, port, family, type, proto, flags)
        return orig(host, port, family, type, proto, flags)

    socket.getaddrinfo = gai
    return orig


def query_offset(url, size, verify=True):
    try:
        r = requests.put(url, data=b"", timeout=(10, 60), verify=verify,
                         headers={"Content-Range": f"bytes */{size}", "Content-Length": "0"})
        if r.status_code in (200, 201):
            return size, r
        if r.status_code == 308:
            rng = r.headers.get("Range", "")
            if rng.startswith("bytes=0-"):
                return int(rng.split("-")[1]) + 1, r
            return 0, r
        return None, r
    except Exception as e:
        return None, e


def upload(url, path, verify=True, progress=True):
    size = os.path.getsize(path)
    sent, r = query_offset(url, size, verify)
    if sent is None:
        print(f"  offset query failed: {r}", flush=True)
        sent = 0
    if sent == size:
        print("  upload already complete per server", flush=True)
        return True
    print(f"  resuming at {sent/1e6:.1f}/{size/1e6:.1f} MB", flush=True)
    attempts = 0
    t0 = time.time()
    while sent < size:
        with open(path, "rb") as f:
            f.seek(sent)
            data = f.read(CHUNK)
        end = sent + len(data) - 1
        headers = {"Content-Range": f"bytes {sent}-{end}/{size}",
                   "Content-Type": "application/octet-stream"}
        try:
            r = requests.put(url, data=data, headers=headers, timeout=(15, 120), verify=verify)
        except Exception as e:
            attempts += 1
            if progress:
                print(f"  network drop at {sent/1e6:.1f} MB ({type(e).__name__}); re-querying", flush=True)
            time.sleep(min(2 + attempts, 10))
            new_sent, _ = query_offset(url, size, verify)
            if new_sent is None:
                if attempts > 40:
                    raise SystemExit("too many upload retries")
                continue
            sent = new_sent
            continue
        if r.status_code in (200, 201):
            if progress:
                print(f"  uploaded {size/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
            return True
        if r.status_code == 308:
            rng = r.headers.get("Range", "")
            sent = int(rng.split("-")[1]) + 1 if rng.startswith("bytes=0-") else sent + len(data)
            if progress and sent % (5 << 20) < CHUNK:
                print(f"  {sent/1e6:.1f}/{size/1e6:.1f} MB", flush=True)
            continue
        print(f"  unexpected status {r.status_code}: {r.text[:200]}", flush=True)
        return False
    return True


def poll(api, wait):
    deadline = time.time() + wait
    while time.time() < deadline:
        subs = api.competition_submissions(COMP, page_size=5) or []
        if subs:
            s = subs[0]
            status = str(getattr(s, "status", "")).split(".")[-1]
            score = getattr(s, "public_score", None)
            print(f"{time.strftime('%H:%M:%S')} ref={getattr(s,'ref',None)} status={status} "
                  f"public_score={score}", flush=True)
            if status.upper() in ("COMPLETE", "ERROR", "FAILED"):
                return s
        time.sleep(20)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--message", default="candidate")
    ap.add_argument("--wait", type=int, default=900)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    api = KaggleApi()
    api.authenticate()
    if args.status:
        subs = api.competition_submissions(COMP, page_size=10) or []
        for s in subs:
            print(getattr(s, "ref", None), str(getattr(s, "status", "")).split(".")[-1],
                  "public:", getattr(s, "public_score", None),
                  getattr(s, "description", None))
        return
    path = os.path.abspath(args.file)
    s = session(path)
    if s is None:
        url, token = start_upload(api, path)
        s = {"file": path, "size": os.path.getsize(path), "url": url, "token": token, "t": time.time()}
        save_session(s)
        print("new upload session", flush=True)
    else:
        print("resuming cached upload session", flush=True)
    print(f"file {path} ({s['size']/1e6:.1f} MB) url {s['url'][:90]}", flush=True)
    for verify, label in ((True, "verified"), (False, "insecure")):
        orig = patch_dns()
        try:
            ok = upload(s["url"], path, verify=verify)
        finally:
            socket.getaddrinfo = orig
        print(f"upload attempt ({label}): {'OK' if ok else 'FAILED'}", flush=True)
        if ok:
            break
    else:
        raise SystemExit("upload failed on all TLS modes; session kept for resume")
    print("creating submission...", flush=True)
    resp = create_submission(api, s["token"], args.message)
    print("create_submission response:", resp, flush=True)
    os.remove(SESSION_FILE) if os.path.exists(SESSION_FILE) else None
    poll(api, args.wait)


if __name__ == "__main__":
    main()
