#!/usr/bin/env bash
# Download the checkpoints listed in MANIFEST.json into this directory and verify their sha256.
# Files released with the paper are fetched from CKPT_BASE_URL; the MAE weights from their official URL.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
base="${CKPT_BASE_URL:-https://REPLACE-WITH-RELEASE-URL/checkpoints}"
python - "$here" "$base" <<'PY'
import hashlib, json, os, sys, urllib.request
here, base = sys.argv[1], sys.argv[2]
for e in json.load(open(os.path.join(here, "MANIFEST.json")))["entries"]:
    dst = os.path.join(here, e["file"])
    url = e["source"] if e["source"].startswith("http") else f"{base.rstrip('/')}/{e['file']}"
    if not os.path.exists(dst):
        print(f"fetching {e['file']} from {url}", flush=True)
        urllib.request.urlretrieve(url, dst)
    h = hashlib.sha256()
    with open(dst, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    status = "ok" if h.hexdigest() == e["sha256"] else "SHA256 MISMATCH"
    print(f"{status}: {e['file']}")
    if status != "ok":
        sys.exit(1)
PY
