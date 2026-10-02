"""Download only the bottle category from a pinned public MVTec AD mirror."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
REPO = "foersben/mvtec-ad"

def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "InspectAI-research-demo/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()

def main():
    data = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}"))
    files = [item["rfilename"] for item in data["siblings"] if item["rfilename"].startswith("bottle/")]
    def download(name):
        target = ROOT / "data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            for attempt in range(4):
                try:
                    content = fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}")
                    target.write_bytes(content)
                    break
                except Exception:
                    if attempt == 3: raise
                    time.sleep(1 + attempt)
        return {"path": name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for count, record in enumerate(pool.map(download, files), 1):
            records.append(record)
            if count % 25 == 0: print(f"Downloaded/verified {count}/{len(files)}", flush=True)
    (ROOT / "data" / "manifest.json").write_text(json.dumps({"mirror": REPO, "revision": REVISION, "files": records}, indent=2))
    print(f"Complete: {len(files)} bottle files", flush=True)

if __name__ == "__main__": main()
