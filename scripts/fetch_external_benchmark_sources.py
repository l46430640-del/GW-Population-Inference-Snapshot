"""Cache the external comparison's papers and frozen reference implementation."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "external_data/literature/external_benchmark"
MANIFEST = ROOT / "docs/analysis/external_benchmark_sources.json"
ASSETS = [
    {"name": f"{paper}.{suffix}", "url": f"https://arxiv.org/{kind}/{paper}",
     "kind": kind}
    for paper, kind, suffix in (
        ("2304.08046v2", "pdf", "pdf"),
        ("2304.08046v1", "src", "tar.gz"),
        ("2404.03166v1", "pdf", "pdf"),
        ("2404.03166v1", "src", "tar.gz"),
    )
] + [{
    "name": "gppop_spin_dev_ee5ffc421.tar.gz",
    "url": "https://zenodo.org/records/14691458/files/gppop_spin_dev_ee5ffc421.tar.gz?download=1",
    "kind": "code",
    "md5": "526b9a2e7508a964c16c60b4de0c0a86",
    "record": "https://zenodo.org/records/14691458",
}]
MAX_BYTES = 32 * 1024 * 1024


def extract_text(archive_path, target, kind):
    """Read only regular text sources, keeping all archive paths in the cache."""
    target = Path(target).resolve()
    suffixes = {".tex", ".bib", ".bbl"}
    if kind == "code":
        suffixes |= {".py", ".md", ".txt", ".toml", ".cfg", ".in"}
    sources = []
    size = 0
    with tarfile.open(archive_path) as archive:
        for member in archive:
            resolved = (target / member.name).resolve()
            if not resolved.is_relative_to(target):
                raise ValueError("Unsafe archive path")
            if not member.isfile():
                continue
            if resolved.suffix.lower() not in suffixes and resolved.name != "LICENSE":
                continue
            size += member.size
            if size > MAX_BYTES:
                raise ValueError("Extracted text exceeds the size limit")
            content = archive.extractfile(member).read()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_bytes(content)
            sources.append({"path": resolved.relative_to(target).as_posix(),
                            "sha256": hashlib.sha256(content).hexdigest()})
    return sources


def fetch(asset, previous=None):
    path = CACHE / asset["name"]
    if path.exists():
        data = path.read_bytes()
    else:
        data = bytearray()
        with requests.get(asset["url"], stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise ValueError("Download exceeds the size limit")
        data = bytes(data)
    if len(data) > MAX_BYTES:
        raise ValueError("Cached file exceeds the size limit")
    digest = hashlib.sha256(data).hexdigest()
    if previous and previous["sha256"] != digest:
        raise ValueError(f"Source changed: {asset['name']}")
    if asset.get("md5") and hashlib.md5(data).hexdigest() != asset["md5"]:
        raise ValueError("Published archive checksum mismatch")
    if asset["kind"] == "pdf" and not data.startswith(b"%PDF"):
        raise ValueError("Response is not a PDF")
    if asset["kind"] != "pdf" and data.lstrip().startswith(b"<"):
        raise ValueError("Source endpoint returned HTML")
    if not path.exists():
        path.write_bytes(data)
    row = {**asset, "local_path": path.relative_to(ROOT).as_posix(),
           "bytes": len(data), "sha256": digest}
    if asset["kind"] != "pdf":
        row["text_sources"] = extract_text(path, CACHE / asset["name"].removesuffix(".tar.gz"),
                                           asset["kind"])
    return row


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    previous = {row["name"]: row for row in old}
    rows = []
    for asset in ASSETS:
        rows.append(fetch(asset, previous.get(asset["name"])))
        print(asset["name"], rows[-1]["bytes"], "verified", flush=True)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
