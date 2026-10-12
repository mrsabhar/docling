"""Verify (or record) the corpus SHA256 hashes and page counts.

    python experiments/intel-cpu/scripts/corpus.py            # verify
    python experiments/intel-cpu/scripts/corpus.py --update   # record hashes
    python experiments/intel-cpu/scripts/corpus.py --out DIR  # also write a manifest

Exit code 1 if a file is missing or its hash differs from config/corpus.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import CORPUS_FILE, REPO_ROOT, load_json, sha256_file, write_json


def page_count(path: Path) -> int | None:
    try:
        import pypdfium2
    except ImportError:
        return None
    pdf = pypdfium2.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def load_corpus() -> list[dict[str, Any]]:
    return load_json(CORPUS_FILE)["documents"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = load_json(CORPUS_FILE)
    ok = True
    for doc in config["documents"]:
        path = REPO_ROOT / doc["path"]
        if not path.exists():
            print(f"MISSING  {doc['path']}")
            ok = False
            continue
        digest = sha256_file(path)
        pages = page_count(path)
        if args.update:
            doc["sha256"] = digest
            doc["bytes"] = path.stat().st_size
            if pages is not None:
                doc["pages"] = pages
        status = "OK" if doc.get("sha256") == digest else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"{status:8} {digest}  {pages!s:>3}p  {doc['path']}")

    if args.update:
        write_json(CORPUS_FILE, config)
    if args.out is not None:
        write_json(args.out / "corpus.json", config)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
