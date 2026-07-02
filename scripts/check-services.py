#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def check(name: str, base_url: str, timeout: float) -> bool:
    url = f"{base_url.rstrip('/')}/api/health"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: Any = json.loads(response.read().decode("utf-8"))
        ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {url}")
        if not ok:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return ok
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {name}: {url}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check video annotation service health endpoints.")
    parser.add_argument("--platform", help="Workflow platform base URL")
    parser.add_argument("--sam31", help="SAM3.1 service base URL")
    parser.add_argument("--locateanything", help="LocateAnything service base URL")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    targets = [
        ("platform", args.platform),
        ("sam31", args.sam31),
        ("locateanything", args.locateanything),
    ]
    selected = [(name, url) for name, url in targets if url]
    if not selected:
        parser.error("provide at least one service URL")

    return 0 if all(check(name, url, args.timeout) for name, url in selected) else 1


if __name__ == "__main__":
    sys.exit(main())
