from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def json_request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LocateAnything HTTP server availability.")
    parser.add_argument("--server", default="http://127.0.0.1:9011")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--task", default="ground_multi")
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--generation-mode", default="hybrid")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--use-path", action="store_true", help="Send image_path instead of base64.")
    args = parser.parse_args()

    server = args.server.rstrip("/")
    print(f"health: {server}/api/health")
    health = json_request("GET", f"{server}/api/health", timeout=args.timeout)
    print(json.dumps(health, ensure_ascii=False, indent=2))

    image_path = args.image.expanduser().resolve()
    payload: dict[str, Any] = {
        "task": args.task,
        "prompt": args.prompt,
        "generation_mode": args.generation_mode,
    }
    if args.use_path:
        payload["image_path"] = str(image_path)
    else:
        payload["image_base64"] = base64.b64encode(image_path.read_bytes()).decode("ascii")

    print(f"ground: {server}/api/ground")
    start = time.perf_counter()
    result = json_request("POST", f"{server}/api/ground", payload, timeout=args.timeout)
    print(f"roundtrip_seconds: {time.perf_counter() - start:.3f}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
