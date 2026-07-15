from __future__ import annotations

import threading
from collections import defaultdict
from typing import Iterable, Optional


def normalize_device(value: str) -> str:
    device = str(value).strip().lower()
    if not device or device == "cuda":
        return "cuda:0"
    if device.isdigit():
        return f"cuda:{device}"
    return device


def parse_devices(value: str | Iterable[str]) -> list[str]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    devices: list[str] = []
    for item in raw:
        device = normalize_device(str(item))
        if device not in devices:
            devices.append(device)
    return devices or ["cuda:0"]


class DevicePool:
    """Process-local exclusive leases shared by all GPU service runtimes."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._busy: set[str] = set()
        self._waiting: dict[str, int] = defaultdict(int)
        self._cursor = 0

    def validate(self, candidates: Iterable[str], requested: Optional[str]) -> None:
        devices = parse_devices(candidates)
        if requested and requested.strip().lower() != "auto":
            device = normalize_device(requested)
            if device not in devices:
                raise ValueError(
                    f"device {device} is not enabled; available devices: {', '.join(devices)}"
                )

    def acquire(self, candidates: Iterable[str], requested: Optional[str] = None) -> str:
        devices = parse_devices(candidates)
        self.validate(devices, requested)
        preferred = (
            normalize_device(requested)
            if requested and requested.strip().lower() != "auto"
            else None
        )
        choices = [preferred] if preferred else devices
        waiting_key = preferred or "auto"
        with self._condition:
            self._waiting[waiting_key] += 1
            try:
                while True:
                    if preferred:
                        if preferred not in self._busy:
                            selected = preferred
                            break
                    else:
                        for offset in range(len(choices)):
                            index = (self._cursor + offset) % len(choices)
                            candidate = choices[index]
                            if candidate not in self._busy:
                                selected = candidate
                                self._cursor = (index + 1) % len(choices)
                                break
                        else:
                            selected = ""
                        if selected:
                            break
                    self._condition.wait()
                self._busy.add(selected)
                return selected
            finally:
                self._waiting[waiting_key] -= 1
                if self._waiting[waiting_key] <= 0:
                    self._waiting.pop(waiting_key, None)

    def release(self, device: str) -> None:
        normalized = normalize_device(device)
        with self._condition:
            self._busy.discard(normalized)
            self._condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "busy_devices": sorted(self._busy),
                "waiting": dict(sorted(self._waiting.items())),
            }


GPU_DEVICE_POOL = DevicePool()
