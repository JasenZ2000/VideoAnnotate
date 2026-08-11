from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from fastapi import HTTPException

from gpu_services_isolated.worker_process import WORKER_MANAGER, WorkerConfig


def _load_compatibility_module() -> ModuleType:
    """Load the existing job/output pipeline into a private module namespace."""
    source = Path(__file__).resolve().parents[1] / "gpu_services" / "locateanything.py"
    module_name = "gpu_services_isolated._compat_locateanything"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load LocateAnything compatibility pipeline: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_compat = _load_compatibility_module()

DEFAULT_CACHE_DIR = _compat.DEFAULT_CACHE_DIR
DEFAULT_MODEL = _compat.DEFAULT_MODEL
DEFAULT_EXTERNAL_ROOT = _compat.DEFAULT_EXTERNAL_ROOT
SETTINGS = _compat.SETTINGS
JOBS = _compat.JOBS
IMAGE_JOBS = _compat.IMAGE_JOBS
LocateAnythingInferenceReq = _compat.LocateAnythingInferenceReq
LocateAnythingVideoReq = _compat.LocateAnythingVideoReq
LocateAnythingImageDirectoryReq = _compat.LocateAnythingImageDirectoryReq


class _NoParentCuda:
    """Prevent the HTTP parent from creating CUDA contexts on worker devices."""

    @staticmethod
    def is_available() -> bool:
        return False


class _NoParentTorch:
    cuda = _NoParentCuda()


def _worker_config(device: str, dtype: str) -> WorkerConfig:
    return WorkerConfig(
        device=device,
        dtype=dtype,
        model=str(SETTINGS["model"]),
        external_root=str(SETTINGS["external_root"]),
        batch_attn=str(SETTINGS["batch_attn"]),
        vision_attn=str(SETTINGS["vision_attn"]),
        scheduler=str(SETTINGS["batch_scheduler"]),
        group_size=int(SETTINGS["batch_group_size"]),
        strict_attn=bool(SETTINGS["strict_attn"]),
        dense_backend=str(SETTINGS["dense_backend"]),
    )


def _ensure_worker(req: Any, device: str) -> Any:
    dtype = req.dtype or str(SETTINGS["default_dtype"])
    return WORKER_MANAGER.ensure(_worker_config(device, dtype))


def _release_worker(device: str, _dtype: str) -> None:
    WORKER_MANAGER.release(device)


def _dependency_preflight() -> dict[str, Any]:
    model_path = Path(str(SETTINGS["model"])).expanduser()
    external_root = Path(str(SETTINGS["external_root"])).expanduser()
    model_resolved = model_path.resolve() if model_path.exists() else model_path
    flash_spec = importlib.util.find_spec("flash_attn")
    kernel_dir = model_resolved / "kernel_utils"
    worker_path = external_root / "locateanything_worker.py"
    worker_source = ""
    if worker_path.is_file():
        try:
            worker_source = worker_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return {
        "python_executable": sys.executable,
        "model_path": str(model_resolved),
        "requested_attn": str(SETTINGS["batch_attn"]),
        "strict_attn_env": "1" if SETTINGS["strict_attn"] else "0",
        "dense_backend": str(SETTINGS["dense_backend"]),
        "flash_attn_importable": flash_spec is not None,
        "flash_attn_module": str(flash_spec.origin or "") if flash_spec else "",
        "kernel_utils_importable": kernel_dir.is_dir(),
        "kernel_utils_module": str(kernel_dir),
        "la_flash_available": bool(flash_spec is not None and kernel_dir.is_dir()),
        "preflight_scope": "parent import discovery; CUDA smoke tests run inside each GPU child",
        "worker_source_supports_batch_runtime": "use_batch_runtime" in worker_source,
    }


def health_payload() -> dict[str, Any]:
    preflight = _dependency_preflight()
    workers = WORKER_MANAGER.snapshot(query_children=False)
    worker_path = Path(str(SETTINGS["external_root"])).expanduser() / "locateanything_worker.py"
    model_path = Path(str(SETTINGS["model"])).expanduser()
    smoke_results = {
        str(row["device"]): dict(row.get("flash_smoke", {}))
        for row in workers
    }
    runtime_ready = bool(workers) and all(
        row.get("alive") and row.get("flash_smoke", {}).get("ok") for row in workers
    )
    la_flash_available = runtime_ready or bool(preflight["la_flash_available"])
    return {
        "ok": not bool(WORKER_MANAGER.startup_errors),
        "service": "locateanything-isolated",
        "architecture": "one-process-per-gpu-v2",
        "parent_pid": os.getpid(),
        "parent_cuda_contexts": False,
        "model": str(SETTINGS["model"]),
        "cache_dir": str(SETTINGS["cache_dir"]),
        "external_root": str(SETTINGS["external_root"]),
        "worker_available": worker_path.is_file(),
        "worker_importable": worker_path.is_file(),
        "worker_import_error": "",
        "batch_runtime_supported": bool(preflight["worker_source_supports_batch_runtime"]),
        "batch_utils_available": bool(model_path.is_dir() and (model_path / "batch_utils").is_dir()),
        "kernel_utils_available": bool(model_path.is_dir() and (model_path / "kernel_utils").is_dir()),
        "devices": list(SETTINGS["devices"]),
        "dtype": str(SETTINGS["default_dtype"]),
        "keep_model_loaded": bool(SETTINGS.get("keep_model_loaded", False)),
        "runtime": "batch",
        "generation_mode": "hybrid",
        "batch_size": int(SETTINGS["batch_size"]),
        "batch_attn": str(SETTINGS["batch_attn"]),
        "vision_attn": str(SETTINGS["vision_attn"]),
        "batch_scheduler": str(SETTINGS["batch_scheduler"]),
        "batch_group_size": int(SETTINGS["batch_group_size"]),
        "strict_attn": bool(SETTINGS["strict_attn"]),
        "dense_backend": str(SETTINGS["dense_backend"]),
        "attention_diagnostics": preflight,
        "la_flash_available": la_flash_available,
        "la_flash_cuda_smoke": smoke_results,
        "min_expected_fps": float(SETTINGS["min_expected_fps"]),
        "scheduler": "per-device-process-v2",
        "parallel_jobs": True,
        "image_directory_jobs": True,
        "image_directory_discovery": True,
        "model_loaded": bool(workers),
        "loaded_workers": workers,
        "worker_count": len(workers),
        "expected_worker_count": len(SETTINGS["devices"]),
        "startup_errors": dict(WORKER_MANAGER.startup_errors),
        "cuda_available": runtime_ready if workers else None,
        "device_pool": _compat.GPU_DEVICE_POOL.snapshot(),
        "allowed_roots": [str(root) for root in SETTINGS.get("allowed_roots", [])],
        "output_allowed_roots": [str(root) for root in SETTINGS.get("output_allowed_roots", [])],
    }


def configure(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model: str = DEFAULT_MODEL,
    external_root: str = DEFAULT_EXTERNAL_ROOT,
    device: str = os.environ.get("LOCANY_DEVICES", os.environ.get("LOCANY_DEVICE", "cuda")),
    dtype: str = os.environ.get("LOCANY_DTYPE", "bf16"),
    keep_model_loaded: Optional[bool] = None,
    allowed_roots: Optional[list[Path]] = None,
    output_allowed_roots: Optional[list[Path]] = None,
) -> None:
    WORKER_MANAGER.close_all()
    _compat.configure(
        cache_dir=cache_dir,
        model=model,
        external_root=external_root,
        device=device,
        dtype=dtype,
        keep_model_loaded=keep_model_loaded,
        allowed_roots=allowed_roots,
        output_allowed_roots=output_allowed_roots,
    )


def preload_workers() -> list[dict[str, Any]]:
    worker_path = Path(str(SETTINGS["external_root"])) / "locateanything_worker.py"
    if not worker_path.is_file():
        raise RuntimeError(f"LOCATEANYTHING_ROOT does not contain locateanything_worker.py: {worker_path.parent}")
    for device in SETTINGS["devices"]:
        WORKER_MANAGER.ensure(_worker_config(device, str(SETTINGS["default_dtype"])))
    return WORKER_MANAGER.snapshot(query_children=True)


# The compatibility module continues to own request validation, job state,
# media I/O and output writing. Only these CUDA-facing hooks are replaced.
_compat._ensure_worker = _ensure_worker
_compat._release_worker = _release_worker
_compat._torch = lambda: _NoParentTorch()
_compat.health_payload = health_payload
router = _compat.router


@router.get("/workers")
async def worker_status(refresh: bool = False) -> dict[str, Any]:
    return {
        "architecture": "one-process-per-gpu-v2",
        "workers": WORKER_MANAGER.snapshot(query_children=refresh),
        "startup_errors": dict(WORKER_MANAGER.startup_errors),
    }


@router.post("/workers/preload")
async def preload_worker_endpoint() -> dict[str, Any]:
    try:
        workers = await asyncio.to_thread(preload_workers)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "workers": workers}

