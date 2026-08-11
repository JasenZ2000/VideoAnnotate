from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class RemoteWorkerError(RuntimeError):
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        detail = str(payload.get("error", "isolated worker failed"))
        remote_type = str(payload.get("error_type", "RuntimeError"))
        remote_traceback = str(payload.get("traceback", ""))
        message = f"{remote_type}: {detail}"
        if remote_traceback:
            message += f"\nRemote traceback:\n{remote_traceback}"
        super().__init__(message)


@dataclass(frozen=True)
class WorkerConfig:
    device: str
    dtype: str
    model: str
    external_root: str
    batch_attn: str
    vision_attn: str
    scheduler: str
    group_size: int
    strict_attn: bool
    dense_backend: str


def _dtype_from_name(torch: Any, name: str) -> Any:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _set_runtime_environment(config: WorkerConfig) -> None:
    os.environ["LA_FLASH_MODEL"] = config.model
    os.environ["LA_FLASH_ATTN"] = config.batch_attn
    os.environ["LA_FLASH_VISION_ATTN"] = config.vision_attn
    os.environ["LA_FLASH_HYBRID_SCHEDULER"] = config.scheduler
    os.environ["LA_FLASH_HYBRID_GROUP_SIZE"] = str(config.group_size)
    os.environ["LA_FLASH_STRICT_ATTN"] = "1" if config.strict_attn else "0"
    os.environ["LA_FLASH_DENSE_BACKEND"] = config.dense_backend


def _add_runtime_paths(config: WorkerConfig) -> None:
    for raw_path in (config.model, config.external_root):
        path = Path(raw_path).expanduser().resolve()
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _flash_smoke_test(torch: Any, device: str) -> dict[str, Any]:
    from flash_attn import flash_attn_varlen_func

    with torch.inference_mode():
        query = torch.randn((4, 2, 64), device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 4], device=device, dtype=torch.int32)
        output = flash_attn_varlen_func(
            query,
            query,
            query,
            cu_seqlens,
            cu_seqlens,
            4,
            4,
            dropout_p=0.0,
            causal=True,
        )
        torch.cuda.synchronize()
    if tuple(output.shape) != tuple(query.shape):
        raise RuntimeError(f"Unexpected FlashAttention output shape: {tuple(output.shape)}")
    return {"ok": True, "shape": list(output.shape)}


def _child_main(connection: Any, raw_config: dict[str, Any]) -> None:
    config = WorkerConfig(**raw_config)
    worker: Any = None
    torch: Any = None
    try:
        _set_runtime_environment(config)
        _add_runtime_paths(config)

        # CUDA must only be imported and initialized after this spawned process
        # knows which physical device it owns.
        import torch as torch_module

        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in isolated LocateAnything worker")
        torch.cuda.set_device(config.device)
        actual_index = int(torch.cuda.current_device())
        actual_device = f"cuda:{actual_index}"
        if actual_device != config.device:
            raise RuntimeError(
                f"CUDA binding mismatch: requested {config.device}, current device is {actual_device}"
            )

        import flash_attn
        import kernel_utils

        kernel_available = getattr(kernel_utils, "is_available", None)
        if not callable(kernel_available) or not kernel_available():
            raise RuntimeError(f"kernel_utils is unavailable: {getattr(kernel_utils, '__file__', '')}")
        smoke = _flash_smoke_test(torch, config.device)

        from locateanything_worker import LocateAnythingWorker

        worker = LocateAnythingWorker(
            config.model,
            device=config.device,
            dtype=_dtype_from_name(torch, config.dtype),
            use_batch_runtime=True,
            attn=config.batch_attn,
            vision_attn=config.vision_attn,
            scheduler=config.scheduler,
            group_size=config.group_size,
            strict_attn=config.strict_attn,
        )
        torch.cuda.synchronize()
        connection.send({
            "kind": "ready",
            "pid": os.getpid(),
            "device": config.device,
            "actual_device": actual_device,
            "gpu_name": str(torch.cuda.get_device_name(actual_index)),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "flash_attn_version": str(getattr(flash_attn, "__version__", "unknown")),
            "flash_attn_module": str(getattr(flash_attn, "__file__", "")),
            "kernel_utils_module": str(getattr(kernel_utils, "__file__", "")),
            "flash_smoke": smoke,
            "model_memory_gib": float(torch.cuda.memory_allocated()) / (1024 ** 3),
        })

        while True:
            command = connection.recv()
            operation = command.get("op")
            if operation == "close":
                connection.send({"kind": "closed"})
                break
            if operation == "status":
                connection.send({
                    "kind": "result",
                    "result": {
                        "pid": os.getpid(),
                        "device": config.device,
                        "actual_device": f"cuda:{torch.cuda.current_device()}",
                        "memory_allocated_gib": float(torch.cuda.memory_allocated()) / (1024 ** 3),
                        "memory_reserved_gib": float(torch.cuda.memory_reserved()) / (1024 ** 3),
                    },
                })
                continue
            if operation == "clear_cache":
                torch.cuda.empty_cache()
                connection.send({"kind": "result", "result": True})
                continue
            if operation != "predict_batch":
                raise ValueError(f"Unsupported isolated-worker operation: {operation}")

            started = time.perf_counter()
            try:
                result = worker.predict_batch(command["pairs"], **command["kwargs"])
                torch.cuda.synchronize()
                connection.send({
                    "kind": "result",
                    "result": result,
                    "inference_seconds": time.perf_counter() - started,
                })
            except BaseException as exc:
                error_text = str(exc)
                oom = "out of memory" in error_text.lower()
                if oom:
                    try:
                        torch.cuda.empty_cache()
                    except BaseException:
                        pass
                connection.send({
                    "kind": "error",
                    "error_type": type(exc).__name__,
                    "error": error_text,
                    "traceback": traceback.format_exc(),
                    "oom": oom,
                    "device": config.device,
                    "pid": os.getpid(),
                })
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send({
                "kind": "startup_error" if worker is None else "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "device": config.device,
                "pid": os.getpid(),
            })
        except BaseException:
            pass
    finally:
        worker = None
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except BaseException:
                pass
        try:
            connection.close()
        except BaseException:
            pass


class IsolatedWorkerProxy:
    def __init__(self, config: WorkerConfig, *, startup_timeout: float, rpc_timeout: float):
        self.config = config
        self.startup_timeout = startup_timeout
        self.rpc_timeout = rpc_timeout
        self._lock = threading.Lock()
        self._closed = False
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self._connection = parent_connection
        self._process = context.Process(
            target=_child_main,
            args=(child_connection, asdict(config)),
            name=f"locany-{config.device.replace(':', '-')}",
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        try:
            message = self._receive(startup_timeout, "model startup")
        except BaseException:
            self._terminate()
            raise
        if message.get("kind") != "ready":
            self._terminate()
            raise RemoteWorkerError(message)
        self.ready_info = dict(message)

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def _receive(self, timeout: float, operation: str) -> dict[str, Any]:
        if not self._connection.poll(timeout):
            if not self._process.is_alive():
                raise RuntimeError(
                    f"LocateAnything worker {self.config.device} exited during {operation} "
                    f"with code {self._process.exitcode}"
                )
            raise TimeoutError(
                f"LocateAnything worker {self.config.device} timed out during {operation} "
                f"after {timeout:.1f}s"
            )
        try:
            return dict(self._connection.recv())
        except EOFError as exc:
            raise RuntimeError(
                f"LocateAnything worker {self.config.device} closed its IPC channel during "
                f"{operation}; exit code {self._process.exitcode}"
            ) from exc

    def _request(self, payload: dict[str, Any], operation: str) -> Any:
        with self._lock:
            if self._closed or not self._process.is_alive():
                raise RuntimeError(f"LocateAnything worker {self.config.device} is not running")
            self._connection.send(payload)
            message = self._receive(self.rpc_timeout, operation)
            if message.get("kind") == "error":
                raise RemoteWorkerError(message)
            if message.get("kind") != "result":
                raise RuntimeError(f"Unexpected isolated-worker response: {message}")
            return message.get("result")

    def predict_batch(self, pairs: list[tuple[Any, str]], **kwargs: Any) -> list[dict[str, Any]]:
        result = self._request(
            {"op": "predict_batch", "pairs": pairs, "kwargs": kwargs},
            "batch inference",
        )
        return list(result)

    def status(self) -> dict[str, Any]:
        return dict(self._request({"op": "status"}, "status"))

    def clear_cache(self) -> None:
        self._request({"op": "clear_cache"}, "clear cache")

    def _terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=5)
        try:
            self._connection.close()
        except BaseException:
            pass
        self._closed = True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._process.is_alive():
                try:
                    self._connection.send({"op": "close"})
                    if self._connection.poll(10):
                        self._connection.recv()
                except BaseException:
                    pass
            self._terminate()


class IsolatedWorkerManager:
    def __init__(self) -> None:
        self._workers: dict[str, IsolatedWorkerProxy] = {}
        self._lock = threading.RLock()
        self.startup_errors: dict[str, str] = {}
        atexit.register(self.close_all)

    def ensure(self, config: WorkerConfig) -> IsolatedWorkerProxy:
        with self._lock:
            existing = self._workers.get(config.device)
            if existing is not None and existing.config == config and existing.alive:
                return existing
            if existing is not None:
                existing.close()
                self._workers.pop(config.device, None)
            startup_timeout = float(os.environ.get("LOCANY_WORKER_START_TIMEOUT", "600"))
            rpc_timeout = float(os.environ.get("LOCANY_WORKER_RPC_TIMEOUT", "3600"))
            try:
                worker = IsolatedWorkerProxy(
                    config,
                    startup_timeout=startup_timeout,
                    rpc_timeout=rpc_timeout,
                )
            except BaseException as exc:
                self.startup_errors[config.device] = str(exc)
                raise
            self.startup_errors.pop(config.device, None)
            self._workers[config.device] = worker
            return worker

    def release(self, device: str) -> None:
        with self._lock:
            worker = self._workers.pop(device, None)
        if worker is not None:
            worker.close()

    def close_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.close()

    def snapshot(self, *, query_children: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            workers = list(self._workers.items())
        result: list[dict[str, Any]] = []
        for device, worker in sorted(workers):
            row = dict(worker.ready_info)
            row.update({"device": device, "pid": worker.pid, "alive": worker.alive})
            if query_children and worker.alive:
                try:
                    row.update(worker.status())
                except BaseException as exc:
                    row["status_error"] = str(exc)
            result.append(row)
        return result


WORKER_MANAGER = IsolatedWorkerManager()
