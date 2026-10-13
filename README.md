# Process-isolated LocateAnything GPU Service

This is a separate test service. It keeps the existing LocateAnything job API
and output layout, but replaces the in-process worker cache with one spawned
Python process per CUDA device. This is required because LocateAnything's
`batch_utils.hybrid_runtime` stores its model in process-global state.

## Start

Edit the path/device defaults in `run_gpu_service.sh`, or export overrides, then:

```bash
chmod +x gpu_services_isolated/run_gpu_service.sh
GPU_SERVICE_ENV_ACTIVATE=/home/hx/miniforge3/envs/zzs_la/bin/activate \
LOCANY_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
bash gpu_services_isolated/run_gpu_service.sh
```

The default port is `10115`, so it can run beside the old service on `10114`.
`LOCANY_PRELOAD_WORKERS=1` loads and smoke-tests every GPU before the server
accepts requests. Set it to `0` only when lazy loading is preferable.

## Verify isolation

```bash
curl -s http://127.0.0.1:10115/api/locateanything/health | python -m json.tool
curl -s 'http://127.0.0.1:10115/api/locateanything/workers?refresh=true' | python -m json.tool
watch -n 1 nvidia-smi
```

Expected evidence:

- `architecture` is `one-process-per-gpu-v2`;
- `worker_count` equals the number of configured devices;
- every `loaded_workers` row has a different PID;
- each row's `device` equals `actual_device`;
- every card holds its own model allocation (roughly the same large VRAM use),
  while `parent_pid` does not own a model-sized CUDA allocation;
- every device has `flash_smoke.ok=true`.

Point the existing Batch Tool at `http://<gpu-host>:10115`. No inference-mode
control is needed: the service remains fixed to batch-hybrid-4.

## Failure behavior

- A child startup/import/FlashAttention failure aborts preload and is retained
  in `startup_errors`.
- CUDA OOM is returned to the existing resilient batch splitter, which retries
  4 as 2+2 and then individual images; cache cleanup happens in the owning child.
- IPC startup and inference timeouts are controlled by
  `LOCANY_WORKER_START_TIMEOUT` and `LOCANY_WORKER_RPC_TIMEOUT`.
- A request for a different dtype replaces only that GPU's child, preventing two
  retained model copies from silently filling one card.

