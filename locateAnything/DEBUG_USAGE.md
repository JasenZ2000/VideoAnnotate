# LocateAnything Debug Usage

These scripts are intentionally small and verbose so inference problems can be
split into model loading, one-image inference, parsing, visualization, and HTTP
serving.

## 1. Install in the LocateAnything environment

Use Python 3.10 or newer. The Hugging Face remote processor code for
`nvidia/LocateAnything-3B` uses `A | B` type annotations, which fail on
Python 3.9 during `AutoProcessor.from_pretrained(..., trust_remote_code=True)`.

```bash
cd /path/to/object-reid-clip/locateAnything
pip install -e .
```

If the model has not been cached on the server, the first run downloads
`nvidia/LocateAnything-3B` from Hugging Face.

## 2. Direct one-image inference

```bash
cd /path/to/object-reid-clip/locateAnything
python debug_infer.py \
  --model nvidia/LocateAnything-3B \
  --image /path/to/test.jpg \
  --task ground_multi \
  --prompt "person wearing black clothes" \
  --device cuda:0 \
  --dtype bf16 \
  --generation-mode hybrid \
  --resize-long-edge 1024 \
  --out-dir /tmp/locany_debug
```

Useful variants:

```bash
python debug_infer.py --image /path/to/test.jpg --task detect --categories person,car,bicycle
python debug_infer.py --image /path/to/test.jpg --task ground_single --prompt "the man on the left"
python debug_infer.py --image /path/to/test.jpg --task point --prompt "traffic light"
python debug_infer.py --image /path/to/test.jpg --question "Locate all the instances that match the following description: red car."
```

Outputs:

- `*.json`: raw answer, parsed boxes/points, timings, model stats.
- `*_vis.jpg`: the input image with parsed boxes/points drawn.

Resize options:

- `--resize-long-edge 1024`: resize only for inference so the longer side is at
  most 1024 pixels. Parsed boxes are still mapped back to original image size.
- `--resize-scale 0.5`: uniformly scale the image for inference. Ignored when
  `--resize-long-edge` is set.
- `--save-resized-image`: save the exact resized image sent to the model.

## 3. Run a simple HTTP debug server

```bash
cd /path/to/object-reid-clip/locateAnything
python serve_locateanything.py \
  --model nvidia/LocateAnything-3B \
  --host 0.0.0.0 \
  --port 9011 \
  --device cuda:0 \
  --dtype bf16 \
  --allowed-root /data/object-reid-clip
```

`--allowed-root` is only enforced for `image_path` requests. Base64 requests do
not need shared storage.

## 4. Check server availability from another shell

Base64 mode:

```bash
python check_image_server.py \
  --server http://127.0.0.1:9011 \
  --image /path/to/test.jpg \
  --task ground_multi \
  --prompt "person wearing black clothes"
```

Shared path mode:

```bash
python check_image_server.py \
  --server http://127.0.0.1:9011 \
  --image /data/object-reid-clip/sample.jpg \
  --use-path \
  --task detect \
  --prompt "person,car,bicycle"
```

## 5. Common debug knobs

- If output format is unstable, try `--generation-mode slow`.
- If memory is tight, try `--dtype fp16`.
- If the GPU does not support bf16 well, use `--dtype fp16`.
- If the process fails during import, confirm `pip install -e .` was run inside
  `locateAnything`.
- If HTTP health is OK but inference fails, first reproduce with
  `debug_infer.py` using the same image and prompt.

## 6. Batch test-set inference to YOLO txt

The batch script accepts multiple image roots and mirrors their directory
structure under the output directory. Each output line is:

```text
class_id x_center y_center width height score
```

All coordinates are normalized to `[0, 1]`. LocateAnything does not emit a
calibrated confidence score, so `score` defaults to the configured value.

Single process:

```bash
cd /path/to/object-reid-clip/locateAnything
CUDA_VISIBLE_DEVICES=0 python batch_yolo_infer.py \
  --input-roots /data/test/folder_a /data/test/folder_b \
  --output-dir /data/test_locany_yolo \
  --task ground_multi \
  --target person \
  --class-id 0 \
  --score 1.0 \
  --resize-long-edge 1024 \
  --generation-mode slow \
  --max-new-tokens 512 \
  --raw-jsonl /data/test_locany_yolo/raw_answers.jsonl
```

Resume behavior:

- Existing `.txt` files are skipped by default.
- Add `--overwrite` to recompute them.
- Failed images are appended to `errors_shard*.jsonl`.
- Images with no parsed boxes still get an empty `.txt`, so resume is clean.

Recommended multi-GPU pattern: run independent shards, one process per GPU.

```bash
CUDA_VISIBLE_DEVICES=0 python batch_yolo_infer.py --input-roots /data/test --output-dir /data/out --num-shards 4 --shard-index 0
CUDA_VISIBLE_DEVICES=1 python batch_yolo_infer.py --input-roots /data/test --output-dir /data/out --num-shards 4 --shard-index 1
CUDA_VISIBLE_DEVICES=2 python batch_yolo_infer.py --input-roots /data/test --output-dir /data/out --num-shards 4 --shard-index 2
CUDA_VISIBLE_DEVICES=3 python batch_yolo_infer.py --input-roots /data/test --output-dir /data/out --num-shards 4 --shard-index 3
```

This is more reliable than trying to use LocateAnything's custom generation
path with a single multi-GPU `device_map`.
