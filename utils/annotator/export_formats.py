from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


Annotation = tuple[int, Sequence[float]]


def write_jpeg(path: Path, frame: np.ndarray, quality: int = 95) -> None:
    """Write a JPEG through Python so Unicode Windows paths work reliably."""
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError(f"Unable to encode JPEG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def write_voc_xml(
    path: Path,
    image_filename: str,
    width: int,
    height: int,
    annotations: Sequence[Annotation],
    class_labels: dict[int, str],
) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = image_filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for class_id, raw_bbox in annotations:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_labels.get(class_id, str(class_id))
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(
            int(x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height)
        )
        ET.SubElement(obj, "difficult").text = "0"
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(max(0, int(round(x1))))
        ET.SubElement(bbox, "ymin").text = str(max(0, int(round(y1))))
        ET.SubElement(bbox, "xmax").text = str(min(width, int(round(x2))))
        ET.SubElement(bbox, "ymax").text = str(min(height, int(round(y2))))

    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(path)
