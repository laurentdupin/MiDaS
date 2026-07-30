"""Compare the native BGR8 path with pinned MiDaS PyTorch CPU."""

from __future__ import annotations

import argparse
import csv
import ctypes
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_tensor import load_reference


class Context:
    def __init__(self, dll, model: Path, backend: str, device: int):
        self.dll = dll
        self.value = ctypes.c_void_p()
        status = (
            dll.midas_create_vulkan(
                str(model.resolve()).encode(),
                0,
                device,
                ctypes.byref(self.value),
            )
            if backend == "vulkan"
            else dll.midas_create(
                str(model.resolve()).encode(),
                0,
                ctypes.byref(self.value),
            )
        )
        if status:
            raise RuntimeError(dll.midas_last_error().decode())

    def close(self):
        if self.value:
            self.dll.midas_destroy(self.value)
            self.value = None


def configure_dll(path: Path):
    dll = ctypes.CDLL(str(path.resolve()))
    dll.midas_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    dll.midas_create.restype = ctypes.c_int
    dll.midas_create_vulkan.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    dll.midas_create_vulkan.restype = ctypes.c_int
    dll.midas_destroy.argtypes = [ctypes.c_void_p]
    dll.midas_infer_bgr8.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    dll.midas_infer_bgr8.restype = ctypes.c_int
    dll.midas_last_error.restype = ctypes.c_char_p
    return dll


def network_size(width: int, height: int, input_size: int):
    scale = min(input_size / width, input_size / height)

    def constrained(value):
        result = int(np.round(value / 32) * 32)
        if result > input_size:
            result = int(np.floor(value / 32) * 32)
        return result

    return constrained(scale * width), constrained(scale * height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--geffnet-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument(
        "--backend", choices=("cpu", "vulkan"), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    reference_model = load_reference(
        args.repository.resolve(),
        args.geffnet_source.resolve(),
        args.checkpoint.resolve(),
    )
    dll = configure_dll(args.dll)
    context = Context(dll, args.model, args.backend, args.device)
    paths = sorted(
        path
        for path in args.assets.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rows = []
    try:
        for index, path in enumerate(paths, 1):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cannot decode {path}")
            height, width = bgr.shape[:2]
            native = np.empty((height, width), dtype=np.float32)
            status = dll.midas_infer_bgr8(
                context.value,
                bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                width,
                height,
                bgr.strides[0],
                args.input_size,
                native.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                native.size,
            )
            if status:
                raise RuntimeError(dll.midas_last_error().decode())

            network_width, network_height = network_size(
                width, height, args.input_size
            )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
            rgb = cv2.resize(
                rgb,
                (network_width, network_height),
                interpolation=cv2.INTER_CUBIC,
            )
            rgb = (
                rgb - np.array([0.485, 0.456, 0.406])
            ) / np.array([0.229, 0.224, 0.225])
            tensor = torch.from_numpy(
                np.transpose(rgb, (2, 0, 1)).astype(np.float32)
            ).unsqueeze(0)
            with torch.inference_mode():
                reference = reference_model(tensor)
                reference = torch.nn.functional.interpolate(
                    reference.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                )[0, 0].numpy()

            difference = np.abs(native - reference)
            denominator = np.abs(reference, dtype=np.float64).sum()
            row = {
                "image": path.as_posix(),
                "width": width,
                "height": height,
                "network_width": network_width,
                "network_height": network_height,
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "relative_l1": float(
                    difference.astype(np.float64).sum() / denominator
                    if denominator
                    else 0.0
                ),
            }
            rows.append(row)
            print(
                f"[{index:02d}/{len(paths)}] {path.name}: "
                f"relative L1={row['relative_l1'] * 100:.6f}%, "
                f"max={row['max_abs']:.6g}",
                flush=True,
            )
    finally:
        context.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
