"""Compare the dependency-free native graph with pinned PyTorch CPU."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


def load_reference(
    repository: Path,
    geffnet_source: Path,
    checkpoint: Path,
):
    sys.path[:0] = [str(geffnet_source), str(repository)]

    # MiDaS imports optional timm backbones eagerly. Small uses none of them.
    timm = types.ModuleType("timm")
    timm.create_model = lambda *args, **kwargs: None
    timm_models = types.ModuleType("timm.models")
    timm_beit = types.ModuleType("timm.models.beit")
    timm_beit.gen_relative_position_index = lambda *args, **kwargs: None
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = timm_models
    sys.modules["timm.models.beit"] = timm_beit

    from geffnet import tf_efficientnet_lite3

    torch.hub.load = lambda *args, **kwargs: tf_efficientnet_lite3(
        pretrained=False, exportable=True
    )
    from midas.midas_net_custom import MidasNet_small

    model = MidasNet_small(
        None,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True,
        blocks={"expand": True},
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--geffnet-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--backend", choices=("cpu", "vulkan"), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.size <= 0 or args.size % 32:
        parser.error("--size must be a positive multiple of 32")

    reference_model = load_reference(
        args.repository.resolve(),
        args.geffnet_source.resolve(),
        args.checkpoint.resolve(),
    )
    generator = np.random.default_rng(args.seed)
    input_tensor = generator.normal(
        0.0, 1.0, (3, args.size, args.size)
    ).astype(np.float32)
    with torch.inference_mode():
        reference = reference_model(
            torch.from_numpy(input_tensor).unsqueeze(0)
        )[0].numpy()

    dll = ctypes.CDLL(str(args.dll.resolve()))
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
    dll.midas_infer_tensor_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    dll.midas_infer_tensor_f32.restype = ctypes.c_int
    dll.midas_last_error.restype = ctypes.c_char_p

    context = ctypes.c_void_p()
    status = (
        dll.midas_create_vulkan(
            str(args.model.resolve()).encode(),
            0,
            args.device,
            ctypes.byref(context),
        )
        if args.backend == "vulkan"
        else dll.midas_create(
            str(args.model.resolve()).encode(),
            0,
            ctypes.byref(context),
        )
    )
    if status:
        raise RuntimeError(dll.midas_last_error().decode())
    native = np.empty((args.size, args.size), dtype=np.float32)
    try:
        status = dll.midas_infer_tensor_f32(
            context,
            input_tensor.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            ),
            args.size,
            args.size,
            native.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            native.size,
        )
        if status:
            raise RuntimeError(dll.midas_last_error().decode())
    finally:
        dll.midas_destroy(context)

    difference = np.abs(native - reference)
    denominator = np.abs(reference, dtype=np.float64).sum()
    result = {
        "size": args.size,
        "seed": args.seed,
        "backend": args.backend,
        "device": args.device if args.backend == "vulkan" else None,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "relative_l1": float(
            difference.astype(np.float64).sum() / denominator
            if denominator
            else 0.0
        ),
        "native_min": float(native.min()),
        "native_max": float(native.max()),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
