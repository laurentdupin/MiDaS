# MiDaS Small native validation

The correctness-first port targets the catalog's `midas_v21_small_256`
variant at source revision
`454597711a62eabcbf7d1e89f3fb9f569051ac9b`.

## Canonical model boundary

- Canonical file: `midas_v21_small_256.pt`
- Canonical SHA-256:
  `70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca`
- Canonical bytes: 85,761,505
- Derived format: `MIDAS1` version 1
- Converter: `midas-export-pytorch-weights-v1`
- Derived tensors: 410 FP32 inference tensors
- Derived bytes: 85,659,136

The development-only converter uses PyTorch's restricted `weights_only=True`
loader, discards only BatchNorm training counters, and rejects non-FP32
inference tensors, unsupported ranks, duplicate names, invalid lengths, and
overlong names. Deployment never parses pickle. The `.midas` output records
the canonical SHA, converter, format, and model kind for a hidden
content-addressed cache.

The C model reader and public ABI build without Vulkan or an inference
framework. Both ABI and bounded-model metadata tests pass, and the real
converted checkpoint passes the native model probe.

## Scalar FP32 graph gate

The dependency-free DLL now executes the complete EfficientNet-Lite3 and
MiDaS refinement graph from normalized RGB CHW FP32 tensors. It implements
standard and depthwise 3x3/5x5 convolution, stride-2 TensorFlow SAME padding,
BatchNorm inference, ReLU6, residual add, align-corners bilinear refinement,
and the final output head.

Deterministic full-graph comparisons against PyTorch CPU:

| Input | Relative L1 | Maximum absolute error |
|---:|---:|---:|
| 32x32 | 0.0000175% | 0.000275 |
| 64x64 | 0.0000375% | 0.000732 |
| 256x256 | 0.0000700% | 0.001556 |

This establishes a very tight native correctness oracle for the Vulkan port.
The 256x256 comparison, including Python model construction and both
executions, completed in 11.6 seconds; performance is intentionally not yet
an acceptance criterion.

## Full image-path gate

The complete `midas_infer_bgr8` path was compared on all 22 Depth Anything V2
asset images at the official input bound of 256. It includes aspect-preserving
upper-bound shape selection, multiple-of-32 rounding, OpenCV-compatible cubic
BGR-to-normalized-RGB preprocessing, the full graph, and PyTorch-compatible
bicubic resize to each source resolution.

| Metric | Result |
|---|---:|
| Images | 22 |
| Minimum relative L1 | 0.0000344% |
| Median relative L1 | 0.0000617% |
| Mean relative L1 | 0.0000620% |
| Maximum relative L1 | 0.0001015% |
| Maximum absolute error | 0.009735 |

Every image is far below the 1% requirement. Detailed evidence is stored in
`assets-256.csv`.

## Vulkan graph gate

The dependency-free DLL also executes the complete network graph through
Vulkan compute. The model weights remain resident in device-local buffers for
the context lifetime. Deterministic tensor comparisons against PyTorch CPU on
the Radeon RX 9070:

| Input | Relative L1 | Maximum absolute error |
|---:|---:|---:|
| 32x32 | 0.0000157% | 0.000244 |
| 64x64 | 0.0000178% | 0.000488 |
| 256x256 | 0.0000571% | 0.001221 |

The same Vulkan context was reused for all 22 image-path cases:

| Metric | Result |
|---|---:|
| Images | 22 |
| Minimum relative L1 | 0.0000349% |
| Median relative L1 | 0.0000708% |
| Mean relative L1 | 0.0000679% |
| Maximum relative L1 | 0.0001146% |
| Maximum absolute error | 0.011597 |

Detailed evidence is stored in `assets-256-vulkan.csv`.

The capability probe truthfully advertises `VULKAN_GRAPH`,
`HOST_TENSOR_UPLOAD`, and `HOST_DEPTH_READBACK`, with one synchronous
in-flight inference. Image preprocessing and final bicubic resizing currently
run on the CPU, the normalized tensor is uploaded, and depth is read back.
There is no claim of an external-resource or zero-copy path. Vulkan work uses
fences; the implementation contains no `vkQueueWaitIdle`.

## Remaining before end-to-end GPU residency

The stable, accurate native CPU and Vulkan tensor graphs are complete.
Direct external-image preprocessing, GPU-resident leased depth output,
explicit producer/consumer synchronization, and a bounded asynchronous job
pool remain future integration work.

## Exact InferBridge worker boundary

The original `midas_infer_bgr8` API intentionally implements the official
MiDaS RGB image path and source-size bicubic output. InferBridge's Python
template differs: it passes the capture's first three BGR bytes as model
channels and returns min/max-normalized uint8 depth at the network dimensions.

ABI 2 therefore adds, without changing existing calls,
`midas_inferbridge_bgra8_u8`. It accepts strided BGRA8, preserves that channel
quirk, uses the exact MiDaS Small transform/network shape, and performs the
worker's FP32 normalization and truncating uint8 conversion. The existing
official and tensor APIs retain their prior behavior.

The exact deployed contract was compared on all 22 Depth Anything V2 assets
against PyTorch CPU on the Radeon RX 9070, GTX 1080, and RX 6700 XT. Every
output has the correct network dimensions and differs by at most one uint8
level on every adapter. `native/tools/compare_assets.py --contract
inferbridge` reproduces this gate.

## Embedded InferBridge harness

The DLL also exports `ibrh_get_api` for InferBridge harness ABI 1.0. The
single catalog MiDaS Small entry selects the hidden content-addressed
`.midas` derivation of its shared canonical `.pt`, accepts host-memory BGRA8,
and returns a leased host-memory `DEPTH_UNORM8` image at the network shape.
`source_frame_id` and timestamp are preserved, and the lease remains valid
after job release.

Capability reporting advertises host input/output and one synchronous
in-flight job only. The selected Vulkan device executes the graph, while
capture preprocessing and output readback/quantization remain host
boundaries. External GPU resources, async execution, and cancellation are not
advertised. The Windows Release ABI, image, metadata, and real full-graph
harness tests all pass.
