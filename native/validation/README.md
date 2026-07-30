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

## Remaining before model completion

GPU capability is not advertised yet. The next gates are exact BGR8
preprocessing and source-size bicubic output resize, followed by Vulkan
implementations of the now-proven operators and external-resource integration.
