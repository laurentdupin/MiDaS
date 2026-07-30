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

The C model reader and public ABI foundation build without Vulkan or an
inference framework. Both ABI and bounded-model metadata tests pass, and the
real converted checkpoint passes the native model probe.

## Graph work remaining

No inference or GPU capability is advertised yet. The exact pinned graph is
EfficientNet-Lite3 plus the MiDaS refinement head. New correctness operators
are depthwise 3x3/5x5 convolution, stride-2 TensorFlow SAME padding,
BatchNorm inference (eligible for converter-side folding after parity),
ReLU6, residual add, and the existing bilinear/refinement convolutions.
These operators and stage outputs must pass PyTorch CPU comparison before the
lifecycle and GPU-resource ABI become available.
