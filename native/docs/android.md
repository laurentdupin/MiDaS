# Android and Meta Quest native inference

MiDaS v2.1 Small builds as a dependency-free Android ARM64 shared library and
exports canonical InferBridge harness ABI 2. The Android HOST path queues its
Vulkan inference on a persistent worker rather than blocking the session
caller. It accepts BGRA8 and RGBA8, preserves the established MiDaS worker
semantics, admits at most three jobs, and keeps a running cancelled job
nonterminal until Vulkan has stopped referencing the borrowed bindings.

Build and run the persistent tensor benchmark with:

```powershell
$env:VULKAN_SDK = "<host Vulkan SDK>"
native/tools/android/run_quest_benchmark.ps1
```

The resulting library is
`native/.BuildAndroid/quest-arm64/libmidas_native.so`. On the Quest 3S Adreno
740, 256x128 tensor inference including upload and readback measured about
165 ms (6.1 FPS). The full InferBridge HOST path from a 1920x1080 BGRA input
through source-sized float output measured 178.056 ms (5.616 FPS).

The dynamic ABI probe passed correlated repeated output, sub-0.02-ms submit,
cancellation, shutdown, and `dlclose`. Exact evidence is recorded in
`android_quest3s_benchmark_2026-08-07.json`.

The Android harness truthfully advertises HOST memory only. It does not claim
AHardwareBuffer or direct Vulkan texture transfer support.
