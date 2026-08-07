param(
    [string]$UnityEditorRoot = "C:\Program Files\Unity\Hub\Editor\6000.3.9f1",
    [string]$DeviceSerial = "340YC10G7Y0X0N",
    [string]$ModelPath = "",
    [int]$Width = 256,
    [int]$Height = 128,
    [int]$Warmup = 3,
    [int]$Iterations = 20
)

$ErrorActionPreference = "Stop"
$nativeRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $nativeRoot `
        "out\foundation\midas_v21_small_256.midas"
}
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
if ([string]::IsNullOrWhiteSpace($env:VULKAN_SDK)) {
    throw "Set VULKAN_SDK to a host Vulkan SDK containing glslc"
}

$androidRoot = Join-Path $UnityEditorRoot `
    "Editor\Data\PlaybackEngines\AndroidPlayer"
$cmake = Join-Path $androidRoot "SDK\cmake\3.22.1\bin\cmake.exe"
$adb = Join-Path $androidRoot "SDK\platform-tools\adb.exe"
$ndk = Join-Path $androidRoot "NDK"
$buildRoot = Join-Path $nativeRoot ".BuildAndroid"
$ndkLink = Join-Path $buildRoot "unity-ndk"
$buildDirectory = Join-Path $buildRoot "quest-arm64"
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
if (-not (Test-Path -LiteralPath $ndkLink)) {
    New-Item -ItemType Junction -Path $ndkLink -Target $ndk | Out-Null
}

& $adb -s $DeviceSerial get-state | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Quest device $DeviceSerial is not available"
}
& $cmake -S $nativeRoot -B $buildDirectory -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$ndkLink\build\cmake\android.toolchain.cmake" `
    "-DANDROID_NDK=$ndkLink" `
    -DANDROID_ABI=arm64-v8a `
    -DANDROID_PLATFORM=android-29 `
    -DANDROID_STL=c++_static `
    -DCMAKE_BUILD_TYPE=Release `
    -DMIDAS_WITH_VULKAN=ON `
    -DBUILD_TESTING=OFF
if ($LASTEXITCODE -ne 0) {
    throw "MiDaS Android configure failed"
}
& $cmake --build $buildDirectory `
    --target midas_native midas_benchmark --parallel 8
if ($LASTEXITCODE -ne 0) {
    throw "MiDaS Android build failed"
}

$remoteRoot = "/data/local/tmp/midas-android-benchmark"
& $adb -s $DeviceSerial shell "rm -rf $remoteRoot && mkdir -p $remoteRoot"
& $adb -s $DeviceSerial push `
    (Join-Path $buildDirectory "midas_benchmark") `
    "$remoteRoot/benchmark" | Out-Null
& $adb -s $DeviceSerial push `
    (Join-Path $buildDirectory "libmidas_native.so") `
    "$remoteRoot/libmidas_native.so" | Out-Null
& $adb -s $DeviceSerial push $ModelPath "$remoteRoot/model.midas" | Out-Null
& $adb -s $DeviceSerial shell `
    "chmod 755 $remoteRoot/benchmark && cd $remoteRoot && export LD_LIBRARY_PATH=. && ./benchmark model.midas $Width $Height $Warmup $Iterations"
if ($LASTEXITCODE -ne 0) {
    throw "MiDaS Android benchmark failed"
}
