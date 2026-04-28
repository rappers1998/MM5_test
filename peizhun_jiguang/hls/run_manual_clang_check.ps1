param(
    [string]$HlsRoot = "F:\Vivado\vivado2022.1\Vitis_HLS\2022.1",
    [switch]$NoRun
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ScriptDir "manual_build"
$Exe = Join-Path $BuildDir "tb_laser_fusion_clang.exe"
$Clang = Join-Path $HlsRoot "tps\win64\msys64\mingw64\bin\clang++.exe"

if (-not (Test-Path -LiteralPath $Clang)) {
    throw "clang++.exe not found: $Clang"
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$RootSlash = $HlsRoot -replace "\\", "/"
$Tool = "$RootSlash/win64/tools"
$Tech = "$RootSlash/common/technology"

$Args = @(
    "-std=c++11",
    "-Wno-unknown-pragmas",
    "-Wno-macro-redefined",
    "-g",
    "-DNT",
    "-D__VITIS_HLS__",
    "-D__SIM_FPO__",
    "-D__SIM_FFT__",
    "-D__SIM_FIR__",
    "-D__SIM_DDS__",
    "-DAESL_TB",
    "-D__xilinx_ip_top=",
    "-I", "$Tool/systemc/include",
    "-I", "$RootSlash/include",
    "-I", "$RootSlash/include/ap_sysc",
    "-I", "$Tech/generic/SystemC",
    "-I", "$Tech/generic/SystemC/AESL_FP_comp",
    "-I", "$Tech/generic/SystemC/AESL_comp",
    "-I", "$Tool/auto_cc/include",
    (Join-Path $ScriptDir "tb_laser_fusion.cpp"),
    (Join-Path $ScriptDir "laser_fusion.cpp"),
    "-o", $Exe
)

& $Clang @Args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $NoRun) {
    & $Exe
    exit $LASTEXITCODE
}
