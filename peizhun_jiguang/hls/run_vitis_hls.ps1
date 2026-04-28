param(
    [string]$HlsBin = "F:\Vivado\vivado2022.1\Vitis_HLS\2022.1\bin",
    [string]$VivadoRoot = "F:\Vivado\vivado2022.1\Vivado\2022.1",
    [switch]$SkipSettings,
    [switch]$UseWrapper
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Tcl = Join-Path $ScriptDir "run_hls.tcl"
$Loader = Join-Path $HlsBin "loader.bat"
$Wrapper = Join-Path $HlsBin "vitis_hls.bat"
$Settings = Join-Path (Split-Path -Parent $HlsBin) "settings64.bat"

if (Test-Path -LiteralPath $VivadoRoot) {
    $env:XILINX_VIVADO = $VivadoRoot
    $env:Path = (Join-Path $VivadoRoot "bin") + ";" + $env:Path
}
$env:RDI_DEPENDENCY = "VITIS_HLS_SETUP"

if ($UseWrapper) {
    if (-not (Test-Path -LiteralPath $Wrapper)) {
        throw "Vitis HLS wrapper not found: $Wrapper"
    }
    if ((-not $SkipSettings) -and (Test-Path -LiteralPath $Settings)) {
        & cmd.exe /d /s /c "call `"$Settings`" && `"$Wrapper`" -f `"$Tcl`""
    } else {
        & $Wrapper -f $Tcl
    }
} else {
    if (-not (Test-Path -LiteralPath $Loader)) {
        throw "Vitis HLS loader not found: $Loader"
    }
    if ((-not $SkipSettings) -and (Test-Path -LiteralPath $Settings)) {
        & cmd.exe /d /s /c "call `"$Settings`" && `"$Loader`" -exec vitis_hls -f `"$Tcl`""
    } else {
        & $Loader -exec vitis_hls -f $Tcl
    }
}

exit $LASTEXITCODE
