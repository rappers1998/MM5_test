set script_dir [file normalize [file dirname [info script]]]

cd $script_dir
open_project -reset laser_fusion_prj
set_top phase25_laser_register_fuse_ip_top

add_files [file join $script_dir laser_fusion.cpp]
add_files -tb [file join $script_dir tb_laser_fusion.cpp]

open_solution -reset "csim"

# Vitis HLS 2022.1 on this workstation has a fragile default GCC/MSYS path.
# Prefer the bundled clang frontend for C simulation when it is available.
set hls_root ""
if {[info exists ::env(RDI_APPROOT)]} {
    set hls_root [file normalize $::env(RDI_APPROOT)]
} elseif {[file exists "F:/Vivado/vivado2022.1/Vitis_HLS/2022.1/tps/win64/msys64/mingw64/bin/clang++.exe"]} {
    set hls_root "F:/Vivado/vivado2022.1/Vitis_HLS/2022.1"
}

if {$hls_root ne ""} {
    set clang_dir [file join $hls_root tps win64 msys64 mingw64 bin]
    set ::env(__USE_CLANG__) 1
    set ::env(AP_CLANG_PATH) $clang_dir
}

csim_design

close_project
exit
