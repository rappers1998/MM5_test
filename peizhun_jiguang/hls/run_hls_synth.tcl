set script_dir [file normalize [file dirname [info script]]]

cd $script_dir
open_project -reset laser_fusion_synth_prj
set_top phase25_laser_register_fuse_ip_top

add_files [file join $script_dir laser_fusion.cpp]

open_solution -reset "synth"

set target_part "xczu15eg-ffvb1156-2-e"
if {[info exists ::env(HLS_PART)] && $::env(HLS_PART) ne ""} {
    set target_part $::env(HLS_PART)
}

set clock_period_ns 10.0
if {[info exists ::env(HLS_CLOCK_PERIOD_NS)] && $::env(HLS_CLOCK_PERIOD_NS) ne ""} {
    set clock_period_ns $::env(HLS_CLOCK_PERIOD_NS)
}

set_part $target_part
create_clock -period $clock_period_ns -name default

csynth_design

close_project
exit
