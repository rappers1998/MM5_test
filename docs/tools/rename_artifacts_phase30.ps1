param(
    [switch]$Apply,
    [string]$ManifestPath = "docs\manifests\rename_manifest_20260428.csv"
)

$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path -LiteralPath ".").Path
$artifactExts = @(
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".svg",
    ".docx", ".md", ".json", ".csv", ".npy", ".npz", ".txt", ".h"
)

$includeRoots = @(
    "runs",
    "darklight_mm5\outputs",
    "darklight_mm5\outputs_calibration_plane",
    "darklight_mm5\outputs_calibration_plane_opt",
    "darklight_mm5\outputs_calibration_plane_boundary",
    "darklight_mm5\calibration_only_method\outputs_phase25_depth_assisted",
    "darklight_mm5\teacher_residual_method\outputs",
    "mar_scholar_compare\analysis",
    "mar_scholar_compare\edge_detection_algorithms",
    "mm5_calib_benchmark\outputs",
    "peizhun_jiguang\generated"
)

function Normalize-Token {
    param([string]$Text)
    $s = $Text.ToLowerInvariant()
    $s = $s -replace "thermal", "th"
    $s = $s -replace "visible", "vis"
    $s = $s -replace "calibration", "cal"
    $s = $s -replace "registered", "reg"
    $s = $s -replace "registration", "reg"
    $s = $s -replace "optimized", "opt"
    $s = $s -replace "baseline", "base"
    $s = $s -replace "official", "off"
    $s = $s -replace "reference", "ref"
    $s = $s -replace "evaluation", "eval"
    $s = $s -replace "comparison", "cmp"
    $s = $s -replace "summary", "sum"
    $s = $s -replace "metrics", "met"
    $s = $s -replace "metric", "met"
    $s = $s -replace "acceptance", "acc"
    $s = $s -replace "engineered", "eng"
    $s = $s -replace "paper", "pap"
    $s = $s -replace "manual", "man"
    $s = $s -replace "teacher_residual", "tres"
    $s = $s -replace "sample_flow_upper_bound", "tsample"
    $s = $s -replace "global_flow", "tflow"
    $s = $s -replace "global_affine", "taff"
    $s = $s -replace "depth_guided", "dg"
    $s = $s -replace "selfcal", "self"
    $s = $s -replace "boundary", "bnd"
    $s = $s -replace "against", "vs"
    $s = $s -replace "source", "src"
    $s = $s -replace "visualization", "vis"
    $s = $s -replace "visual", "vis"
    $s = $s -replace "[^a-z0-9]+", "_"
    $s = $s.Trim("_")
    $s = $s -replace "_+", "_"
    if ($s.Length -gt 52) {
        $s = $s.Substring(0, 52).Trim("_")
    }
    return $s
}

function Get-FamilyTag {
    param([string]$RelPath)
    $r = $RelPath -replace "/", "\"

    if ($r -match "^runs\\") { return "run" }
    if ($r -match "^mar_scholar_compare\\") { return "mar" }
    if ($r -match "^peizhun_jiguang\\generated\\") { return "laser" }

    if ($r -match "darklight_mm5\\calibration_only_method\\outputs_phase25_depth_assisted") { return "dl_p25" }
    if ($r -match "darklight_mm5\\outputs_calibration_plane_boundary") { return "dl_bnd" }
    if ($r -match "darklight_mm5\\outputs_calibration_plane_opt") { return "dl_opt" }
    if ($r -match "darklight_mm5\\outputs_calibration_plane") { return "dl_plane" }
    if ($r -match "darklight_mm5\\outputs\\") { return "dl_ref" }
    if ($r -match "teacher_residual_method\\outputs\\global_affine") { return "dl_taff" }
    if ($r -match "teacher_residual_method\\outputs\\global_flow") { return "dl_tflow" }
    if ($r -match "teacher_residual_method\\outputs\\sample_flow_upper_bound") { return "dl_tsample" }
    if ($r -match "teacher_residual_method\\outputs\\diagnostics") { return "dl_tdiag" }

    if ($r -match "legacy_mar_scene282_reproduced\\engineered_best") { return "legacy_eng" }
    if ($r -match "legacy_mar_scene282_reproduced\\paper_final") { return "legacy_pap" }
    if ($r -match "scene_282_3_comparison") { return "cmp282" }
    if ($r -match "method_M([0-9]+)_[^\\]*_thermal") { return "m$($Matches[1])th" }
    if ($r -match "method_M([0-9]+)_[^\\]*_uv") { return "m$($Matches[1])uv" }

    return "art"
}

function Get-SampleTag {
    param([string]$RelPath, [string]$BaseName)
    $text = "$RelPath\$BaseName"
    if ($text -match "([0-9]{3})_seq[0-9]+") { return "s$($Matches[1])" }
    if ($text -match "scene[_-]?([0-9]{3})") { return "s$($Matches[1])" }
    if ($BaseName -match "^([0-9]{3})(?:_|$)") { return "s$($Matches[1])" }
    return ""
}

function Get-ContentTag {
    param([string]$RelPath, [string]$BaseName, [string]$Ext)
    $b = $BaseName.ToLowerInvariant()
    $r = $RelPath.ToLowerInvariant()

    $exact = @{
        "manual_gt_thermal" = "gt_th"
        "manual_gt_thermal_id" = "gt_id"
        "manual_gt_thermal_overlay" = "gt_ov"
        "manual_paper_engineered_metrics" = "met_gt_pap_eng"
        "mar_final_acceptance_vs_paper" = "s282_eng_vs_pap"
        "mar_full_story_scene282_20260420" = "s282_story_20260420"
        "mar_full_story_scene282_20260420_grid" = "s282_story_grid_20260420"
        "mar_scholar_comparison_scene282_20260421" = "s282_scholar_20260421"
        "mar_scholar_comparison_scene282_paper_style_20260421" = "s282_scholar_pap_20260421"
        "mar_流程全链路说明_四宫格版_scene282_20260420" = "s282_flow_grid_cn_20260420"
        "mm5_校准算法总结_中文版_20260421" = "mm5_cal_sum_cn_20260421"
        "inverse_map_filled" = "inv_fill"
        "inverse_map_raw" = "inv_raw"
        "inverse_valid_mask" = "inv_valid"
        "label_source_vis" = "label_src"
        "laser_lut" = "lut"
    }
    if ($exact.ContainsKey($b)) { return $exact[$b] }

    if ($b -match "^mar_.*scene282_20260420$") { return "s282_flow_grid_cn_20260420" }
    if ($b -match "^mm5_.*20260421$") { return "mm5_cal_sum_cn_20260421" }

    if ($b -match "phase24_affine_lmeds_baseline") { return "p24_base" }
    if ($b -match "depth_registered_global_shift_depth_fill") { return "p25_regfill" }
    if ($b -match "depth_registered_global_shift") { return "p25_reg" }
    if ($b -match "depth_project_raw_lwir") { return "p25_proj" }
    if ($b -match "depth_holefill_keep_phase24") { return "p25_keep" }
    if ($b -match "depth_holefill_union") { return "p25_union" }
    if ($b -match "median_holefill_control") { return "p25_med" }

    if ($b -match "error_map") { return "err" }
    if ($b -match "gt_mask") { return "gt" }
    if ($b -match "pred_mask") { return "pred" }
    if ($b -match "contours") { return "cont" }
    if ($b -match "heatmap") { return "heat" }
    if ($b -match "warped") { return "warp" }
    if ($b -match "quad") { return "quad" }
    if ($b -match "five_panel") { return "p5" }
    if ($b -match "evaluation_panel") { return "eval_panel" }
    if ($b -match "edge_review") { return "edge_rev" }
    if ($b -match "official_check") { return "off_chk" }
    if ($b -match "lwir_calibration_check") { return "lwir_chk" }
    if ($b -match "edge_overlay") { return "edge" }
    if ($b -match "overlay") { return "ov" }
    if ($b -match "fused_heat") { return "fheat" }
    if ($b -match "fused_intensity") { return "fgray" }
    if ($b -match "fusion_alpha_mask") { return "alpha" }
    if ($b -match "lwir_calibration_plane_initial_to_rgb1_raw") { return "lwir_init" }
    if ($b -match "lwir_calibrated_to_rgb1_raw") { return "lwir_reg" }
    if ($b -match "rgb1_raw_display_enhanced") { return "rgb1_enh" }
    if ($b -match "rgb1_raw_to_official_rgb1") { return "rgb1_off" }
    if ($b -match "rgb1_raw") { return "rgb1" }
    if ($b -match "rgb3_raw_bright_reference") { return "rgb3_ref" }
    if ($b -match "lwir_raw_to_official_lwir") { return "lwir_off" }
    if ($b -match "lwir_raw_norm") { return "lwir" }
    if ($b -match "lwir_official_absdiff") { return "lwir_diff" }
    if ($b -match "lwir_official_edge_check") { return "lwir_edge" }
    if ($b -match "valid_mask_calibration_plane") { return "valid" }
    if ($b -match "valid_mask_teacher_residual") { return "valid" }
    if ($b -match "valid_mask_sample_flow") { return "valid" }
    if ($b -match "teacher_lwir_reference") { return "teacher" }
    if ($b -match "baseline_lwir_calibration_plane_opt") { return "base" }
    if ($b -match "lwir_teacher_residual_to_rgb1_raw") { return "tres_lwir" }
    if ($b -match "lwir_sample_flow_to_rgb1_raw") { return "sflow_lwir" }
    if ($b -match "h_raw_lwir_to_aligned_lwir") { return "h_lwir2al" }
    if ($b -match "h_raw_lwir_to_raw_rgb1") { return "h_lwir2rgb" }
    if ($b -match "h_raw_rgb1_to_aligned_rgb1") { return "h_rgb2al" }
    if ($b -match "h_lwir_to_rgb_plane") { return "h_lwir2rgb" }
    if ($b -match "h_rgb_to_lwir_plane") { return "h_rgb2lwir" }
    if ($b -match "transform_info") { return "tf" }
    if ($b -match "selected_dark_samples") { return "samples" }

    if ($b -eq "fusion_metrics") { return "met_fusion" }
    if ($b -eq "fusion_summary") { return "sum_fusion" }
    if ($b -eq "per_sample") { return "met_sample" }
    if ($b -eq "registration_stages") { return "met_reg_stage" }
    if ($b -eq "registration_summary") { return "sum_reg" }
    if ($b -eq "summary") { return "sum" }
    if ($b -eq "evaluation_against_reference") { return "eval_ref" }
    if ($b -eq "evaluation_summary") { return "eval_sum" }
    if ($b -eq "evaluation_report") { return "eval_report" }
    if ($b -eq "optimization_summary") { return "opt_sum" }
    if ($b -eq "candidate_variants") { return "variants" }
    if ($b -eq "per_sample_residuals") { return "residuals" }
    if ($b -eq "boundary_metrics_baseline") { return "bnd_base" }
    if ($b -eq "boundary_metrics_optimized") { return "bnd_opt" }
    if ($b -eq "optimization_candidates_coarse") { return "opt_coarse" }
    if ($b -eq "optimization_candidates_residual") { return "opt_resid" }
    if ($b -eq "phase25_board_correspondences") { return "board_pts" }
    if ($b -eq "phase25_board_transforms") { return "board_tf" }
    if ($b -eq "phase25_depth_assisted_metrics") { return "met_p25" }
    if ($b -eq "phase25_depth_assisted_report") { return "report_p25" }
    if ($b -eq "phase25_depth_assisted_summary") { return "sum_p25" }
    if ($b -eq "phase25_depth_registration_scores") { return "score_p25" }
    if ($b -match "figure_guide") { return "fig_guide" }
    if ($b -match "mar_history_note") { return "mar_note" }
    if ($b -match "metric_explanation") { return "met_note" }
    if ($b -match "thermal_per_scene_normalized_error_note") { return "th_norm_note" }

    $stageBase = $b -replace "^[0-9]{3}_", ""
    if ($stageBase -match "^([0-9]{2}[a-z]?)_(.+)$") {
        return "st$($Matches[1])_" + (Normalize-Token $Matches[2])
    }

    return Normalize-Token $BaseName
}

function New-ArtifactName {
    param([System.IO.FileInfo]$File)
    $rel = $File.FullName.Substring($workspace.Length + 1)
    $base = [System.IO.Path]::GetFileNameWithoutExtension($File.Name)
    $ext = $File.Extension.ToLowerInvariant()
    $family = Get-FamilyTag $rel
    $sample = Get-SampleTag $rel $base
    $content = Get-ContentTag $rel $base $ext

    $parts = @($family)
    if ($sample -and ($content -notmatch "^s[0-9]{3}(?:_|$)")) { $parts += $sample }
    if ($content -and $content -ne $family) { $parts += $content }

    $name = (($parts -join "_") -replace "_+", "_").Trim("_") + $ext
    if ($name.Length -gt 78) {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($name)
        $name = $stem.Substring(0, [Math]::Min(72, $stem.Length)).Trim("_") + $ext
    }
    return $name
}

$roots = foreach ($root in $includeRoots) {
    $full = Join-Path $workspace $root
    if (Test-Path -LiteralPath $full) {
        (Resolve-Path -LiteralPath $full).Path
    }
}

$files = foreach ($root in $roots) {
    Get-ChildItem -LiteralPath $root -Recurse -File -Force |
        Where-Object {
            $relForFilter = $_.FullName.Substring($workspace.Length + 1)
            ($artifactExts -contains $_.Extension.ToLowerInvariant()) -and
            ($relForFilter -notmatch "^mm5_calib_benchmark\\outputs\\mm5_benchmark\\splits\\")
        }
}

$files = $files | Sort-Object FullName -Unique
$targetUse = @{}
$records = New-Object System.Collections.Generic.List[object]

foreach ($file in $files) {
    $rootOk = $false
    foreach ($root in $roots) {
        if ($file.FullName.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
            $rootOk = $true
            break
        }
    }
    if (-not $rootOk) { continue }

    $dir = $file.DirectoryName
    $newName = New-ArtifactName $file
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($newName)
    $ext = [System.IO.Path]::GetExtension($newName)
    $candidate = Join-Path $dir $newName
    $i = 2
    while ($targetUse.ContainsKey($candidate.ToLowerInvariant()) -or ((Test-Path -LiteralPath $candidate) -and ($candidate -ne $file.FullName))) {
        $newName = "{0}_{1:D2}{2}" -f $stem, $i, $ext
        $candidate = Join-Path $dir $newName
        $i++
    }
    $targetUse[$candidate.ToLowerInvariant()] = $file.FullName

    if ($candidate -eq $file.FullName) { continue }

    $oldRel = $file.FullName.Substring($workspace.Length + 1)
    $newRel = $candidate.Substring($workspace.Length + 1)
    $records.Add([pscustomobject]@{
        action = "rename"
        old_path = $oldRel
        new_path = $newRel
        old_name = $file.Name
        new_name = $newName
    })
}

$manifestFull = Join-Path $workspace $ManifestPath
$records | Export-Csv -LiteralPath $manifestFull -NoTypeInformation -Encoding UTF8

Write-Host ("Planned artifact renames: {0}" -f $records.Count)
Write-Host ("Manifest: {0}" -f $manifestFull)

if ($Apply) {
    foreach ($record in $records) {
        $src = Join-Path $workspace $record.old_path
        $dst = Join-Path $workspace $record.new_path
        $srcResolvedParent = (Resolve-Path -LiteralPath (Split-Path -Parent $src)).Path
        $dstParent = Split-Path -Parent $dst
        if (-not (Test-Path -LiteralPath $dstParent)) {
            throw "Missing destination directory: $dstParent"
        }
        $dstResolvedParent = (Resolve-Path -LiteralPath $dstParent).Path
        if (-not $srcResolvedParent.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing source outside workspace: $src"
        }
        if (-not $dstResolvedParent.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing destination outside workspace: $dst"
        }
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Source missing before rename: $src"
        }
        if (Test-Path -LiteralPath $dst) {
            throw "Destination already exists before rename: $dst"
        }
        Move-Item -LiteralPath $src -Destination $dst
    }
    Write-Host "Applied artifact renames."
}
