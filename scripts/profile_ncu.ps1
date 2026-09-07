# =============================================================================
# NVIDIA Nsight Compute (ncu) Profiling Script for FlashKernel-Engine
# Collects detailed hardware counters for Roofline, Memory Subsystem, & Warp Occupancy
# =============================================================================

param(
    [string]$Executable = "build/bin/bench_attention.exe",
    [string]$OutputReport = "reports/profile_attention"
)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Launching NVIDIA Nsight Compute Kernel Profiler..." -ForegroundColor Cyan
Write-Host " Executable: $Executable"
Write-Host " Output:     $OutputReport.ncu-rep"
Write-Host "================================================================="

# Create reports directory if missing
if (-not (Test-Path "reports")) {
    New-Item -ItemType Directory -Path "reports" | Out-Null
}

# Key Metrics to extract:
# 1. sm__throughput.avg.pct_of_peak_sustained_elapsed : SM Compute utilization
# 2. dram__throughput.avg.pct_of_peak_sustained_elapsed: HBM/DRAM bandwidth utilization
# 3. l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   : Coalesced global load transactions
# 4. l1tex__data_bank_conflicts_pipe_lsu.sum          : Shared memory bank conflicts (target = 0)
# 5. sm__warps_active.avg.pct_of_peak_sustained_active : Warp occupancy

$Metrics = @(
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum"
) -join ","

$Command = "ncu --metrics $Metrics --export $OutputReport --force-overwrite $Executable"

Write-Host "Executing command:" -ForegroundColor Yellow
Write-Host $Command

Invoke-Expression $Command

Write-Host "`nProfiling completed! Inspect $OutputReport.ncu-rep in NVIDIA Nsight Compute GUI." -ForegroundColor Green
