## Description
<!--- Describe your changes in detail -->

## Motivation & Context
<!--- Why is this change required? What problem does it solve? -->
<!--- If it fixes an open issue, please link to the issue here. -->

## Changes Checklist
- [ ] **Kernel Design**: Follows CUDA memory coalescing and avoids shared memory bank conflicts.
- [ ] **Device Safety**: Kernel device code is free of host-only runtime calls and exceptions.
- [ ] **Architecture Compatibility**: Tested or guarded for compute capabilities `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89`, and `sm_90`.
- [ ] **Numerical Verification**: Unit tests added in `tests/` or `python/tests/` confirming numerical tolerance against FP32/FP64 ground truth.
- [ ] **Benchmarking**: Profiling metrics or benchmark throughput numbers included below.
- [ ] **Documentation**: Updated `README.md` and module docstrings where appropriate.

## Benchmark & Profiling Results
```
<!-- Paste benchmark terminal output or ncu profiling metrics here -->
```
