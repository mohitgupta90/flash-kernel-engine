# Security Policy

## Supported Versions

FlashKernel-Engine releases receiving active security and performance maintenance:

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security and integrity of our codebase seriously. If you identify a vulnerability (such as out-of-bounds GPU global memory access, buffer overflows in host dispatch wrappers, or device hangs caused by uncontrolled thread grid sizes):

1. **Do not create a public issue.**
2. Send a detailed report via email to **mohitgupta.nitk@gmail.com** with:
   - Kernel module affected (e.g. `src/paged_attention.cu`)
   - Description of the memory vulnerability or invalid pointer dereference
   - Minimal reproducible test script or CUDA launch configuration
   - Hardware platform and CUDA driver version

## Vulnerability Response Process

- **Acknowledgment**: You will receive an initial response within 48 hours confirming receipt.
- **Triage & Patching**: A triage evaluation will occur within 5 business days, followed by a patched release candidate.
- **Disclosure**: Public CVE disclosure and changelog attribution will be coordinated once the patch is verified and merged.
