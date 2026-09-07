import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect source files
include_dirs = [
    os.path.abspath("include"),
]

sources = [
    "python/bindings.cpp",
    "src/flash_attention.cu",
    "src/fused_rmsnorm.cu",
    "src/fused_activations.cu",
    "src/fused_rope.cu",
    "src/tensor_core_attention.cu",
    "src/cuda_graph_runner.cu",
    "src/flash_engine_api.cpp",
]

# CUDA architecture flags (Ampere, Ada, Hopper, Volta, Turing)
nvcc_args = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
]

cxx_args = ["-O3", "-std=c++17"]

setup(
    name="flash_engine",
    version="1.0.0",
    author="Mohit Gupta",
    author_email="mohitgupta.nitk@gmail.com",
    description="Production-Grade High-Performance CUDA & WebGPU Kernel Optimization Engine for Transformers",
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/mohitgupta90/flash-kernel-engine",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="flash_engine_cuda",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
