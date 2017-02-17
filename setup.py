################################################################################
# Install Tensile
# - installs python scripts, c++ source files and a few configs
# - creates executables for running benchmarking
# - installs TensileConfig.cmake so one call find_package(Tensile)
################################################################################
from setuptools import setup
import sys
import os.path

setup(
  name="Tensile",
  version="2.0",
  description="An auto-tuning tool for GEMMs and higher-dimensional tensor contractions on GPUs.",
  url="https://github.com/RadeonOpenCompute/Tensile",
  author="Advanced Micro Devices",
  license="MIT",
  install_requires=["pyyaml"],
  packages=["Tensile"],
  data_files=[
    ("Source", [
      "Source/BenchmarkClient.cmake",
      "Source/CMakeLists.txt",
      "Source/FindHCC.cmake",
      "Source/MathTemplates.cpp",
      "Source/SetupTeardown.cpp",
      "Source/TensileTypes.h",
      "Source/Client.cpp",
      "Source/FindHIP.cmake",
      "Source/MathTemplates.h",
      "Source/SolutionHelper.cpp",
      "Source/Tools.cpp",
      "Source/Client.h",
      "Source/EnableWarnings.cmake",
      "Source/FindOpenCL.cmake",
      "Source/ReferenceCPU.h",
      "Source/SolutionHelper.h",
      "Source/Tools.h",
      ] ),
    ("Configs", [
      "Configs/jenkins_sgemm_defaults.yaml",
      "Configs/jenkins_dgemm_defaults.yaml",
      "Configs/rocblas_sgemm.yaml",
      "Configs/rocblas_dgemm.yaml",
      "Configs/rocblas_cgemm.yaml",
      "Configs/sgemm_5760.yaml",
      ] ),
    (os.path.join(sys.exec_prefix, "cmake"), ["Source/TensileConfig.cmake"]),
    ], 
  include_package_data=True,
  entry_points={"console_scripts": [
    # user runs a benchmark
    "tensile = Tensile.Tensile:main",
    # CMake calls this to create Tensile.lib
    "TensileCreateLibrary = Tensile.TensileCreateLibrary:TensileCreateLibrary"
    # automatic benchmarking for rocblas
    "tensile_rocblas_sgemm = Tensile.Tensile:TensileROCBLASSGEMM",
    "tensile_rocblas_dgemm = Tensile.Tensile:TensileROCBLASDGEMM",
    "tensile_rocblas_cgemm = Tensile.Tensile:TensileROCBLASCGEMM",
    "tensile_rocblas_zgemm = Tensile.Tensile:TensileROCBLASZGEMM",
    # automatically find fastest sgemm exhaustive search
    "tensile_sgemm = Tensile.Tensile:TensileSGEMM5760",
    ]}
  )
