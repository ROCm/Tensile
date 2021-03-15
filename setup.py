################################################################################
# Install Tensile
# - installs python scripts, c++ source files and a few configs
# - creates executables for running benchmarking
# - installs TensileConfig.cmake so one call find_package(Tensile)
################################################################################
from setuptools import setup
import os.path
import re

def readRequirementsFromTxt():
  requirements = []
  with open("requirements.txt") as req_file:
    for line in req_file.read().splitlines():
      if not line.strip().startswith("#"):
        requirements.append(line)
  return requirements

def readVersionFromInit():
    import Tensile
    return Tensile.__version__

setup(
  name="Tensile",
  version=readVersionFromInit(),
  description="An auto-tuning tool for GEMMs and higher-dimensional tensor contractions on GPUs.",
  url="https://github.com/JeremyAdamHart/Tensile",
  author="Advanced Micro Devices",
  license="MIT",
  install_requires=readRequirementsFromTxt(),
  python_requires='>=3.5',
  packages=["Tensile"],
  package_data={ "Tensile": ["Tensile/cmake/*"] },
  data_files=[ ("cmake", ["Tensile/cmake/TensileConfig.cmake", "Tensile/cmake/TensileConfigVersion.cmake"]) ],
  include_package_data=True,
  entry_points={"console_scripts": [
    # user runs a benchmark
    "Tensile = Tensile.Tensile:main",
    "tensileBenchmarkLibraryClient = Tensile.TensileBenchmarkLibraryClient:main",
    # CMake calls this to create Tensile.lib
    "TensileCreateLibrary = Tensile.TensileCreateLibrary:TensileCreateLibrary",

    "TensileGetPath = Tensile:PrintTensileRoot",
    # automatic benchmarking for rocblas
    "tensile_rocblas_sgemm = Tensile.Tensile:TensileROCBLASSGEMM",
    "tensile_rocblas_dgemm = Tensile.Tensile:TensileROCBLASDGEMM",
    "tensile_rocblas_cgemm = Tensile.Tensile:TensileROCBLASCGEMM",
    "tensile_rocblas_zgemm = Tensile.Tensile:TensileROCBLASZGEMM",
    # automatically find fastest sgemm exhaustive search
    "tensile_sgemm = Tensile.Tensile:TensileSGEMM5760",
    # Run tensile benchmark from cluster
    "TensileBenchmarkCluster = Tensile.TensileBenchmarkCluster:main",
    # Retune library logic file
    "TensileRetuneLibrary = Tensile.TensileRetuneLibrary:main"
    ]}
  )
