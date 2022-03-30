# Change Log for Tensile

## (Unreleased) Tensile 4.33.0
### Added
- TensileUpdateLibrary for updating old library logic files
- Support for TensileRetuneLibrary to use sizes from separate file
- ZGEMM DirectToVgpr/DirectToLds/StoreCInUnroll/MIArchVgpr support
- Tests for denorm correctness
- Option to write different architectures to different TensileLibrary files
### Optimizations
- Optimize MessagePackLoadLibraryFile by switching to fread
- DGEMM tail loop optimization for PrefetchAcrossPersistentMode=1/DirectToVgpr
### Changed
- Alpha/beta datatype remains as F32 for HPA HGEMM
- Force assembly kernels to not flush denorms
- Use hipDeviceAttributePhysicalMultiProcessorCount as multiProcessorCount
### Fixed
- Fix segmentation fault when run i8 datatype with TENSILE_DB=0x80

## Tensile 4.32.0 for ROCm 5.1.0
### Added
- Better control of parallelism to control memory usage
- Support for multiprocessing on Windows for TensileCreateLibrary
- New JSD metric and metric selection functionality
- Initial changes to support two-tier solution selection
### Optimizations
- Optimized runtime of TensileCreateLibraries by reducing max RAM usage
- StoreCInUnroll additional optimizations plus adaptive K support
- DGEMM NN optimizations with PrefetchGlobalRead(PGR)=2 support
### Changed
- Update Googletest to 1.11.0
### Removed
- Remove no longer supported benchmarking steps

## Tensile 4.31.0 for ROCm 5.0.0
### Added
- DirectToLds support (x2/x4)
- DirectToVgpr support for DGEMM
- Parameter to control number of files kernels are merged into to better parallelize kernel compilation
- FP16 alternate implementation for HPA HGEMM on aldebaran
### Optimizations
- Add DGEMM NN custom kernel for HPL on aldebaran
### Changed
- Update tensile_client executable to std=c++14
### Removed
- Remove unused old Tensile client code
### Fixed
- Fix hipErrorInvalidHandle during benchmarks
- Fix addrVgpr for atomic GSU
- Fix for Python 3.8: add case for Constant nodeType
- Fix architecture mapping for gfx1011 and gfx1012
- Fix PrintSolutionRejectionReason verbiage in KernelWriter.py
- Fix vgpr alignment problem when enabling flat buffer load

## Tensile 4.30.0 for ROCm 4.5.0
### Added
- Custom Kernel mechanism for adding custom assembly kernels to Tensile
- New assertions for problems sizes, alpha/beta values, and C equals D
- Support setting VectorWidth in M dimension in MFMA SourceSwap configuration
### Fixed
- Fix merge.py keeping duplicate solutions
- Fix ScheduleIterAlg 2,3 cases for aldebaran

## Tensile 4.28.0 for ROCm 4.3.0
### Added
- TensileRetuneLibrary for updating existing library logic files
- Support GFX1030
- Support NHWC

### Fixed
- TensileCreateLibrary crash with relative output and --merge-files

### Changed
- Change cmake_minimum_required to VERSION 3.13

## Tensile 4.27.0 for ROCm 4.2.0
### Added
- Benchmarking and library support for CU efficiency vs. overall speed
- support general batch GEMM
- Support offset for each input/output buffer in Tensile
- support support ldc != ldd for all GEMM kernel

### Optimizations
- Refactor ConvolutionVsContraction

### Fixed
- Fixed MasterSolutionLibrary having duplicated hardware rows
- channel stride is incorrect when converting conv problem into tensor contraction problem

## Tensile 4.26.0 for ROCm 4.1.0
### Added
- Make messagepack python dependency optional
- TensileCreateLibraryFiles: auto create target for build time lib generation
- Tensile cluster tuning tool
- Framework for filtering solutions
- Workflow for manually editing Kernels
- Tuning client design doc
- MatrixInstruction for general int8
- Tensile integration test for TensileCreateLibrary
- Trig float and random narrow init patterns for new client
- Summation dimension mirroring (contributed by timlathy & Slimakanzer)
- ROCm 4.1 TargetID support in Tensile; source kernels force xnack=OFF
- Tensile/Utilities/merge.py revamp for merging logic yaml files
  - now merge.py requires python3
  - add `-v` verbosity levels (up to 2)
  - add `--notrim` to retain leading dimensions in sizes
- New BoundsCheck design: Access guard page will trigger memory fault
- Solution fitness metric
- Auto-tuning documentation and build script improvements
- Support for High Precision Accumulate FP16/BF16 In FP32 Out
- CHANGELOG.md

### Optimizations
- Refine PersistentKernel: support PKn1, EPS, optimize LW-vmcnt and sMagicDiv2

### Fixed
- targets to clang-offload-bundler updated to use hipv4 prefix when appropriate
- Fix bugs of tail-loop branch label, and LR addr restore
- locateExe in Tensile/Common.py looks in defaultPath first
- Honor $ENV{ROCM_PATH} to support relocatable ROCm location
