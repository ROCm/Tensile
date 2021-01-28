# Change Log for Tensile

## [(Unreleased) Tensile 4.27.0 for ROCm 4.2.0]
### Added
- support general batch GEMM
- Support offset for each input/output buffer in Tensile
- support support ldc != ldd for all GEMM kernel


## [Tensile 4.26.0 for ROCm 4.1.0]
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
