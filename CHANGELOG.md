# Change Log for Tensile

## (Unreleased) Tensile 4.39.0
### Added
- Added aquavanjaram support: gfx940/gfx941/gfx942, fp8/bf8 datatype, xf32 datatype, and stochastic rounding for various datatypes
- Added/updated tuning scripts
- Added DirectToLds support for larger data types with 32bit global load (old parameter DirectToLds is replaced with DirectToLdsA and DirectToLdsB), and the corresponding test cases
- Added the average of frequency, power consumption, and temperature information for the winner kernels to the CSV file
- Added asmcap check for MFMA + const src
- Added support for wider local read + pack with v_perm (with VgprForLocalReadPacking=True)
- Added a new parameter to increase miLatencyLeft

### Optimizations
- Enabled InitAccVgprOpt for MatrixInstruction cases
- Implemented local read related parameter calculations with DirectToVgpr
- Adjusted miIssueLatency for gfx940
- Enabled dedicated vgpr allocation for local read + pack
- Optimized code initialization
- Optimized sgpr allocation
- Supported DGEMM TLUB + RLVW=2 for odd N (edge shift change)
- Enabled miLatency optimization for (gfx940/gfx941 + MFMA) for specific data types, and fixed instruction scheduling

### Changed
- Removed old code for DTL + (bpe * GlobalReadVectorWidth > 4)
- Changed/updated failed CI tests for gfx11xx, InitAccVgprOpt, and DTLds
- Removed unused CustomKernels and ReplacementKernels
- Added a reject condition for DTVB + TransposeLDS=False (not supported so far)
- Removed unused code for DirectToLds
- Updated test cases for DTV + TransposeLDS=False
- Moved parameter MinKForGSU from globalparameter to BenchmarkCommonParameter to support smaller K
- Changed how to calculate latencyForLR for miLatency
- Set minimum value of latencyForLRCount for 1LDSBuffer to avoid getting rejected by overflowedResources=5 (related to miLatency)
- Refactored allowLRVWBforTLUandMI and renamed it as VectorWidthB
- Supported multi-gpu for different architectures in lazy library loading
- Enabled dtree library for batch > 1
- Added problem scale feature for dtree selection
- Enabled ROCm SMI for gfx940/941.
- Modified non-lazy load build to skip experimental logic

### Fixed
- Fixed predicate ordering for fp16alt impl round near zero mode to unbreak distance modes
- Fixed boundary check for mirror dims and re-enable disabled mirror dims test cases
- Fixed merge error affecting i8 with wmma
- Fixed mismatch issue with DTLds + TSGR + TailLoop
- Fixed a bug with InitAccVgprOpt + GSU>1 and a mismatch issue with PGR=0
- Fixed override for unloaded solutions when lazy loading
- Fixed build some errors (adding missing headers)
- Fixed boost link for a clean build on ubuntu22
- Fixed bug in forcestoresc1 arch selection
- Fixed compiler directive for gfx941 and gfx942
- Fixed formatting for DecisionTree_test.cpp

## Tensile 4.38.0 for ROCm 5.7
### Added
- Added support for FP16 Alt Round Near Zero Mode (this feature allows the generation of alternate kernels with intermediate rounding instead of truncation)
- Added user-driven solution selection feature
### Optimizations
- Enabled LocalSplitU with MFMA for I8 data type
- Optimized K mask code in mfmaIter
- Enabled TailLoop code in NoLoadLoop to prefetch global/local read
- Enabled DirectToVgpr in TailLoop for NN, TN, and TT matrix orientations
- Optimized DirectToLds test cases to reduce the test duration
### Changed
- Removed DGEMM NT custom kernels and related test cases
- Changed noTailLoop logic to apply noTailLoop only for NT
- Changed the range of AssertFree0ElementMultiple and Free1
- Unified aStr, bStr generation code in mfmaIter
### Fixed
- Fixed LocalSplitU mismatch issue for SGEMM
- Fixed BufferStore=0 and Ldc != Ldd case
- Fixed mismatch issue with TailLoop + MatrixInstB > 1

## Tensile 4.37.0 for ROCm 5.6
### Added
- Added user driven tuning API
- Added decision tree fallback feature
- Added SingleBuffer + AtomicAdd option for GlobalSplitU
- DirectToVgpr support for fp16 and Int8 with TN orientation
- Added new test cases for various functions
- Added SingleBuffer algorithm for ZGEMM/CGEMM
- Added joblib for parallel map calls
- Added support for MFMA + LocalSplitU + DirectToVgprA+B
- Added asmcap check for MIArchVgpr
- Added support for MFMA + LocalSplitU
- Added frequency, power, and temperature data to the output
### Optimizations
- Improved the performance of GlobalSplitU with SingleBuffer algorithm
- Reduced the running time of the extended and pre_checkin tests
- Optimized the Tailloop section of the assembly kernel
- Optimized complex GEMM (fixed vgpr allocation, unified CGEMM and ZGEMM code in MulMIoutAlphaToArch)
- Improved the performance of the second kernel of MultipleBuffer algorithm
### Changed
- Updated custom kernels with 64-bit offsets
- Adapted 64-bit offset arguments for assembly kernels
- Improved temporary register re-use to reduce max sgpr usage
- Removed some restrictions on VectorWidth and DirectToVgpr
- Updated the dependency requirements for Tensile
- Changed the range of AssertSummationElementMultiple
- Modified the error messages for more clarity
- Changed DivideAndReminder to vectorStaticRemainder in case quotient is not used
- Removed dummy vgpr for vectorStaticRemainder
- Removed tmpVgpr parameter from vectorStaticRemainder/Divide/DivideAndReminder
- Removed qReg parameter from vectorStaticRemainder
### Fixed
- Fixed tmp sgpr allocation to avoid over-writing values (alpha)
- 64-bit offset parameters for post kernels
- Fixed gfx908 CI test failures
- Fixed offset calculation to prevent overflow for large offsets
- Fixed issues when BufferLoad and BufferStore are equal to zero
- Fixed StoreCInUnroll + DirectToVgpr + no useInitAccVgprOpt mismatch
- Fixed DirectToVgpr + LocalSplitU + FractionalLoad mismatch
- Fixed the memory access error related to StaggerU + large stride
- Fixed ZGEMM 4x4 MatrixInst mismatch
- Fixed DGEMM 4x4 MatrixInst mismatch
- Fixed ASEM + GSU + NoTailLoop opt mismatch
- Fixed AssertSummationElementMultiple + GlobalSplitU issues
- Fixed ASEM + GSU + TailLoop inner unroll

## Tensile 4.36.0 for ROCm 5.5.0
### Added
- Add functions for user-driven tuning
- Add GFX11 support: HostLibraryTests yamls, rearragne FP32(C)/FP64(C) instruction order, archCaps for instruction renaming condition, adjust vgpr bank for A/B/C for optimize, separate vscnt and vmcnt, dual mac
- Add binary search for Grid-Based algorithm
- Add reject condition for (StoreCInUnroll + BufferStore=0) and (DirectToVgpr + ScheduleIterAlg<3 + PrefetchGlobalRead==2)
- Add support for (DirectToLds + hgemm + NN/NT/TT) and (DirectToLds + hgemm + GlobalLoadVectorWidth < 4)
- Add support for (DirectToLds + hgemm(TLU=True only) or sgemm + NumLoadsCoalesced > 1)
- Add GSU SingleBuffer algorithm for HSS/BSS
- Add gfx900:xnack-, gfx1032, gfx1034, gfx1035
- Enable gfx1031 support
### Optimizations
- Use AssertSizeLessThan for BufferStoreOffsetLimitCheck if it is smaller than MT1
- Improve InitAccVgprOpt
### Changed
- Use global_atomic for GSU instead of flat and global_store for debug code
- Replace flat_load/store with global_load/store
- Use global_load/store for BufferLoad/Store=0 and enable scheduling
- LocalSplitU support for HGEMM+HPA when MFMA disabled
- Update Code Object Version
- Type cast local memory to COMPUTE_DATA_TYPE in LDS to avoid precision loss
- Update asm cap cache arguments
- Unify SplitGlobalRead into ThreadSeparateGlobalRead and remove SplitGlobalRead
- Change checks, error messages, assembly syntax, and coverage for DirectToLds
- Remove unused cmake file
- Clean up the LLVM dependency code
- Update ThreadSeparateGlobalRead test cases for PrefetchGlobalRead=2
- Update sgemm/hgemm test cases for DirectToLds and ThreadSepareteGlobalRead
### Fixed
- Add build-id to header of compiled source kernels
- Fix solution index collisions
- Fix h beta vectorwidth4 correctness issue for WMMA
- Fix an error with BufferStore=0
- Fix mismatch issue with (StoreCInUnroll + PrefetchGlobalRead=2)
- Fix MoveMIoutToArch bug
- Fix flat load correctness issue on I8 and flat store correctness issue
- Fix mismatch issue with BufferLoad=0 + TailLoop for large array sizes
- Fix code generation error with BufferStore=0 and StoreCInUnrollPostLoop
- Fix issues with DirectToVgpr + ScheduleIterAlg<3
- Fix mismatch issue with DGEMM TT + LocalReadVectorWidth=2
- Fix mismatch issue with PrefetchGlobalRead=2
- Fix mismatch issue with DirectToVgpr + PrefetchGlobalRead=2 + small tile size
- Fix an error with PersistentKernel=0 + PrefetchAcrossPersistent=1 + PrefetchAcrossPersistentMode=1
- Fix mismatch issue with DirectToVgpr + DirectToLds + only 1 iteration in unroll loop case
- Remove duplicate GSU kernels: for GSU = 1, GSUAlgorithm SingleBuffer and MultipleBuffer kernels are identical
- Fix for failing CI tests due to CpuThreads=0
- Fix mismatch issue with DirectToLds + PrefetchGlobalRead=2
- Remove the reject condition for ThreadSeparateGlobalRead and DirectToLds (HGEMM, SGEMM only)
- Modify reject condition for minimum lanes of ThreadSeparateGlobalRead (SGEMM or larger data type only)

## Tensile 4.35.0 for ROCm 5.4.0
### Added
- Async DMA support for Transpose Data Layout (ThreadSeparateGlobalReadA/B)
- Option to output library logic in dictionary format
- No solution found error message for benchmarking client
- Exact K check for StoreCInUnrollExact
- Support for CGEMM + MIArchVgpr
- client-path parameter for using prebuilt client
- CleanUpBuildFiles global parameter
- Debug flag for printing library logic index of winning solution
- NumWarmups global parameter for benchmarking
- Windows support for benchmarking client
- DirectToVgpr support for CGEMM
- TensileLibLogicToYaml for creating tuning configs from library logic solutions
### Optimizations
- Put beta code and store separately if StoreCInUnroll = x4 store
- Improved performance for StoreCInUnroll + b128 store
### Changed
- Re-enable HardwareMonitor for gfx90a
- Decision trees use MLFeatures instead of Properties
### Fixed
- Reject DirectToVgpr + MatrixInstBM/BN > 1
- Fix benchmark timings when using warmups and/or validation
- Fix mismatch issue with DirectToVgprB + VectorWidth > 1
- Fix mismatch issue with DirectToLds + NumLoadsCoalesced > 1 + TailLoop
- Fix incorrect reject condition for DirectToVgpr
- Fix reject condition for DirectToVgpr + MIWaveTile < VectorWidth
- Fix incorrect instruction generation with StoreCInUnroll

## Tensile 4.34.0 for ROCm 5.3.0
### Added
- Lazy loading of solution libraries and code object files
- Support for dictionary style logic files
- Support for decision tree based logic files using dictionary format
- DecisionTreeLibrary for solution selection
- DirectToLDS support for HGEMM
- DirectToVgpr support for SGEMM
- Grid based distance metric for solution selection
- Support for gfx11xx
- Support for DirectToVgprA/B + TLU=False
- ForkParameters Groups as a way of specifying solution parameters
- Support for a new Tensile yaml config format
- TensileClientConfig for generating Tensile client config files
- Options for TensileCreateLibrary to build client and create client config file
### Optimizations
- Solution generation is now cached and is not repeated if solution parameters are unchanged
### Changed
- Default MACInstruction to FMA
### Fixed
- Accept StaggerUStride=0 as valid
- Reject invalid data types for UnrollLoopEfficiencyEnable
- Fix invalid code generation issues related to DirectToVgpr
- Return hipErrorNotFound if no modules are loaded
- Fix performance drop for NN ZGEMM with 96x64 macro tile
- Fix memory violation for general batched kernels when alpha/beta/K = 0

## Tensile 4.33.0 for ROCm 5.2.0
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
