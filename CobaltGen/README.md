# CobaltGen

CobaltGen is a suite of Python scripts which automates benchmarking and library source generation.

## (1) GenBenchmark.py: app\_log.xml -> Benchmark.cpp

1. Parse Problems from AssignSolution section log.xml and create ProblemList.
2. For each Problem in ProblemList
  1. SolutionPerformanceParameters[] = GetSolutionsSet(problem)
  2. SolutionSet = SolutionCorrectnessParameters
      + SolutionPerformanceParameters[]
  3. Problem\_SolutionSet\_List.append(Problem,SolutionSet)
3. For each {Problem, SolutionSet} in Problem\_SolutionSet\_List
  1. For each {Problem, Solution} pair
    1. append Solution to MegaSolutionSet
    2. write Problem,Solution pair to ToBeBenchmarked list std::queue
      - queue.append(Problem,Solution); will call solution->enqueue(problem,ctrl)
4. For each Solution in MegaSolutionSet
    1. append kernels of Solution to KernelSet
    2. write Solution.cpp class
5. For each kernel of KernelSet
  - write kernel to file

## (2) Benchmark.exe -> FastestSolutionForProblem.xml
1. Options:
  1. writes compiled kernels to file
  2. do validation (will address all problem/solutions pairs; thorough)
2. Preface
  1. Allocate largest matrices
  2. Create control obj
3. For each SolutionSet (contains Problem)
  1. For each Solution in SolutionSet
    1. Solution->enqueue(Problem)
    2. benchmark performance
    3. validate
  2. choose fastest Solution for Problem
  3. write
    1. raw .csv of Problem, Solution, performance, validated
    2. Solution.xml of {Problem, Fastest Solution} Pairs

## (3) GenLibrary.py - FastestSolutionForProblem.xml's -> CobaltLib.cpp
1. For all Solution.xml files (in priority order)
  1. For all Problem->Solution pairs
    1. append Problem->Solution pair to SolutionMap
      - SolutionMap[Solution].append(Problem)
      - SolutionMap Data Structure
        - list of devices
          - list of Problems(-Dev)
            - list of SolutionPerformanceParameters
              - leaf of tree has list of problems to which solution applies
2. For each device
  1. For each SolutionCorrectnessParameters(numdim,precision) in Map
    1. For each SolutionPerformanceParameters(dimensions sizes)
      1. condense list of Problems into list of ProblemRanges
        - outer range (num work items)
        - sum range (sum per write)
        - num dimensions
        - range[dim0] min
        - range[dim0] max
        - range[dim0] mod
  2. For each ProblemRange
    1. Write SolutionMap.cpp if Problem in ProblemRange return Solution(Correctness,Performance)
3. Write Solution.cpp classes
4. Write Kernels
5. Write pre-compile app

if (dim





ProblemRange
  match dim0? min, max, mod
  match dim1? min, max, mod
  match write-size? min, max, mod
  match product-size? min, max, mod
  product-reuse range? - square, skinny, vector (dimSize=1)

Problem
  tensorA
    precision
    dimensions[i].size .stride
  tensorB
  tensorC
  operation
  device

SolutionCorrectness
  tensorA
    precision
    numDims
    order least to greatest stride
  tensorB
  tensorC
  operation
  NOT device - don't need to know device to write kernel or enqueue


SolutionPerformance
  tile size
  branches
  alpha beta identity
  order of dimensions

# Example Contraction

for (i = 0; i < iMax; i++) {
  for (j = 0; j < jMax; j++) {
    for (k = 0; k < kMax; k++) {
      Cijk = 0;
      for (a = 0; a < aMax; a++) {
        for (b = 0; b < bMax; b++) {
          for (c = 0; c < cMax; c++) {
            Cijk += A[i,k,j,a,b,c] * B[i,j,k,a,b,c];
          }
        } // b
      } // a
    } // k
  } // j
} // i
