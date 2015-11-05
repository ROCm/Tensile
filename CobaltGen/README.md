# CobaltGen

CobaltGen is a suite of Python scripts which automates benchmarking and library source generation.

## Automated benchmarking:
CobaltGen's benchmark-writer tool:

1. Parses out the Problems in a problems.xml generated from using CobaltLib with log mode turned on.
2. Enumerates all possible Solutions to each Problem.
3. Writes a benchmarking application which:
  1. Executes all Solutions.
  2. Chooses the fastest Solution for each Problem.
  3. Creates a mapping Problem->Solution.
  4. Writes the mapping to a ProblemSolutionMap.xml file.

## Automated library source generation:
CobaltGen's source-writer tool:

1. Parses a ProblemSolutionMap.xml file.
2. Writes the required GPU kernels.
3. Writes the required host code for enqueueing the kernels.
4. Writes the "GetSolution" method which maps Problem->Solution.

## Automated validation?
