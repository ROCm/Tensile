# CobaltGen

## Description
CobaltGen is a suite of Python scripts which automates benchmarking and library source generation.

## Automated benchmarking:
CobaltGen's benchmark-writer tool:

- Parses out the Problems in a problems.xml generated from using CobaltLib with log mode turned on.
- Enumerates all possible Solutions to each Problem.
- Writes a benchmarking application which:
  - Executes all Solutions.
  - Chooses the fastest Solution for each Problem.
  - Creates a mapping Problem->Solution.
  - Writes the mapping to a ProblemSolutionMap.xml file.

## Automated library source generation:
CobaltGen's source-writer tool:

- Parses a ProblemSolutionMap.xml file.
- Writes the required GPU kernels.
- Writes the required host code for enqueueing the kernels.
- Writes the "GetSolution" method which maps Problem->Solution.

## Automated validation?
