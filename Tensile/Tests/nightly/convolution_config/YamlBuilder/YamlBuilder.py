import os
import pytest
from Tensile.SolutionStructs import Convolution
import Tensile.Tensile as Tensile


class YamlBuilder:

    @staticmethod
    def catFile(outfile,fname):
        with open(fname) as infile:
            outfile.write(infile.read())

    @staticmethod
    def write_yaml(request, testYamlFile, conv, problemType, dataType):
        yaml_dir = os.path.join(request.fspath.dirname,"YamlBuilder")
        with open(os.path.join(testYamlFile), 'w') as outfile:
            YamlBuilder.catFile(outfile, os.path.join(yaml_dir, "header.yml"))
            outfile.write("BenchmarkProblems:\n")
            outfile.write("  -\n")
            outfile.write("    -\n")
            outfile.write("      OperationType: TensorContraction\n")
            outfile.write("      DataType: %s\n" % dataType)
            for k in sorted(problemType.keys()):
                outfile.write("      %s: %s\n" % (k,problemType[k]))

            YamlBuilder.catFile(outfile, os.path.join(yaml_dir,"solutions/sgemm_1.yml"))
            outfile.write("         - ProblemSizes:\n")

            (problemSizes,problemStrides) = conv.makeProblem(8, 32, 16, [14]*conv.formatNumSpatialDims)
            outfile.write("           - Exact: [" + ', '.join([str(d) for d in problemSizes]) + "]\n")

    @staticmethod
    def run_tensile_client(request, conv, problemType, tensile_dir):
        print ("TD=", tensile_dir)
        if request.config.getoption("--run_client") > 0:
            testYamlFile = os.path.join(request.fspath.dirname, "Yamls", request.node.name +".yaml")
            YamlBuilder.write_yaml(request, testYamlFile, conv, problemType, dataType='s')
            if request.config.getoption("--run_client") > 1:
                Tensile.Tensile([Tensile.TensileTestPath(testYamlFile), str(tensile_dir)])

    setupHeader="""
GlobalParameters:
  MinimumRequiredVersion: 4.2.0
  ForceRedoBenchmarkProblems: True
  ForceRedoLibraryLogic: True
  ForceRedoLibraryClient: True
  CMakeBuildType: Release
  EnqueuesPerSync: 1
  SyncsPerBenchmark: 1
  LibraryPrintDebug: True
  NumElementsToValidate: 1000
  ValidationMaxToPrint: 4
  ValidationPrintValids: False
  ShortNames: False
  MergeFiles: True
  Platform: 0
  Device: 0
  KernelTime: True
  DataInitTypeBeta : 0
  SolutionSelectionAlg: 1
  NewClient: 2
  CpuThreads: 0
BenchmarkProblems:
  -
    -
      OperationType: TensorContraction
      DataType: s
      IndexAssignmentsA: [0, 2]
      IndexAssignmentsB: [2, 1]
      NumIndicesC: 2
      UseBeta: False
      UseInitialStrides: True
    -
      BenchmarkCommonParameters:
      ForkParameters:
      BenchmarkForkParameters:
      JoinParameters:
      BenchmarkJoinParameters:
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [4, 4, 4]
"""

