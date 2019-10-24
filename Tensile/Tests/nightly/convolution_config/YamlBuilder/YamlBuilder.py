import os
import pytest
from Tensile.SolutionStructs import Convolution
import Tensile.Tensile as Tensile


class YamlBuilder:

    @staticmethod
    def catFile(outfile,fname):
        with open(fname) as infile:
            outfile.write(infile.read())

    def write_problem_sizes(conv, outfile):
        outfile.write("         - ProblemSizes:\n")

        try:
            spatialIn = conv.spatial
        except KeyError:
            spatialIn = [14]*conv.formatNumSpatialDims
        # replace any TBD spatials with something:
        spatialIn = [x if x>0 else 42 for x in spatialIn]

        if not all(i!=-1 for i in conv.filter):
            raise RuntimeError('Filter must be completely specified, not "%s"'%conv.config['Filter'])

        (problemSizes,problemStrides) = conv.makeProblem(False, 8, 32, 16, spatialIn)
        outfile.write("           - Exact: [" + ', '.join([str(d) for d in problemSizes]) + "]\n")

    @staticmethod
    def write_conv_yaml(request, testYamlFile, conv, problemType, dataType):
        yaml_builder_dir = os.path.join(request.fspath.dirname,"YamlBuilder")
        with open(os.path.join(testYamlFile), 'w') as outfile:
            YamlBuilder.catFile(outfile, os.path.join(yaml_builder_dir, "header.yml"))
            outfile.write("  ConvolutionVsContraction: 1\n")
            outfile.write("BenchmarkProblems:\n")
            outfile.write("  -\n")
            outfile.write("    -\n")
            outfile.write("      OperationType: %s\n" % conv.convolutionType)
            outfile.write("      DataType: %s\n" % dataType)
            outfile.write("      ConvolutionConfig:\n")
            for k in sorted(conv.config.keys()):
                outfile.write("      - %s: %s\n" % (k, conv.config[k]))

            YamlBuilder.catFile(outfile, os.path.join(yaml_builder_dir,"solutions/sgemm_1.yml"))
            YamlBuilder.write_problem_sizes(conv, outfile)

    @staticmethod
    def write_yaml(request, testYamlFile, conv, problemType, dataType):
        yaml_builder_dir = os.path.join(request.fspath.dirname,"YamlBuilder")
        with open(os.path.join(testYamlFile), 'w') as outfile:
            YamlBuilder.catFile(outfile, os.path.join(yaml_builder_dir, "header.yml"))

            outfile.write("BenchmarkProblems:\n")
            outfile.write("  -\n")
            outfile.write("    -\n")
            outfile.write("      OperationType: TensorContraction\n")
            outfile.write("      DataType: %s\n" % dataType)
            for k in sorted(problemType.keys()):
                outfile.write("      %s: %s\n" % (k,problemType[k]))

            YamlBuilder.catFile(outfile, os.path.join(yaml_builder_dir,"solutions/sgemm_1.yml"))
            YamlBuilder.write_problem_sizes(conv, outfile)


    @staticmethod
    def run_tensile_client(request, conv, problemType, tensile_dir):
        run_client = request.config.getoption("--run_client")

        if run_client>=3:
            YamlBuilder.run_convolution_vs_contraction(request,conv,problemType, tensile_dir)

        if run_client>=1:
            testYamlFile = os.path.join(request.fspath.dirname, "Yamls", request.node.name +".yaml")
            YamlBuilder.write_yaml(request, testYamlFile, conv, problemType, dataType='s')
            if run_client>=2:
                Tensile.Tensile([Tensile.TensileTestPath(testYamlFile), str(tensile_dir)])

    @staticmethod
    def run_convolution_vs_contraction(request, conv, problemType, tensile_dir):
        run_client = request.config.getoption("--run_client")

        testYamlFile = os.path.join(request.fspath.dirname, "Yamls", request.node.name +".conv.yaml")
        YamlBuilder.write_conv_yaml(request, testYamlFile, conv, problemType, dataType='s')
        Tensile.Tensile([Tensile.TensileTestPath(testYamlFile), str(tensile_dir)])


    setupHeader="""
GlobalParameters:
  MinimumRequiredVersion: 4.2.0
  CMakeBuildType: Release
  EnqueuesPerSync: 1
  SyncsPerBenchmark: 1
  LibraryPrintDebug: False
  NumElementsToValidate: -1
  ValidationMaxToPrint: 4
  ValidationPrintValids: False
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

