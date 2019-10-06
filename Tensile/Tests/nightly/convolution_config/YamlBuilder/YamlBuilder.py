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

            numDims = 1 + max(max(problemType["IndexAssignmentsA"]),max(problemType["IndexAssignmentsB"]))
            problemSize=[32]*numDims
            outfile.write("           - Exact: [" + ', '.join([str(d) for d in problemSize]) + "]\n")

    @staticmethod
    def run_tensile_client(request, conv, problemType):
        testYamlFile = os.path.join(request.fspath.dirname, "Yamls", request.node.name +".yaml")
        YamlBuilder.write_yaml(request, testYamlFile, conv, problemType, dataType='s')
        Tensile.Tensile([Tensile.TensileTestPath(testYamlFile), "/tmp"])

