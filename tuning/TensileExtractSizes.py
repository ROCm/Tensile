import os
import sys
import argparse

import csv

HR = "################################################################################"

################################################################################
# Print Debug
################################################################################


def printWarning(message):
    print "Tensile::WARNING: %s" % message
    sys.stdout.flush()


def printExit(message):
    print "Tensile::FATAL: %s" % message
    sys.stdout.flush()
    sys.exit(-1)


try:
    import yaml
except ImportError:
    printExit(
        "You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions."
    )


def ensurePath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


################################################################################
# Library Logic Container
################################################################################
class LibraryLogic:
    def __init__(self, filename=None):

        if filename is not None:
            print("# Reading Library Logic: " + filename)
            try:
                stream = open(filename, "r")
            except IOError:
                printExit("Cannot open file: %s" % filename)
            data = yaml.load(stream, yaml.SafeLoader)

            self.__set_versionString(data[0]["MinimumRequiredVersion"])
            self.__set_scheduleName(data[1])
            self.__set_architectureName(data[2])
            self.__set_deviceNames(data[3])
            self.__set_problemType(data[4])
            self.__set_solutionStates(data[5])
            self.__set_indexOrder(data[6])
            self.__set_exactLogic(data[7])
            self.__set_rangeLogic(data[8])

            stream.close()

        else:
            self.__set_versionString(None)
            self.__set_scheduleName(None)
            self.__set_architectureName(None)
            self.__set_deviceNames(None)
            self.__set_problemType(None)
            self.__set_solutionStates(None)
            self.__set_indexOrder(None)
            self.__set_exactLogic(None)
            self.__set_rangeLogic(None)

    #versionString
    def __get_versionString(self):
        return self.__versionString

    def __set_versionString(self, value):
        self.__versionString = value

    versionString = property(__get_versionString, __set_versionString)

    #scheduleName
    def __get_scheduleName(self):
        return self.__scheduleName

    def __set_scheduleName(self, value):
        self.__scheduleName = value

    scheduleName = property(__get_scheduleName, __set_scheduleName)

    #architectureName
    def __get_architectureName(self):
        return self.__architectureName

    def __set_architectureName(self, value):
        self.__architectureName = value

    architectureName = property(__get_architectureName, __set_architectureName)

    #deviceNames
    def __get_deviceNames(self):
        return self.__deviceNames

    def __set_deviceNames(self, value):
        self.__deviceNames = value

    deviceNames = property(__get_deviceNames, __set_deviceNames)

    #problemTypeState
    def __get_problemType(self):
        return self.__problemType

    def __set_problemType(self, value):
        self.__problemType = value

    problemType = property(__get_problemType, __set_problemType)

    #solutionStates
    def __get_solutionStates(self):
        return self.__solutionStates

    def __set_solutionStates(self, value):
        self.__solutionStates = value

    solutionStates = property(__get_solutionStates, __set_solutionStates)

    #indexOrder
    def __get_indexOrder(self):
        return self.__indexOrder

    def __set_indexOrder(self, value):
        self.__indexOrder = value

    indexOrder = property(__get_indexOrder, __set_indexOrder)

    #exactLogic
    def __get_exactLogic(self):
        return self.__exactLogic

    def __set_exactLogic(self, value):
        self.__exactLogic = value

    exactLogic = property(__get_exactLogic, __set_exactLogic)

    #rangeLogic
    def __get_rangeLogic(self):
        return self.__rangeLogic

    def __set_rangeLogic(self, value):
        self.__rangeLogic = value

    rangeLogic = property(__get_rangeLogic, __set_rangeLogic)

    def writeLibraryLogic(self, filename):

        data = []

        if self.versionString is not None:
            data.append({"MinimumRequiredVersion": self.versionString})

        if self.scheduleName is not None:
            data.append(self.scheduleName)

        if self.architectureName is not None:
            data.append(self.architectureName)

        if self.deviceNames is not None:
            data.append(self.deviceNames)

        if self.problemType is not None:
            data.append(self.problemType)

        if self.solutionStates is not None:
            data.append(self.solutionStates)

        if self.indexOrder is not None:
            data.append(self.indexOrder)

        if self.exactLogic is not None:
            data.append(self.exactLogic)

        if self.rangeLogic is not None:
            data.append(self.rangeLogic)

        if not data:
            printExit("No data to output")
        else:
            try:
                stream = open(filename, "w")
                yaml.safe_dump(data, stream)
                stream.close()
            except IOError:
                printExit("Cannot open file: %s" % filename)


def makeCSVFileName(filePath):

    _, fullFileName = os.path.split(filePath)
    fileName, _ = os.path.splitext(fullFileName)

    outputFileName = fileName + "-sizes.csv"

    return outputFileName


def makeAugmentedFileName(filePath, tagExtension):
    _, fullFileName = os.path.split(filePath)
    fileName, _ = os.path.splitext(fullFileName)

    outputFileName = fileName + tagExtension

    return outputFileName


def ExtractSizes(inputFilePath, outputFilePath):

    libraryLogic = LibraryLogic(inputFilePath)
    exactLogic = libraryLogic.exactLogic
    exactSizes = [esize[0] for esize in exactLogic]

    #with open(outputFilePath, "wb") as f:
    #  writer = csv.writer(f)
    #  writer.writerows(exactSizes)

    return exactSizes


#def sizeToBenchArgs(size):
#  m = size[0]
#  n = size[1]
#  k = size[2]
#  l = size[3]

#  alpha = 1
#  beta = 0

#  line = "./rocblas-bench -f gemm -r h --transposeA N --transposeB N -m %u -n %u -k %u --alpha %u --lda %u --ldb %u --beta %u --ldc %u \n" \
#    % (m,n,l,alpha,m,k,beta,m)

#  return line


def getMapping(label, mapper):

    mapped = ""
    if label in mapper:
        mapped = mapper[label]

    return mapped


def getRunParametersFromName(ligicSignature):

    fields = ligicSignature.split('_')
    nFields = len(fields)

    matrixLabelA = fields[nFields - 3]
    matrixLabelB = fields[nFields - 2]
    typeLabel = fields[nFields - 1]

    transposeMapperA = {"Ailk": "N", "Alik": "T"}
    transposeMapperB = {"Bjlk": "T", "Bljk": "N"}
    functionMapper = {
        "HB": "gemm",
        "SB": "gemm",
        "DB": "gemm",
        "HBH": "gemm_ex"
    }
    typeNameMapper = {"HB": "h", "SB": "s", "DB": "d", "HBH": "h"}

    transposeA = getMapping(matrixLabelA, transposeMapperA)
    transposeB = getMapping(matrixLabelB, transposeMapperB)
    functionName = getMapping(typeLabel, functionMapper)
    typeName = getMapping(typeLabel, typeNameMapper)

    runParameters = [transposeA, transposeB, functionName, typeLabel, typeName]

    return runParameters


def makeLine(runParams, size):

    m = size[0]
    n = size[1]
    k = size[2]
    l = size[3]

    alpha = 1
    beta = 0

    transposeA = runParams[0]
    transposeB = runParams[1]
    functionName = runParams[2]
    label = runParams[3]
    typeName = runParams[4]

    line = "./rocblas-bench -f %s -r %s --transposeA %s --transposeB %s" % (
        functionName, typeName, transposeA, transposeB)
    line += " -m %u -n %u -k %u --alpha %u --lda %u --ldb %u --beta %u --ldc %u" % (
        m, n, l, alpha, m, k, beta, m)

    if label == "HBH":
        line += " --a_type h --b_type h --c_type h --d_type h --compute_type s"

    line += " \n"

    return line


def writeBenchmarkScript(scriptFilePath, exactSizes, runParams):

    f = open(scriptFilePath, "wb")
    f.writelines(["#!/bin/sh\n", "\n", "\n"])

    lines = []
    for size in exactSizes:
        line = makeLine(runParams, size)
        lines.append(line)

    f.writelines(lines)

    f.close()


def RunMergeTensileLogicFiles():

    print ""
    print HR
    print "# Extract sizes"
    print HR
    print ""

    ##############################################################################
    # Parse Command Line Arguments
    ##############################################################################

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "ExactLogicPath",
        help="Path to the exact LibraryLogic.yaml input files.")
    argParser.add_argument("OutputPath", help="Where to write library files?")
    #argParser.add_argument("-b", dest="BenchmarkScript", help="write benchmark test script")
    argParser.add_argument(
        "-b",
        dest="doBenchmarkScript",
        action="store_true",
        help="write benchmark test script")

    args = argParser.parse_args()

    exactLogicPath = args.ExactLogicPath
    outputPath = args.OutputPath
    #doBenchmarkScript = args.doBenchmarkScript

    ensurePath(outputPath)
    if not os.path.exists(exactLogicPath):
        printExit("LogicPath %s doesn't exist" % exactLogicPath)

    exactLogicFiles = [os.path.join(exactLogicPath, f) for f in os.listdir(exactLogicPath) \
        if (os.path.isfile(os.path.join(exactLogicPath, f)) \
        and os.path.splitext(f)[1]==".yaml")]

    #print exactLogicFiles

    for f in exactLogicFiles:

        print "processing " + f
        fullFilePath = os.path.join(exactLogicPath, f)

        name = makeAugmentedFileName(fullFilePath, "")
        runParameters = getRunParametersFromName(name)

        #print runParameters

        outputFileName = makeCSVFileName(fullFilePath)
        outputFile = os.path.join(outputPath, outputFileName)
        sizes = ExtractSizes(fullFilePath, outputFile)

        #print sizes
        benchmarkFileName = makeAugmentedFileName(fullFilePath,
                                                  "-benchmark.sh")
        benchmarkScriptName = os.path.join(outputPath, benchmarkFileName)
        writeBenchmarkScript(benchmarkScriptName, sizes, runParameters)


################################################################################
# Main
################################################################################
if __name__ == "__main__":
    RunMergeTensileLogicFiles()
