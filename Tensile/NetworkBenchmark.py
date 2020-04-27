################################################################################
# Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

if __name__ == "__main__":
    print("This file cannot be run as a script.  Run 'Tensile/bin/NetworkBenchmark' instead.")
    exit(1)

from . import Code
from . import Common
from .Common import assignGlobalParameters, globalParameters

import argparse
import csv
import os
import subprocess
import sys
import yaml

def convertTranspose(transpose):
    if transpose: 
        return 'T'
    return 'N'

def convertToProblemIdentifier(transpose):
    contractionMap = {"NN":"Contraction_l_Ailk_Bljk_Cijk_Dijk","NT":"Contraction_l_Ailk_Bjlk_Cijk_Dijk","TN":"Contraction_l_Alik_Bljk_Cijk_Dijk","TT":"Contraction_l_Alik_Bjlk_Cijk_Dijk"}
    for item in contractionMap:
        if item == transpose:
            return contractionMap[item]

def convertToDataType(dataType):
    dataTypeMap = {'s':"Float",'d':"Double",'h':"Half"}
    for item in dataTypeMap:
        if item == dataType:
            return dataTypeMap[item]

def ParseNetworkConfig(network, problemSizes):
    problemTypeCounter = 1
    with open(network) as f:
        data = yaml.load(f,yaml.FullLoader)
        for problem in data:
            if "GlobalParameters" in problem:
                assignGlobalParameters(data[problem])
            else:
                for param in data[problem]:
                    if param == "DataType":
                        dataType = data[problem][param]
                    elif param == "TransposeA":
                        transposeA = convertTranspose(data[problem][param])
                    elif param == "TransposeB":
                        transposeB = convertTranspose(data[problem][param])
                        matType = (globalParameters["NetworkName"], transposeA+transposeB,dataType)
                        problemSizes[matType] = problemTypeCounter
                        problemTypeCounter += 1
                    elif param == "ProblemSizes":
                        for action in data[problem]["ProblemSizes"]:
                            if "Train" in action: #ignore validate for now
                                for exact in data[problem]["ProblemSizes"]["Train"]:
                                    for items in exact.values():
                                        count = items[0]["count"]
                                        size = items[1]["size"]
                                        problemSizes[size] = count
                                    
def RunTensileClient(client, kernelTimes, counts, libraryFile, architecture):
    fileNum = 1
    splitLibraryFile = libraryFile.split('.')
    splitLibraryFile.append(splitLibraryFile[0].replace('TensileLibrary',''))
    contraction = ''
    dataType = ''
    for size in counts.keys():
        if len(size) == 3:
            contraction = convertToProblemIdentifier(str(size[1]))
            dataType = convertToDataType(str(size[2]))
            hpa = "True" if dataType == "Half" else "False"
        elif len(size) == 4:
            replaceChars = '( )'
            strSize = str(size)
            for char in replaceChars:
                strSize = strSize.replace(char,'')
            args = [client,"--library-file="+libraryFile,"--code-object="+splitLibraryFile[2]+"Kernels.so-000-"+architecture+".hsaco","--code-object="+splitLibraryFile[0]+"_"+architecture+".co","--results-file="+splitLibraryFile[2]+"../../../Data/00_Final-new.csv","--problem-identifier="+contraction,"--a-type="+dataType,"--b-type="+dataType,"--c-type="+dataType,"--d-type="+dataType,"--alpha-type="+dataType,"--beta-type="+dataType,"--high-precision-accumulate="+hpa,"--best-solution=True","--log-file=bestSolution"+str(fileNum),"--problem-size="+strSize]
            subprocess.check_call(args)

def PrintOutput(counts, kernelTimes):
    for size in counts.keys():
        if len(size) == 3:
            print("ProblemType{},".format(counts[size]),end=" ")
            print("Network Name: {0}, ProblemType: {1}, DataType: {2}".format(*size,counts[size]))
        else:
            print("{}, count: {}, kernel time: {} ms, total time: {} ms\n".format(size, counts[size], kernelTimes[size], counts[size]*kernelTimes[size])) 

def NetworkBenchmark():
    userArgs = sys.argv[1:]
    argParser = argparse.ArgumentParser()
    argParser.add_argument("network_config", help="path and name of network config yaml file")
    argParser.add_argument("tensile_library", help="path of TensileLibrary.yaml")
    argParser.add_argument("client_path", help="path of tensile_client", default=os.path.join(globalParameters["WorkingPath"], globalParameters["ClientBuildPath"],"tensile_client"))
    argParser.add_argument("architecture", help="gpu architecture", default="gfx906")

    args = argParser.parse_args(userArgs)
    network = args.network_config
    library = args.tensile_library
    client = args.client_path
    gfx = args.architecture
    
    counts = dict()
    ParseNetworkConfig(network, counts)
    
    kernelTimes = dict()
    RunTensileClient(client,kernelTimes,counts,library,gfx)

    PrintOutput(counts,kernelTimes)

if __name__ == "__main__":
    NetworkBenchmark()
