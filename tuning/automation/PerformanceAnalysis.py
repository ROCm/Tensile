################################################################################
# Copyright (C) 2016-2019 Advanced Micro Devices, Inc. All rights reserved.
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


import os
import sys
import argparse
import re

import pandas as pd
from ExtractSizes import *
from TuningConfiguration import *

headers = ""

def MatchLine(headerPattern, linePattern, line):

    global headers

    if not headers:
        matched = headerPattern.match(line)

        if matched:
            headers = line

        return matched
    else:
        matched = linePattern.match(line)

        return matched

def ResultsFilesList(inputPath, resultsName):

    resultsFilePattern = re.compile(resultsName + "\.[0-9]*")
    resultsFiles = [f for f in os.listdir(inputPath)]

    filteredFiles = [f for f in resultsFiles if resultsFilePattern.match(f)]

    return filteredFiles

def ParseResults(inputPath, outputPath, resultsName):

    global headers

    headers = ""

    filteredFiles = ResultsFilesList(inputPath, resultsName)

    headerPattern = re.compile("transA,transB")
    linePattern = re.compile(r"(N|T),(N|T).*")

    outfilename = resultsName + ".csv"

    outputFilePath = os.path.join(outputPath, outfilename)
    outfile = open(outputFilePath,'w')

    for fl in filteredFiles:
        flPath = os.path.join(inputPath,fl)
        filteredLines = [ line for line in open(flPath) if MatchLine(headerPattern, linePattern, line)]
        outfile.writelines(filteredLines)
    outfile.flush()
    outfile.close()

def ProcessResults(outputPath, resultsName, freqM, sz, call_count):
    
    global headers
    resultsFilename = resultsName + ".csv"
    
    resultsFilePath = os.path.join(outputPath, resultsFilename)

    data = None
    data = pd.read_csv(resultsFilePath)

    headerValues = headers.split(",")
    headerLength = len(headerValues)
    key = headerValues[0:headerLength-2]

    performanceField = "rocblas-Gflops"

    df = data.groupby(key)

    results = df[performanceField].mean().to_frame()

    freq=freqM * 1000000.0
    factor=sz * 4096 
    results['eff'] = 1e9*results['rocblas-Gflops'] / (factor * freq) 
    results['wa'] = results['rocblas-Gflops']*call_count

    aggragateFileName = resultsName + "-aggregated.csv"
    aggragateFilePath = os.path.join(outputPath, aggragateFileName)

    results.to_csv(aggragateFilePath, header=True)
    
    resultsBad = results[results['eff'] < 0.70]
    badResultsFileName = resultsName + "-bad.csv"
    badResultsFilePath = os.path.join(outputPath, badResultsFileName)
    resultsBad.to_csv(badResultsFilePath, header=True)    

    large1 = data
    large1['N'] = pd.to_numeric(large1['N'])
    large1['M'] = pd.to_numeric(large1['M'])
    large2 = large1[large1['N']>1000]
    large = large2[large2['M']>1000]
    
    largeAgg = large.groupby(key)
    largeResults = largeAgg[performanceField].mean().to_frame()
    largeResults['eff'] = 1e9*largeResults['rocblas-Gflops'] / (factor * freq)
    largeResults['wa'] = results['rocblas-Gflops']*call_count

    resultsFileName = resultsName + "-large.csv"
    resultsFilePath = os.path.join(outputPath, resultsFileName)
    largeResults.to_csv(resultsFilePath, header=True)  

    resultsBad = largeResults[largeResults['eff'] < 0.70]
    badResultsFileName = resultsName + "-bad-large.csv"
    badResultsFilePath = os.path.join(outputPath, badResultsFileName)
    resultsBad.to_csv(badResultsFilePath, header=True)  

def RunMain():

    userArgs = sys.argv[1:]

    argParser = argparse.ArgumentParser()
    argParser.add_argument("input_file_name", help="configuration file path") 
    argParser.add_argument("input_path", help="path where the results are located")
    argParser.add_argument("output_path", help="path where the processed files are to go")
    argParser.add_argument("frequency", help="frequecy in megahertz used in testing", type=int,default=1301)
    argParser.add_argument("data_size", help="data size",type=int,default=2)
    
    args = argParser.parse_args(userArgs)

    inputFileName = args.input_file_name
    inputPath = args.input_path
    outputPath = args.output_path
    freqM = args.frequency
    sz = args.data_size
    
    problemMapper = list(ProcessFile(inputFileName).values())
    callCounts = list()

    for i in problemMapper:
        for klist in i:
            for key in klist:
                if key == "call_count":
                    callCounts.append(klist[key])
    
    resultsFiles = [f for f in os.listdir(inputPath) if (os.path.isfile(os.path.join(inputPath, f)))]
    resultsNameSet = set()

    for resultsFile in resultsFiles:
        resultsName, _ = os.path.splitext(resultsFile)
        resultsNameSet.add(resultsName)
    
    resultsNames = list(resultsNameSet)

    for resultsName in resultsNames:
        ParseResults(inputPath, outputPath, resultsName)
        ProcessResults(outputPath, resultsName, freqM, sz, callCounts)


if __name__ == "__main__":
    RunMain()
