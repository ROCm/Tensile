

import os
import sys
import argparse
import re

import pandas as pd



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

def ProcessResults(outputPath, resultsName):
    
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

    results = df[performanceField].mean()

    aggragateFileName = resultsName + "-aggregated.csv"
    aggragateFilePath = os.path.join(outputPath, aggragateFileName)

    results.to_csv(aggragateFilePath, header=True)
    

def RunMain():

    userArgs = sys.argv[1:]

    argParser = argparse.ArgumentParser()
    argParser.add_argument("input_path", help="path where the results are located")
    argParser.add_argument("output_path", help="path where the processed files are to go")

    args = argParser.parse_args(userArgs)

    inputPath = args.input_path
    outputPath = args.output_path

    resultsFiles = [f for f in os.listdir(inputPath) if (os.path.isfile(os.path.join(inputPath, f)))]

    resultsNameSet = set()

    for resultsFile in resultsFiles:
        resultsName, _ = os.path.splitext(resultsFile)
        resultsNameSet.add(resultsName)

    resultsNames = list(resultsNameSet)

    for resultsName in resultsNames:
        ParseResults(inputPath, outputPath, resultsName)
        ProcessResults(outputPath, resultsName)

if __name__ == "__main__":
    RunMain()