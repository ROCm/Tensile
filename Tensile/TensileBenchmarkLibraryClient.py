################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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
import argparse
import sys
import csv
import os
from subprocess import Popen, PIPE

################################################################################
# Benchmark Problem Size
################################################################################
def BenchmarkProblemSize(cmdPrefix, row):
  cmd = cmdPrefix
  cmd += " --sizes"
  for size in row:
    cmd += " "
    cmd += size
  process = Popen(cmd, stdout=PIPE, shell=True)
  stdout = process.communicate()[0]

  # find beginning of data
  initializingIdx = stdout.find("Initializing")
  newLineIdx = stdout.find("\n", initializingIdx)
  stdout = stdout[newLineIdx+1:]

  totalGFlops = 0
  totalMs = 0
  numSamples = 0
  gflopList = []
  msList = []

  # skip first b/c warmup/lookup
  newLineIdx = stdout.find("\n")
  stdout = stdout[newLineIdx+1:]

  # parse every line of data
  while "\n" in stdout:
    newLineIdx = stdout.find("\n")
    lineString = stdout[:newLineIdx]
    splits = lineString.split(",")
    if len(splits) < 3:
      break
    gflopsString = splits[1].lstrip()
    msString = splits[3].lstrip()
    gflops = float(gflopsString)
    ms = float(msString)
    gflopList.append(gflops)
    msList.append(ms)
    
    # next line
    stdout = stdout[newLineIdx+1:]

  #gflops = totalGFlops / numSamples
  #ms = totalMs / numSamples
  return (gflopList, msList)

################################################################################
# Benchmark Problem Sizes
################################################################################
def TensileBenchmarkLibraryClient(userArgs):
  if len(userArgs) < 2:
    print "USAGE:   python TensileBenchmarkLibraryClient.py sizes.csv library_client_command" 
    print "Example: python TensileBenchmarkLibraryClient.py sizes.csv ./4_LibraryClient/build/client --function-idx 1 --num-benchmarks 100 --use-gpu-timer 0" 
    exit(-1)

  # parse problem sizes path
  problemSizesPath = os.path.realpath( userArgs[0] )
  print "ProblemSizesPath: ", problemSizesPath

  # parse library client command
  libraryClientCommandList = userArgs[1:]
  libraryClientCommand = " ".join(libraryClientCommandList)
  print "LibraryClientCommand: ", libraryClientCommand
  
  # read problem sizes file
  csvFileRaw = open(problemSizesPath, "r")
  csvFile = csv.reader(csvFileRaw)

  # print column headers
  numIndices = -1
  for row in csvFile:
    numIndices = len(row)
    firstRow = row
    break
  output = ""
  for i in range(0, numIndices):
    sizeStr = "size%u" % i
    output += "%6s, " % sizeStr
  output += "%9s, %9s, " % ( "medianGF", "medianMs" )
  output += "%9s, %9s, " % ( "meanGF", "meanMs" )
  output += "%9s, %9s" % ( "rstdGF", "rstdMs" )
  print output

  # benchmark each problem size
  for row in [firstRow]:
    (gflopList, msList) = BenchmarkProblemSize(libraryClientCommand, row)
    meanGFlops = mean(gflopList)
    meanMs = mean(msList)
    medianGFlops = median(gflopList)
    medianMs = median(msList)
    stddevGFlops = stddev(gflopList)
    stddevMs = stddev(msList)

    # format output
    output = ""
    for size in row:
      output += "%6u, " % int(size)
    output += "%9.2f, %9.4f, " % (medianGFlops, medianMs)
    output += "%9.2f, %9.4f, " % (meanGFlops, meanMs)
    output += "%9.5f, %9.5f" % (stddevGFlops, stddevMs)
    print output

  # benchmark each problem size
  for row in csvFile:
    (gflopList, msList) = BenchmarkProblemSize(libraryClientCommand, row)
    meanGFlops = mean(gflopList)
    meanMs = mean(msList)
    medianGFlops = median(gflopList)
    medianMs = median(msList)
    stddevGFlops = stddev(gflopList) / meanGFlops
    stddevMs = stddev(msList) / meanMs

    # format output
    output = ""
    for size in row:
      output += "%6u, " % int(size)
    output += "%9.2f, %9.4f, " % (medianGFlops, medianMs)
    output += "%9.2f, %9.4f, " % (meanGFlops, meanMs)
    output += "%9.5f, %9.5f" % (stddevGFlops, stddevMs)
    print output

def median(lst):
  sortedList = sorted(lst)
  return sortedList[len(sortedList)/2]

def mean(lst):
  total = 0
  for i in range(len(lst)):
    total += lst[i]
  return (total / len(lst))

def stddev(lst):
  total = 0
  mn = mean(lst)
  for i in range(len(lst)):
    total += pow((lst[i]-mn),2)
  return (total/(len(lst)-1))**0.5

# installed "tensileBenchmarkLibraryClient" command
def main():
  TensileBenchmarkLibraryClient(sys.argv[1:])

if __name__ == "__main__":
  main()

