################################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
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
import os
import yaml

typeIndexToName = {0: "S", 1: "D", 4: "H", 7: "B"}

def parseArgs():
    argParser = argparse.ArgumentParser()

    h = {"inDir"   : "Directory containing input logic files", \
         "outDir"  : "Output directory for modified logic files", \
         "sclk"    : "SCLK frequency in MHz tuning was done at", \
         "specs"   : ".yaml file containing hardware specifications", \
         "per-cu"  : "If tuning was done per CU", \
         "name"    : "Name substring to filter which files are modified", \
         "mfma"    : "If MFMA instructions were used for tuning", \
         "mi50"    : "For vega20, if tuning was done on mi50"
    }

    argParser.add_argument("inDir", metavar="input-dir", type=str, help=h["inDir"])
    argParser.add_argument("outDir", metavar="output-dir", type=str, help=h["outDir"])
    argParser.add_argument("sclk", type=int, help=h["sclk"])
    argParser.add_argument("specs", metavar="hardware-specs", nargs="?", type=str, default="default_specs.yaml", help=h["specs"])
    argParser.add_argument("-p", "--per-cu", action="store_true", help=h["per-cu"])
    argParser.add_argument("-n", "--name", type=str, help=h["name"])
    argParser.add_argument("-m", "--mfma", action="store_true", help=h["mfma"])
    argParser.add_argument("--mi50", action="store_true", help=h["mi50"])

    return argParser.parse_args()

def allFiles(startDir):
    current = os.listdir(startDir)
    files = []
    for filename in current:
        fullPath = os.path.join(startDir, filename)
        if os.path.isdir(fullPath):
            files = files + allFiles(fullPath)
        else:
            files.append(fullPath)
    return files

# sclk: MHz
# alu: flops/cycle/CU
def peakGFlops(sclk, alu, numCUs):
    return (sclk / 1000) * alu * numCUs

def main():
    args = parseArgs()
    mfmaKey = "mfma" if args.mfma else "non_mfma"

    with open(args.specs) as y:
        specs = yaml.safe_load(y)

    try:
        os.makedirs(args.outDir)
    except OSError:
        pass

    files = allFiles(args.inDir)
    for f in files:
        if not args.name or args.name in f:
            print(f)
            with open(f) as y:

                data = yaml.safe_load(y)

                sched = data[1]
                type = typeIndexToName[data[4]["DataType"]]
                alu = specs["ALU"][sched][mfmaKey][type]

                # get CU count
                if args.per_cu:
                    numCUs = 1
                elif sched == "vega20":
                    gpu = "mi50" if args.mi50 else "mi60"
                    numCUs = specs[sched]["numCUs"][gpu]
                else:
                    numCUs = specs[sched]["numCUs"]

                peak = peakGFlops(args.sclk, alu, numCUs)

                # update each entry
                for entry in data[7]:
                    print("Size: ", entry[0])
                    eff = entry[1][1] / peak
                    entry[1][1] = round (100 * eff, 3)
                    print("Efficiency: ", entry[1][1])
                    print()

            fName = os.path.basename(f)
            outFile = os.path.join(args.outDir, fName)

            with open(outFile, "w") as y:
                yaml.safe_dump(data, y, default_flow_style=None)

if __name__ == "__main__":
    main()
