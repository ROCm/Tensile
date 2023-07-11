
################################################################################
#
# Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

# Generates rocblas-bench input files from the library logic files.
# sample command:
# $ python3 rocblas-benchInputCreator.py ../libLogics/aldebaran_Cijk_Ailk_Bjlk_BBS_BH.yaml ./ BSS_NT

import argparse
import os
import yaml


typeIndexToName = {0: "f32_r", 1: "f64_r", 2: "f32_c", 3: "f64_c", 4: "f16_r", 5: "i8_r", 6: "i32_r", 7: "bf16_r", 8: "i8_r", 10: "f8_r", 11: "bf8_r", 12: "f8b8", 13: "b8f8"}


def parseArgs():
    argParser = argparse.ArgumentParser()

    h = {"libLogic" : "Input library logic file",
         "outDir"   : "Output directory for rocBLAS-bench yaml files",
         "verify"   : "Also output verify version of yaml files",
         "outfile"  : "the name of output file"
    }

    argParser.add_argument("libLogic", metavar="logic-file", type=str, help=h["libLogic"])
    argParser.add_argument("outDir", metavar="output-dir", type=str, help=h["outDir"])
    argParser.add_argument("outfile", metavar="output-file", type=str, help=h["outfile"])
    argParser.add_argument("--verify", "-v", action="store_true", help=h["verify"])

    return argParser.parse_args()

def getProblemType(problem):
    # transA/B, a/b/c/d/compute_type
    problemDict = {}
    problemDict["transA"] = "T" if problem["TransposeA"] else "N"
    problemDict["transB"] = "T" if problem["TransposeB"] else "N"

    problemDict["a_type"] = typeIndexToName[problem["DataType"]]
    problemDict["b_type"] = typeIndexToName[problem["DataType"]]
    problemDict["c_type"] = typeIndexToName[problem["DestDataType"]]
    problemDict["d_type"] = typeIndexToName[problem["DestDataType"]]


    f8gemm = True if (problem["DataType"]>=10) else False # is it f8

    if "ComputeDataType" in problem:
        compType = typeIndexToName[problem["ComputeDataType"]]
        if f8gemm: # f8 gemm
          if (typeIndexToName[problem["DataType"]] =="f8b8" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for F8B8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["new_compute_type"] = "f8_bf8_f32"
          elif (typeIndexToName[problem["DataType"]] =="b8f8" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for B8F8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["new_compute_type"] = "bf8_f8_f32"
          elif (typeIndexToName[problem["DataType"]] =="f8_r" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for F8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["new_compute_type"] = "f8_f8_f32"
          elif (typeIndexToName[problem["DataType"]] =="bf8_r" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for B8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["new_compute_type"] = "bf8_bf8_f32"  
          elif (typeIndexToName[problem["DataType"]] =="f8b8" and typeIndexToName[problem["DestDataType"]]=="f32_r"): # for B8HS
            problemDict["a_type"] = "f8_r"
            problemDict["b_type"] = "bf8_r"
            problemDict["new_compute_type"] = "f32"  
          elif (typeIndexToName[problem["DataType"]] =="b8f8" and typeIndexToName[problem["DestDataType"]]=="f32_r"): # for B8HS
            problemDict["a_type"] = "bf8_r"
            problemDict["b_type"] = "f8_r"
            problemDict["new_compute_type"] = "f32"  
          else:
            problemDict["new_compute_type"] = "f32"
        else:
          problemDict["compute_type"] = compType
    else:
        if problemDict["a_type"] == "f16_r" and problem["HighPrecisionAccumulate"]:
            problemDict["compute_type"] = "f32_r"
        elif problem["DataType"] == 5:
            problemDict["compute_type"] = "i32_r"
        else:
            problemDict["compute_type"] = problemDict["a_type"]

    return problemDict

def getSizeParams(size, transA, transB):
    # M/N/K, batch_count, lda/b/c/d
    sizeDict = {}
    sizeDict["M"] = size[0]
    sizeDict["N"] = size[1]
    sizeDict["K"] = size[3]

    if size[2] != 1:
        sizeDict["batch_count"] = size[2]
        # rocBLAS-bench will handle setting default strides

    if not transA and not transB: # NN
        sizeDict["lda"] = size[0]
        sizeDict["ldb"] = size[3]
        sizeDict["ldc"] = size[0]
        sizeDict["ldd"] = size[0]
    elif transA and not transB:   # TN
        sizeDict["lda"] = size[3]
        sizeDict["ldb"] = size[3]
        sizeDict["ldc"] = size[0]
        sizeDict["ldd"] = size[0]
    elif not transA and transB:   # NT
        sizeDict["lda"] = size[0]
        sizeDict["ldb"] = size[1]
        sizeDict["ldc"] = size[0]
        sizeDict["ldd"] = size[0]
    else:                         # TT
        sizeDict["lda"] = size[3]
        sizeDict["ldb"] = size[1]
        sizeDict["ldc"] = size[0]
        sizeDict["ldd"] = size[0]

    return sizeDict

def createYaml(args, problem, sizeMappings, verify):
    bench = []
    benchStrided = []

    # get GEMM fucnt and matrix orientation - Fixed for each library
    problemParams = getProblemType(problem)
    transA = problem["TransposeA"]
    transB = problem["TransposeB"]
    
    # check if this is f8/b8:
    f8gemm = True if (problem["DataType"]>=10) else False
    
    if verify:
        otherParams = {"alpha": 1, "beta": 1, "iters": 1, "cold_iters": 0, "norm_check": 1}
    else:
        otherParams = {"alpha": 1, "beta": 1, "iters": 10, "cold_iters": 2}

    # create rocBLAS-bench call for each size in logic file
    for (size, _) in sizeMappings: # size[0] = M, size[1] = N, size[2] = batch_count, size[3] = K
        params = {}
 
        if (size[2] == 1 and not f8gemm):  # non-f8, non-batched gemm (severes both HPA and non-HPA)
            params["rocblas_function"] = "rocblas_gemm_ex"
        elif (size[2] != 1 and not f8gemm): # non-f8, strided_batched gemm (severes both HPA and non-HPA)
            params["rocblas_function"] = "rocblas_gemm_strided_batched_ex"
        else: # f8
            params["rocblas_function"] = "rocblas_gemm_ex3"

        sizeParams = getSizeParams(size, transA, transB)

        params.update(problemParams)
        params.update(sizeParams)
        params.update(otherParams)

        if size[2] == 1:
            bench.append(params)
        else:
            benchStrided.append(params)

    # output file names
    prefix = args.outfile
    prefix += "_verify" if verify else ""

    benchPath = os.path.join(args.outDir, prefix + "_bench.yaml") 
    benchStridedPath = os.path.join(args.outDir, prefix +"bench-strided.yaml") 
    
    # write output
    if len(bench) > 0:
        with open(benchPath, "w") as f:
            yaml.safe_dump(bench, f, default_flow_style=None, sort_keys=False, width=5000)
    if len(benchStrided) > 0:
        with open(benchStridedPath, "w") as f:
            yaml.safe_dump(benchStrided, f, default_flow_style=None, sort_keys=False, width=5000)

def main():
    args = parseArgs()

    with open(args.libLogic) as f:
        logicData = yaml.safe_load(f)

    try:
        os.makedirs(args.outDir)
    except OSError:
        raise 

    problem = logicData[4]
    sizeMappings = logicData[7]

    createYaml(args, problem, sizeMappings, False)
    if args.verify:
        createYaml(args, problem, sizeMappings, True)

if __name__ == "__main__":
    main()
