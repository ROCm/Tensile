################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# Usage:
# $ python3 rocblas-benchInputCreator.py [-v] [-i <init>] <lib logic dir> <output dir>

# creates the benchmark and verification files:
# $ python3 rocblas-benchInputCreator.py -v ../libLogics ./
# creates the benchmark and verification files with hpl initialization:
# $ python3 rocblas-benchInputCreator.py -v -i hpl ../libLogics ./
# creates the benchmark file:
# $ python3 rocblas-benchInputCreator.py ../libLogics ./

import argparse
import os
import yaml

typeIndexToName = {0: "f32_r", 1: "f64_r", 2: "f32_c", 3: "f64_c", 4: "f16_r", 5: "i8_r", 6: "i32_r", 7: "bf16_r", 8: "i8_r", 10: "f8_r", 11: "bf8_r", 12: "f8b8", 13: "b8f8"}

def parseArgs():
    argParser = argparse.ArgumentParser()

    h = {"libLogic" : "Input library logic file",
         "outDir"   : "Output directory for rocBLAS-bench yaml files",
         "verify"   : "Also output verify version of yaml files",
         "initial"  : "Matrix initialization: hpl, trig, int. The default is trig for non Int8 datatype, and int for Int8."
    }

    argParser.add_argument("libLogic", metavar="logic-file", type=str, help=h["libLogic"])
    argParser.add_argument("outDir", metavar="output-dir", type=str, help=h["outDir"])
    argParser.add_argument("--verify", "-v", action="store_true", help=h["verify"])
    argParser.add_argument("--initialization", "-i", action="store", type=str, default = 'trig',  help=h["initial"])

    return argParser.parse_args()

def getProblemType(problem):
    # transA/B, a/b/c/d/compute_type
    problemDict = {}

    if problem["ComplexConjugateA"]:
        problemDict["transA"] = "C"
    elif problem["TransposeA"]:
        problemDict["transA"] = "T"
    else:
        problemDict["transA"] = "N"

    if problem["ComplexConjugateB"]:
        problemDict["transB"] = "C"
    elif problem["TransposeB"]:
        problemDict["transB"] = "T"
    else:
        problemDict["transB"] = "N"

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
            problemDict["composite_compute_type"] = "f8_bf8_f32"
          elif (typeIndexToName[problem["DataType"]] =="b8f8" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for B8F8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["composite_compute_type"] = "bf8_f8_f32"
          elif (typeIndexToName[problem["DataType"]] =="f8_r" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for F8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["composite_compute_type"] = "f8_f8_f32"
          elif (typeIndexToName[problem["DataType"]] =="bf8_r" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for B8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["composite_compute_type"] = "bf8_bf8_f32"
          elif (typeIndexToName[problem["DataType"]] =="f8b8" and typeIndexToName[problem["DestDataType"]]=="f32_r"): # for B8SS
            problemDict["a_type"] = "f8_r"
            problemDict["b_type"] = "bf8_r"
            problemDict["composite_compute_type"] = "f32"
          elif (typeIndexToName[problem["DataType"]] =="b8f8" and typeIndexToName[problem["DestDataType"]]=="f32_r"): # for B8SS
            problemDict["a_type"] = "bf8_r"
            problemDict["b_type"] = "f8_r"
            problemDict["composite_compute_type"] = "f32"
          elif (typeIndexToName[problem["DataType"]] =="b8f8" and typeIndexToName[problem["DestDataType"]]=="bf8_r"): # for B8F8B8S
            problemDict["a_type"] = "bf8_r"
            problemDict["b_type"] = "f8_r"
            problemDict["composite_compute_type"] = "f32"
          elif (typeIndexToName[problem["DataType"]] =="f8b8" and typeIndexToName[problem["DestDataType"]]=="bf8_r"): # for F8B8B8S
            problemDict["a_type"] = "f8_r"
            problemDict["b_type"] = "bf8_r"
            problemDict["composite_compute_type"] = "f32"
          else:
            problemDict["composite_compute_type"] = "f32"
        else:
          problemDict["compute_type"] = compType
    else:
        if problemDict["a_type"] == "f16_r" and problem["HighPrecisionAccumulate"]:
            problemDict["compute_type"] = "f32_r"
        elif problem["DataType"] == 5:
            problemDict["compute_type"] = "i32_r"
        else:
            problemDict["compute_type"] = problemDict["a_type"]

    if "F32XdlMathOp" in problem and problem["F32XdlMathOp"]==9: # XF32
        problemDict["math_mode"] = 1

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

    if len(size)==8: # ld defined in the library logic
        sizeDict["ldc"] = size[4]
        sizeDict["ldd"] = size[5]
        sizeDict["lda"] = size[6]
        sizeDict["ldb"] = size[7]
    else: # no ld defined in the library logic
        sizeDict["ldc"] = size[0]
        sizeDict["ldd"] = size[0]
        if not transA and not transB: # NN
            sizeDict["lda"] = size[0]
            sizeDict["ldb"] = size[3]
        elif transA and not transB:   # TN
            sizeDict["lda"] = size[3]
            sizeDict["ldb"] = size[3]
        elif not transA and transB:   # NT
            sizeDict["lda"] = size[0]
            sizeDict["ldb"] = size[1]
        else:                         # TT
            sizeDict["lda"] = size[3]
            sizeDict["ldb"] = size[1]

    return sizeDict

def createYaml(args, outputfile, problem, sizeMappings, verify):
    bench = []
    benchStrided = []
    benchGeneralBatched = []

    # get GEMM function and matrix orientation - Fixed for each library
    problemParams = getProblemType(problem)
    transA = problem["TransposeA"]
    transB = problem["TransposeB"]
    
    # check if this is f8/b8:
    f8gemm = True if (problem["DataType"]>=10) else False
    
    if verify:
        otherParams = {"alpha": 1, "beta": 1, "iters": 1, "cold_iters": 0, "norm_check": 1}
    else:
        otherParams = {"alpha": 1, "beta": 1, "iters": 10, "cold_iters": 2}

    #initialization
    if (args.initialization=='hpl' and problemParams["a_type"]!="i8_r"):
        init = {"initialization": "hpl"}
    elif (args.initialization=='trig' and problemParams["a_type"]!="i8_r"):
        init = {"initialization": "trig_float"}
    elif args.initialization== 'int':
        init = {"initialization": "rand_int"}
    else:
      print(f"Initialization {args.initialization} is not allowed for int8 datatype. Initialization changed to rand_int.")
      init = {"initialization": "rand_int"}

    # check if the library is General Batched based on the library name
    generalBatched = True if "_GB.yaml" in os.path.split(args.libLogic)[-1] else False

    # create rocBLAS-bench call for each size in logic file
    for (size, _) in sizeMappings: # size[0] = M, size[1] = N, size[2] = batch_count, size[3] = K, size[4] = ldc, size[5] = ldd, size[6] = lda, size[7] = ldb
        params = {}
 
        if (not generalBatched and size[2] == 1 and not f8gemm):  # non-f8, non-batched gemm (serves both HPA and non-HPA)
            params["rocblas_function"] = "rocblas_gemm_ex"
        elif (not generalBatched and size[2] != 1 and not f8gemm): # non-f8, strided_batched gemm (serves both HPA and non-HPA)
            params["rocblas_function"] = "rocblas_gemm_strided_batched_ex"
        elif not generalBatched: # f8
            params["rocblas_function"] = "rocblas_gemm_ex3"
        elif (generalBatched and not f8gemm):  # non-f8, general batched gemm (serves both HPA and non-HPA) currently there is no f8 general batched
            params["rocblas_function"] = "rocblas_gemm_batched_ex"
        else:
            raise RuntimeError(" F8 GEMM is not supporting General Batched.")

        sizeParams = getSizeParams(size, transA, transB)

        params.update(problemParams)
        params.update(sizeParams)
        params.update(otherParams)
        params.update(init)

        if (size[2] == 1 and not generalBatched):
            bench.append(params)
        elif (generalBatched):
            benchGeneralBatched.append(params)
        else:
            benchStrided.append(params)

    # output file names
    postfix = "_verify" if verify else "_bench"

    benchPath = os.path.join(args.outDir, outputfile + postfix + ".yaml")
    benchStridedPath = os.path.join(args.outDir, outputfile + postfix +"-strided.yaml")
    benchGeneralBatchedPath = os.path.join(args.outDir, outputfile + postfix+ "-general-batched.yaml")

    # write output
    if len(bench) > 0:
        with open(benchPath, "w") as f:
            yaml.safe_dump(bench, f, default_flow_style=None, sort_keys=False, width=5000)
            f.write(f"# End of {benchPath} \n")
    if len(benchStrided) > 0:
        with open(benchStridedPath, "w") as f:
            yaml.safe_dump(benchStrided, f, default_flow_style=None, sort_keys=False, width=5000)
            f.write(f"# End of {benchStrided} \n")
    if len(benchGeneralBatched) > 0:
        with open(benchGeneralBatchedPath, "w") as f:
            yaml.safe_dump(benchGeneralBatched, f, default_flow_style=None, sort_keys=False, width=5000)
            f.write(f"# End of {benchGeneralBatched} \n")

def main():
    args = parseArgs()

    if not (args.initialization in ['hpl', 'trig', 'int']):
        raise RuntimeError(f"Initialization {args.initialization} is not allowed. Choose from hpl, trig, or int.")

    for libname in os.listdir(args.libLogic):
        output = os.path.splitext(libname)[0]
        print(f" working on {output}")
        yamlName = os.path.join(args.libLogic,libname)
        with open(yamlName) as f:
            logicData = yaml.safe_load(f)

        try:
            os.makedirs(args.outDir)
        except OSError:
            pass

        problem = logicData[4]
        sizeMappings = logicData[7]

        createYaml(args, output, problem, sizeMappings, False)
        if args.verify:
            createYaml(args, output, problem, sizeMappings, True)

if __name__ == "__main__":
    main()

