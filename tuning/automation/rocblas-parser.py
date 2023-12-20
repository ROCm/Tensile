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

# reads the output of rocblas-bench with ROCBLAS_LAYER=2, and summarizes the sizes based on GEMM types.

# Usage
# python3 rocblas-parser.py rocblas-configs.log sizes.log

import os
import argparse

def gemmfinder(config):
    output ="unrecognized"
    # BBS
    if   (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "bf16_r"  and config.d_type == "bf16_r" and config.compute_type == "f32_r"): 
        output = "BBS_NN"
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "bf16_r"  and config.d_type == "bf16_r" and config.compute_type == "f32_r"): 
        output = "BBS_NT"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "bf16_r"  and config.d_type == "bf16_r" and config.compute_type == "f32_r"): 
        output = "BBS_TN"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "bf16_r"  and config.d_type == "bf16_r" and config.compute_type == "f32_r"): 
        output = "BBS_TT"
    # HHS
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f16_r"  and config.d_type == "f16_r" and config.compute_type == "f32_r"): 
        output = "HHS_NN"
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f16_r"  and config.d_type == "f16_r" and config.compute_type == "f32_r"): 
        output = "HHS_NT"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f16_r"  and config.d_type == "f16_r" and config.compute_type == "f32_r"): 
        output = "HHS_TN"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f16_r"  and config.d_type == "f16_r" and config.compute_type == "f32_r"): 
        output = "HHS_TT"
    # BSS
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "BSS_NN"
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "BSS_NT"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "BSS_TN"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "BSS_TT"
    # HSS
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "HSS_NN"
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "HSS_NT"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "HSS_TN"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "HSS_TT"
    # SGEMM
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "f32_r" and config.b_type == "f32_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r") or (config.f == "gemm" and config.transposeA == "N" and config.transposeB == "N" and config.r == "f32_r"): 
        output = "SGEMM_NN"
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "f32_r" and config.b_type == "f32_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r") or (config.f == "gemm" and config.transposeA == "N" and config.transposeB == "T" and config.r == "f32_r"): 
        output = "SGEMM_NT"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "f32_r" and config.b_type == "f32_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r") or (config.f == "gemm" and config.transposeA == "T" and config.transposeB == "N" and config.r == "f32_r"): 
        output = "SGEMM_TN"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "f32_r" and config.b_type == "f32_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r") or (config.f == "gemm" and config.transposeA == "T" and config.transposeB == "T" and config.r == "f32_r"): 
        output = "SGEMM_TT"
    # DGEMM
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "f64_r" and config.b_type == "f64_r"  and config.c_type == "f64_r"  and config.d_type == "f64_r" and config.compute_type == "f64_r") or (config.f == "gemm" and config.transposeA == "N" and config.transposeB == "N" and config.r == "f64_r"): 
        output = "DGEMM_NN"
    elif (config.f == "gemm_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "f64_r" and config.b_type == "f64_r"  and config.c_type == "f64_r"  and config.d_type == "f64_r" and config.compute_type == "f64_r") or (config.f == "gemm" and config.transposeA == "N" and config.transposeB == "T" and config.r == "f64_r"): 
        output = "DGEMM_NT"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "f64_r" and config.b_type == "f64_r"  and config.c_type == "f64_r"  and config.d_type == "f64_r" and config.compute_type == "f64_r") or (config.f == "gemm" and config.transposeA == "T" and config.transposeB == "N" and config.r == "f64_r"): 
        output = "DGEMM_TN"
    elif (config.f == "gemm_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "f64_r" and config.b_type == "f64_r"  and config.c_type == "f64_r"  and config.d_type == "f64_r" and config.compute_type == "f64_r") or (config.f == "gemm" and config.transposeA == "T" and config.transposeB == "T" and config.r == "f64_r"): 
        output = "DGEMM_TT"
    # strided_batched
    # BBS
    elif   (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "bf16_r"  and config.d_type == "bf16_r" and config.compute_type == "f32_r"): 
        output = "BBS_NN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "bf16_r"  and config.d_type == "bf16_r" and config.compute_type == "f32_r"): 
        output = "BBS_NT_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "bf16_r"  and config.d_type == "bf16_r" and config.compute_type == "f32_r"): 
        output = "BBS_TN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "bf16_r"  and config.d_type == "bf16_r" and config.compute_type == "f32_r"): 
        output = "BBS_TT_SB"
    # HHS
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f16_r"  and config.d_type == "f16_r" and config.compute_type == "f32_r"): 
        output = "HHS_NN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f16_r"  and config.d_type == "f16_r" and config.compute_type == "f32_r"): 
        output = "HHS_NT_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f16_r"  and config.d_type == "f16_r" and config.compute_type == "f32_r"): 
        output = "HHS_TN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f16_r"  and config.d_type == "f16_r" and config.compute_type == "f32_r"): 
        output = "HHS_TT_SB"
    # BSS
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "BSS_NN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "BSS_NT_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "BSS_TN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "bf16_r" and config.b_type == "bf16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "BSS_TT_SB"
    # HSS
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "HSS_NN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "HSS_NT_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "HSS_TN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "f16_r" and config.b_type == "f16_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
        output = "HSS_TT_SB"
    # SGEMM
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "f32_r" and config.b_type == "f32_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r") or (config.f == "gemm_strided_batched" and config.transposeA == "N" and config.transposeB == "N" and config.r == "f32_r"): 
        output = "SGEMM_NN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "f32_r" and config.b_type == "f32_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r") or (config.f == "gemm_strided_batched" and config.transposeA == "N" and config.transposeB == "T" and config.r == "f32_r"): 
        output = "SGEMM_NT_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "f32_r" and config.b_type == "f32_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r") or (config.f == "gemm_strided_batched" and config.transposeA == "T" and config.transposeB == "N" and config.r == "f32_r"): 
        output = "SGEMM_TN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "f32_r" and config.b_type == "f32_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r") or (config.f == "gemm_strided_batched" and config.transposeA == "T" and config.transposeB == "T" and config.r == "f32_r"): 
        output = "SGEMM_TT_SB"
    # DGEMM
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "N" and config.a_type == "f64_r" and config.b_type == "f64_r"  and config.c_type == "f64_r"  and config.d_type == "f64_r" and config.compute_type == "f64_r") or (config.f == "gemm_strided_batched" and config.transposeA == "N" and config.transposeB == "N" and config.r == "f64_r"): 
        output = "DGEMM_NN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "N" and config.transposeB == "T" and config.a_type == "f64_r" and config.b_type == "f64_r"  and config.c_type == "f64_r"  and config.d_type == "f64_r" and config.compute_type == "f64_r") or (config.f == "gemm_strided_batched" and config.transposeA == "N" and config.transposeB == "T" and config.r == "f64_r"): 
        output = "DGEMM_NT_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "N" and config.a_type == "f64_r" and config.b_type == "f64_r"  and config.c_type == "f64_r"  and config.d_type == "f64_r" and config.compute_type == "f64_r") or (config.f == "gemm_strided_batched" and config.transposeA == "T" and config.transposeB == "N" and config.r == "f64_r"): 
        output = "DGEMM_TN_SB"
    elif (config.f == "gemm_strided_batched_ex" and config.transposeA == "T" and config.transposeB == "T" and config.a_type == "f64_r" and config.b_type == "f64_r"  and config.c_type == "f64_r"  and config.d_type == "f64_r" and config.compute_type == "f64_r") or (config.f == "gemm_strided_batched" and config.transposeA == "T" and config.transposeB == "T" and config.r == "f64_r"): 
        output = "DGEMM_TT_SB"

    return output

def parseBenchCofnig():
    argParser = argparse.ArgumentParser()

    h = {"inputfile" : "input file",
         "outputfile" : "output file"
    }

    argParser.add_argument("input", metavar="logic-file", type=str, help=h["inputfile"])
    argParser.add_argument("output", metavar="output-dir", type=str, help=h["outputfile"])

    return argParser.parse_args()

def readfile(filename):

    argParser = argparse.ArgumentParser()


    argParser.add_argument("-f", action="store", type=str, default = '')
    argParser.add_argument("-r", action="store", type=str, default = '')
    
    argParser.add_argument("--transposeA", action="store", type=str, default = '')
    argParser.add_argument("--transposeB", action="store", type=str, default = '')
    
    argParser.add_argument("-m", action="store", type=int, default = '')
    argParser.add_argument("-n", action="store", type=int, default = '')
    argParser.add_argument("-k", action="store", type=int, default = '')
    
    argParser.add_argument("--a_type", action="store", type=str, default = '')
    argParser.add_argument("--b_type", action="store", type=str, default = '')
    argParser.add_argument("--c_type", action="store", type=str, default = '')
    argParser.add_argument("--d_type", action="store", type=str, default = '')
    
    argParser.add_argument("--batch_count",  action="store", type=str, default = '')
    argParser.add_argument("--compute_type", action="store", type=str, default = '')

    matrices = {}

    config = open(filename, 'r')
    Lines = config.readlines()

    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        config, unknown = argParser.parse_known_args(line.strip().split())
        gemmtype = gemmfinder(config)
        if not (gemmtype in matrices):
            matrices[gemmtype] = []
        if ("_SB" in gemmtype): # for strided
            matrices[gemmtype].append((config.m, config.n, config.batch_count, config.k)) ## batch
        else: 
          matrices[gemmtype].append((config.m, config.n, config.k)) ## batch
    return matrices

def sortfuc(size):
    return (size[0],size[1],size[2])

def sortfuc_strided(size):
    return (size[0],size[1],size[3], size[2])


def createOutput(filename, matrices):
  matrices = dict(sorted(matrices.items()))
  count = 0
  with open(filename,'w') as f:
    for gemmtype in matrices:
      count += len(matrices[gemmtype])
      f.write(f"\n\n--- {gemmtype}: {len(matrices[gemmtype])} \n")
      if ("_SB" in gemmtype): # for strided
        matrices[gemmtype].sort(key=sortfuc_strided)
      else:
        matrices[gemmtype].sort(key=sortfuc)
      for size in matrices[gemmtype]:
        if ("_SB" in gemmtype): # for strided
            f.write(f" - [ {size[0]}, {size[1]}, {size[2]}, {size[3]} ]\n")
        else:
            f.write(f" - [ {size[0]}, {size[1]}, 1, {size[2]} ]\n")
    (f" total is: {count}")
    f.write(f"\n -- total is: {count}")

def main():
    args = parseBenchCofnig()    

    # check if input exists
    if not os.path.isfile(args.input): 
      raise FileNotFoundError("{0} input file does not exist!".format(args.input))

    matrices = readfile(args.input)

    createOutput(args.output, matrices)
    
if __name__ == "__main__":
    main()

