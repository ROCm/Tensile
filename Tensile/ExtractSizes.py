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

import csv

from Tensile.TuningConfiguration import defaultHeader, libraryLogicMapper, appendSizes, appendThreadTiles, appendWorkGroups, generateDefaultScheme, generateEmptyBenchmarkGroup, generateProblemType, TuningConfiguration
from copy import deepcopy

rocblas_parameters = ["f","transposeA","transposeB","m","n","k","alpha","a_type","lda","stride_a","b_type","ldb","stride_b","beta","c_type","ldc","stride_c","d_type","ldd","stride_d","batch","compute_type","algo" ,"solution_index","flags"] #,"workspace_size" ]

gemm_ex_keys = ["-f", "--transposeA","--transposeB","-m","-n","-k","--alpha","--a_type","--lda","--b_type","--ldb","--beta","--c_type","--ldc","--d_type","--ldd","--compute_type","--algo","--solution_index","--flags"] #,"--workspace_size"]
gemm_keys = ["-f","-r","--transposeA","--transposeB","-m","-n","-k","--alpha","--lda","--ldb","--beta","--ldc"]

gemm_strided_batched_ex_keys = ["-f","--transposeA","--transposeB","-m","-n","-k","--alpha","--a_type","--lda","--stride_a","--b_type","--ldb","--stride_b","--beta","--c_type","--ldc","--stride_c","--d_type","--ldd","--stride_d","--batch","--compute_type","--algo","--solution_index","--flags"]#,"--workspace_size"]
gemm_strided_batched_keys = ["-f","-r","--transposeA","--transposeB","-m","-n","-k","--alpha","--lda","--stride_a","--ldb","--stride_b","--beta","--ldc","--stride_c","--batch"]

rocblas_key_mapping = {"gemm_ex":gemm_ex_keys, "gemm":gemm_keys, "gemm_strided_batched_ex":gemm_strided_batched_ex_keys, "gemm_strided_batched":gemm_strided_batched_keys}

def GetRocBLASParser():

    lineParser = argparse.ArgumentParser()

    lineParser.add_argument("-f",dest="f", type=str)
    lineParser.add_argument("-r",dest="r", type=str)
    lineParser.add_argument("--transposeA",dest="transposeA", type=str)
    lineParser.add_argument("--transposeB",dest="transposeB", type=str)
    lineParser.add_argument("-m",dest="m", type=str)
    lineParser.add_argument("-n",dest="n", type=str)
    lineParser.add_argument("-k",dest="k", type=str)
    lineParser.add_argument("--batch",dest="batch", type=int,default=1)
    lineParser.add_argument("--a_type",dest="a_type", type=str)
    lineParser.add_argument("--b_type",dest="b_type", type=str)
    lineParser.add_argument("--c_type",dest="c_type", type=str)
    lineParser.add_argument("--d_type",dest="d_type", type=str)
    lineParser.add_argument("--compute_type",dest="compute_type", type=str)
    lineParser.add_argument("--alpha",dest="alpha", type=float,default=1.0)
    lineParser.add_argument("--beta",dest="beta", type=float,default=0.0)
    lineParser.add_argument("--lda",dest="lda", type=int,default=1)
    lineParser.add_argument("--ldb",dest="ldb", type=int,default=1)
    lineParser.add_argument("--ldc",dest="ldc", type=int,default=1)
    lineParser.add_argument("--ldd",dest="ldd", type=int,default=1)
    lineParser.add_argument("--stride_a",dest="stride_a", type=int,default=0)
    lineParser.add_argument("--stride_b",dest="stride_b", type=int,default=0)
    lineParser.add_argument("--stride_c",dest="stride_c", type=int,default=0)
    lineParser.add_argument("--stride_d",dest="stride_d", type=int,default=0)
    lineParser.add_argument("--algo",dest="algo", type=int,default=0)
    lineParser.add_argument("--solution_index",dest="solution_index", type=int,default=0)
    lineParser.add_argument("--flags",dest="flags", type=int,default=0)

    return lineParser


def GetInceptionParser():

    argParser = argparse.ArgumentParser()

    argParser.add_argument("--verification_cache","-C",dest="verification_cache",help="Use specified directory to cache verification data. Off by default.",type=int,default=0)
    argParser.add_argument("--dout_data","-D",dest="dout_data",help="dy data filename for backward weight computation (Default=,type=str)",type=str)
    argParser.add_argument("--forw","-F",dest="forw",help="Run only Forward Convolution (Default=0,type=str)",type=int,default=0)
    argParser.add_argument("--in_h","-H",dest="in_h",help="Input Height (Default=32,type=str)",type=int,default=32)
    argParser.add_argument("--printconv","-P",dest="printconv",help="Print Convolution Dimensions (Default=1,type=str)",type=int,default=1)
    argParser.add_argument("--verify","-V",dest="verify",help="Verify Each Layer (Default=1,type=str)",type=int,default=1)
    argParser.add_argument("--in_w","-W",dest="in_w",help="Input Width (Default=32,type=str)",type=int,default=32)
    argParser.add_argument("--in_bias","-a",dest="in_bias",help="Input bias filename (Default=,type=str)",type=str)
    argParser.add_argument("--bias","-b",dest="bias",help="Use Bias (Default=0,type=str)",type=int,default=0)
    argParser.add_argument("--in_channels","-c",dest="in_channels",help="Number of Input Channels (Default=3,type=str)",type=int,default=3)
    argParser.add_argument("--in_data","-d",dest="in_data",help="Input data filename (Default=,type=str)",type=str)
    argParser.add_argument("--weights","-e",dest="weights",help="Input weights filename (Default=,type=str)",type=str)
    argParser.add_argument("--group_count","-g",dest="group_count",help="Number of Groups (Default=1,type=str)",type=int,default=1)
    argParser.add_argument("--iter","-i",dest="iter",help="Number of Iterations (Default=10,type=str)",type=int,default=10)
    argParser.add_argument("--dilation_w","-j",dest="dilation_w",help="Dilation of Filter Width (Default=1,type=str)",type=int,default=1)
    argParser.add_argument("--out_channels","-k",dest="out_channels",help="Number of Output Channels (Default=32,type=str)",type=int,default=32)
    argParser.add_argument("--dilation_h","-l",dest="dilation_h",help="Dilation of Filter Height (Default=1,type=str)",type=int,default=1)
    argParser.add_argument("--mode","-m",dest="mode",help="Convolution Mode (conv, trans, group, dw,type=str) (Default=conv,type=str)",type=str,default="conv")
    argParser.add_argument("--batchsize","-n",dest="batchsize",help="Mini-batch size (Default=100,type=str)",type=int,default=100)
    argParser.add_argument("--dump_output","-o",dest="dump_output",help="Dumps the output buffers (Default=0,type=str)",type=int,default=0)
    argParser.add_argument("--pad_h","-p",dest="pad_h",help="Zero Padding Height (Default=0,type=str)",type=int,default=0)
    argParser.add_argument("--pad_w","-q",dest="pad_w",help="Zero Padding Width (Default=0,type=str)",type=int,default=0)
    argParser.add_argument("--pad_val","-r",dest="pad_val",help="Padding Value (Default=0,type=str)",type=int,default=0)
    argParser.add_argument("--search","-s",dest="search",help="Search Kernel Config (Default=0,type=str)",type=int,default=0)
    argParser.add_argument("--time","-t",dest="time",help="Time Each Layer (Default=0,type=str)",type=int,default=1)
    argParser.add_argument("--conv_stride_0","-u",dest="conv_stride_0",help="Convolution Stride Vertical (Default=1,type=str)",type=int,default=1)
    argParser.add_argument("--conv_stride_1","-v",dest="conv_stride_1",help="Convolution Stride Horizontal (Default=1,type=str)",type=int,default=1)
    argParser.add_argument("--wall","-w",dest="wall",help="Wall-clock Time Each Layer, Requires time == 1 (Default=0,type=str)",type=int,default=0)
    argParser.add_argument("--fil_w","-x",dest="fil_w",help="Filter Width (Default=3,type=str)",type=int,default=3)
    argParser.add_argument("--fil_h","-y",dest="fil_h",help="Filter Height (Default=3,type=str)",type=int,default=3)
    argParser.add_argument("--pad_mode","-z",dest="pad_mode",help="Padding Mode (same, valid, default,type=str) (Default=default,type=str)",type=str,default="default")

    return argParser

def GenCommon(parameters):
    #--compute_type f32_r --algo 0 --solution_index 0 --flags 0 --workspace_size 0
    parameters["a_type"] = "f32_r"
    parameters["b_type"] = "f32_r"
    parameters["c_type"] = "f32_r"
    parameters["d_type"] = "f32_r"
    parameters["compute_type"] = "f32_r"

    parameters["algo"] = 0
    parameters["solution_index"] = 0
    parameters["flags"] = 0
    parameters["workspace_size"] = 0

def GenConvolutionBackwardWeightsConv1x1(input,weights,convolution,output):

    input_n = input["in_n"]
    input_c = input["in_c"]
    input_h = input["in_h"]
    input_w = input["in_w"]

    filter_k = weights["wei_n"]
    filter_c = weights["wei_c"] / weights["group_count"]
    filter_h = weights["wei_h"]
    filter_w = weights["wei_w"]

    dilation_h = convolution["dilation_h"]
    dilation_w = convolution["dilation_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    u = convolution["u"]
    v = convolution["v"]

    output_n = input_n
    output_c = filter_k
    output_h = max(1, (input_h - (1 + dilation_h * (filter_h - 1)) + 2 * pad_h) / u + 1)
    output_w = max(1, (input_w - (1 + dilation_w * (filter_w - 1)) + 2 * pad_w) / v + 1)

    m = filter_k 
    n = input_c
    k = input_h * input_w
    lda = k
    ldb = k
    ldc = n
    batch_count = 1 
    strideA = 0
    strideB = 0
    strideC = 0
    alpha = 1.
    beta = 1.

    problemDefinition = {}
    problemDefinition["f"] = "gemm_ex"
    problemDefinition["transposeA"] = "T"
    problemDefinition["transposeB"] = "N"
    problemDefinition["m"] = n
    problemDefinition["n"] = m
    problemDefinition["k"] = k
    problemDefinition["batch"] = batch_count
    problemDefinition["lda"] = ldb
    problemDefinition["ldb"] = lda
    problemDefinition["ldc"] = ldc
    problemDefinition["ldd"] = ldc
    problemDefinition["stride_a"] = strideB
    problemDefinition["stride_b"] = strideA
    problemDefinition["stride_c"] = strideC
    problemDefinition["stride_d"] = strideC
    problemDefinition["alpha"] = alpha
    problemDefinition["beta"] = beta

    GenCommon(problemDefinition)

    return problemDefinition

    
def GenConvolutionBackwardWeights(input,weights,convolution,output):

    input_n = input["in_n"]
    input_c = input["in_c"]
    input_h = input["in_h"]
    input_w = input["in_w"]

    filter_k = weights["wei_n"]
    filter_c = weights["wei_c"] / weights["group_count"]
    filter_h = weights["wei_h"]
    filter_w = weights["wei_w"]

    dilation_h = convolution["dilation_h"]
    dilation_w = convolution["dilation_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    u = convolution["u"]
    v = convolution["v"]

    output_n = input_n
    output_c = filter_k
    output_h = max(1, (input_h - (1 + dilation_h * (filter_h - 1)) + 2 * pad_h) / u + 1)
    output_w = max(1, (input_w - (1 + dilation_w * (filter_w - 1)) + 2 * pad_w) / v + 1)

    m = filter_k 
    n = input_c * filter_h * filter_w
    k = output_h * output_w
    lda = k
    ldb = k
    ldc = n
    batch_count = 1
    strideA = 0
    strideB = 0
    strideC = 0
    alpha = 1.
    beta = 1.

    problemDefinition = {}
    problemDefinition["f"] = "gemm_ex"
    problemDefinition["transposeA"] = "T"
    problemDefinition["transposeB"] = "N"
    problemDefinition["m"] = n
    problemDefinition["n"] = m
    problemDefinition["k"] = k
    problemDefinition["batch"] = batch_count
    problemDefinition["lda"] = ldb
    problemDefinition["ldb"] = lda
    problemDefinition["ldc"] = ldc
    problemDefinition["ldd"] = ldc
    problemDefinition["stride_a"] = strideB
    problemDefinition["stride_b"] = strideA
    problemDefinition["stride_c"] = strideC
    problemDefinition["stride_d"] = strideC
    problemDefinition["alpha"] = alpha
    problemDefinition["beta"] = beta

    GenCommon(problemDefinition)

    return problemDefinition


def GenConvolutionBackwardWeightsDefinition(input,weights,convolution,output):

    fil_h = weights["wei_h"]
    fil_w = weights["wei_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    conv_stride_0 = convolution["u"]
    conv_stride_1 = convolution["v"]

    problemDefinition = None

    if (fil_h == 1 and fil_w == 1) and (pad_h == 0 and pad_w == 0) and (conv_stride_0 == 1 and conv_stride_1 == 1):
        problemDefinition = GenConvolutionBackwardWeightsConv1x1(input,weights,convolution,output)
    else:
        problemDefinition = GenConvolutionBackwardWeights(input,weights,convolution,output)

    return problemDefinition

def GenConvolutionBackwardDataConv1x1(input,weights,convolution,output):

    input_n = input["in_n"]
    input_c = input["in_c"]
    input_h = input["in_h"]
    input_w = input["in_w"]

    filter_k = weights["wei_n"]
    filter_c = weights["wei_c"] / weights["group_count"]
    filter_h = weights["wei_h"]
    filter_w = weights["wei_w"]

    dilation_h = convolution["dilation_h"]
    dilation_w = convolution["dilation_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    u = convolution["u"]
    v = convolution["v"]

    output_n = input_n
    output_c = filter_k
    output_h = max(1, (input_h - (1 + dilation_h * (filter_h - 1)) + 2 * pad_h) / u + 1)
    output_w = max(1, (input_w - (1 + dilation_w * (filter_w - 1)) + 2 * pad_w) / v + 1)

    m = input_c
    n = input_h * input_w
    k = filter_k 
    lda = m
    ldb = n
    ldc = n
    batch_count = input_n
    strideA = 0
    strideB = k * n
    strideC = m * n
    alpha = 1.
    beta = 0.

    problemDefinition = {}
    problemDefinition["f"] = "gemm_strided_batched_ex"
    problemDefinition["transposeA"] = "N"
    problemDefinition["transposeB"] = "T"
    problemDefinition["m"] = n
    problemDefinition["n"] = m
    problemDefinition["k"] = k
    problemDefinition["batch"] = batch_count
    problemDefinition["lda"] = ldb
    problemDefinition["ldb"] = lda
    problemDefinition["ldc"] = ldc
    problemDefinition["ldd"] = ldc
    problemDefinition["stride_a"] = strideB
    problemDefinition["stride_b"] = strideA
    problemDefinition["stride_c"] = strideC
    problemDefinition["stride_d"] = strideC
    problemDefinition["alpha"] = alpha
    problemDefinition["beta"] = beta

    GenCommon(problemDefinition)

    return problemDefinition

def GenConvolutionBackwardData(input,weights,convolution,output):

    input_n = input["in_n"]
    input_c = input["in_c"]
    input_h = input["in_h"]
    input_w = input["in_w"]

    filter_k = weights["wei_n"]
    filter_c = weights["wei_c"] / weights["group_count"]
    filter_h = weights["wei_h"]
    filter_w = weights["wei_w"]

    dilation_h = convolution["dilation_h"]
    dilation_w = convolution["dilation_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    u = convolution["u"]
    v = convolution["v"]

    output_n = input_n
    output_c = filter_k
    output_h = max(1, (input_h - (1 + dilation_h * (filter_h - 1)) + 2 * pad_h) / u + 1)
    output_w = max(1, (input_w - (1 + dilation_w * (filter_w - 1)) + 2 * pad_w) / v + 1)


# found in MIOpen CreateGemmDescriptorConvBwdData call
    m = input_c * filter_h * filter_w 
    n =  output_h * output_w
    k = filter_k
    lda = m
    ldb = n
    ldc = n
    batch_count = 1
    strideA = 0
    strideB = 0
    strideC = 0
    alpha = 1.
    beta = 0.

    problemDefinition = {}
    problemDefinition["f"] = "gemm_ex"
    problemDefinition["transposeA"] = "N"
    problemDefinition["transposeB"] = "T"
    problemDefinition["m"] = n
    problemDefinition["n"] = m
    problemDefinition["k"] = k
    problemDefinition["batch"] = batch_count
    problemDefinition["lda"] = ldb
    problemDefinition["ldb"] = lda
    problemDefinition["ldc"] = ldc
    problemDefinition["ldd"] = ldc
    problemDefinition["stride_a"] = strideB
    problemDefinition["stride_b"] = strideA
    problemDefinition["stride_c"] = strideC
    problemDefinition["stride_d"] = strideC
    problemDefinition["alpha"] = alpha
    problemDefinition["beta"] = beta

    GenCommon(problemDefinition)

    return problemDefinition

def GenConvolutionBackwardDataDefinition(input,weights,convolution,output):


    fil_h = weights["wei_h"]
    fil_w = weights["wei_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    conv_stride_0 = convolution["u"]
    conv_stride_1 = convolution["v"]

    problemDefinition = None

    if (fil_h == 1 and fil_w == 1) and (pad_h == 0 and pad_w == 0) and (conv_stride_0 == 1 and conv_stride_1 == 1):
        problemDefinition = GenConvolutionBackwardDataConv1x1(input,weights,convolution,output)
    else:
        problemDefinition = GenConvolutionBackwardData(input,weights,convolution,output)

    return problemDefinition
        
def GenConvolutionForwardCNHWFwd(input,weights,convolution,output):

    input_n = input["in_n"]
    input_c = input["in_c"]
    input_h = input["in_h"]
    input_w = input["in_w"]

    filter_k = weights["wei_n"]
    filter_c = weights["wei_c"] / weights["group_count"]
    filter_h = weights["wei_h"]
    filter_w = weights["wei_w"]

    dilation_h = convolution["dilation_h"]
    dilation_w = convolution["dilation_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    u = convolution["u"]
    v = convolution["v"]

    output_n = input_n
    output_c = filter_k
    output_h = max(1, (input_h - (1 + dilation_h * (filter_h - 1)) + 2 * pad_h) / u + 1)
    output_w = max(1, (input_w - (1 + dilation_w * (filter_w - 1)) + 2 * pad_w) / v + 1)

    m = filter_k #wei_n
    n = input_n * output_h * output_w
    k = input_c
    lda = k
    ldb = n
    ldc = n
    batch_count = 1
    strideA = 0
    strideB = 0
    strideC = 0
    alpha = 1.
    beta = 0.

    problemDefinition = {}
    problemDefinition["f"] = "gemm_ex"
    problemDefinition["transposeA"] = "N"
    problemDefinition["transposeB"] = "N"
    problemDefinition["m"] = n
    problemDefinition["n"] = m
    problemDefinition["k"] = k
    problemDefinition["batch"] = batch_count
    problemDefinition["lda"] = ldb
    problemDefinition["ldb"] = lda
    problemDefinition["ldc"] = ldc
    problemDefinition["ldd"] = ldc
    problemDefinition["stride_a"] = strideB
    problemDefinition["stride_b"] = strideA
    problemDefinition["stride_c"] = strideC
    problemDefinition["stride_d"] = strideC
    problemDefinition["alpha"] = alpha
    problemDefinition["beta"] = beta

    GenCommon(problemDefinition)

    return problemDefinition

def GenConvolutionForwardConv1x1(input,weights,convolution,output):

    input_n = input["in_n"]
    input_c = input["in_c"]
    input_h = input["in_h"]
    input_w = input["in_w"]

    filter_k = weights["wei_n"]
    filter_c = weights["wei_c"] / weights["group_count"]
    filter_h = weights["wei_h"]
    filter_w = weights["wei_w"]

    dilation_h = convolution["dilation_h"]
    dilation_w = convolution["dilation_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    u = convolution["u"]
    v = convolution["v"]

    output_n = input_n
    output_c = filter_k
    output_h = max(1, (input_h - (1 + dilation_h * (filter_h - 1)) + 2 * pad_h) / u + 1)
    output_w = max(1, (input_w - (1 + dilation_w * (filter_w - 1)) + 2 * pad_w) / v + 1)

    m = filter_k # =wei_n
    n = input_h * input_w
    k = input_c
    lda = k
    ldb = n
    ldc = n
    batch_count = input_n
    strideA = 0
    strideB = k * n
    strideC = m * n
    alpha = 1.
    beta = 0.

    problemDefinition = {}
    problemDefinition["f"] = "gemm_strided_batched_ex"
    problemDefinition["transposeA"] = "N"
    problemDefinition["transposeB"] = "N"
    problemDefinition["m"] = n
    problemDefinition["n"] = m
    problemDefinition["k"] = k
    problemDefinition["batch"] = batch_count
    problemDefinition["lda"] = ldb
    problemDefinition["ldb"] = lda
    problemDefinition["ldc"] = ldc
    problemDefinition["ldd"] = ldc
    problemDefinition["stride_a"] = strideB
    problemDefinition["stride_b"] = strideA
    problemDefinition["stride_c"] = strideC
    problemDefinition["stride_d"] = strideC
    problemDefinition["alpha"] = alpha
    problemDefinition["beta"] = beta

    GenCommon(problemDefinition)

    return problemDefinition

def GenConvolutionForward(input,weights,convolution,output):

    input_n = input["in_n"]
    input_c = input["in_c"]
    input_h = input["in_h"]
    input_w = input["in_w"]

    filter_k = weights["wei_n"]
    filter_c = weights["wei_c"] / weights["group_count"]
    filter_h = weights["wei_h"]
    filter_w = weights["wei_w"]

    dilation_h = convolution["dilation_h"]
    dilation_w = convolution["dilation_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    u = convolution["u"]
    v = convolution["v"]

    output_n = input_n
    output_c = filter_k
    output_h = max(1, (input_h - (1 + dilation_h * (filter_h - 1)) + 2 * pad_h) / u + 1)
    output_w = max(1, (input_w - (1 + dilation_w * (filter_w - 1)) + 2 * pad_w) / v + 1)


# cound in MIOpen CreateGemmDescriptorConvFwd call
    m = filter_k 
    n =  output_h * output_w
    k = input_c * filter_h * filter_w
    lda = k
    ldb = n
    ldc = n
    batch_count = 1
    strideA = 0
    strideB = 0
    strideC = 0
    alpha = 1.
    beta = 0.

    problemDefinition = {}
    problemDefinition["f"] = "gemm_ex"
    problemDefinition["transposeA"] = "N"
    problemDefinition["transposeB"] = "N"
    problemDefinition["m"] = n
    problemDefinition["n"] = m
    problemDefinition["k"] = k
    problemDefinition["batch"] = batch_count
    problemDefinition["lda"] = ldb
    problemDefinition["ldb"] = lda
    problemDefinition["ldc"] = ldc
    problemDefinition["ldd"] = ldc
    problemDefinition["stride_a"] = strideB
    problemDefinition["stride_b"] = strideA
    problemDefinition["stride_c"] = strideC
    problemDefinition["stride_d"] = strideC
    problemDefinition["alpha"] = alpha
    problemDefinition["beta"] = beta

    GenCommon(problemDefinition)

    return problemDefinition

def GenConvolutionForwardDefinition(input,weights,convolution,output):

    fil_h = weights["wei_h"]
    fil_w = weights["wei_w"]
    in_h = input["in_h"]
    in_w = input["in_w"]
    pad_h = convolution["pad_h"]
    pad_w = convolution["pad_w"]
    conv_stride_0 = convolution["u"]
    conv_stride_1 = convolution["v"]

    problemDefinition = None

    if (fil_h == 1 and fil_w == 1) and (in_h < 14 and in_w < 14) and (pad_h == 0 and pad_w == 0) and (conv_stride_0 == 1 and conv_stride_1 == 1):
        problemDefinition = GenConvolutionForwardCNHWFwd(input,weights,convolution,output)
    else:
        if (fil_h == 1 and fil_w == 1) and (pad_h == 0 and pad_w == 0) and (conv_stride_0 == 1 and conv_stride_1 == 1):
            problemDefinition = GenConvolutionForwardConv1x1(input,weights,convolution,output)
        else:
            problemDefinition = GenConvolutionForward(input,weights,convolution,output)

    return problemDefinition

def ExtractProblemDefinitions(parsedArgs):

    # whight tensor definition
    # found in conv_driver.cpp  ConvDriver<Tgpu, Tref, Tfile>::GetWeightTensorLengthsFromCmdLine
    wei_n = parsedArgs.out_channels
    wei_c = parsedArgs.in_channels
    wei_h = parsedArgs.fil_h
    wei_w = parsedArgs.fil_w
    group_count = parsedArgs.group_count
    weights = {"wei_n":wei_n, "wei_c":wei_c, "wei_h":wei_h, "wei_w":wei_w, "group_count":group_count}

    # convolution definition
    # found in conv_driver.cpp ConvDriver<Tgpu, Tref, Tfile>::SetConvDescriptorFromCmdLineArgs()
    in_h = parsedArgs.in_h
    in_w = parsedArgs.in_w
    pad_h = parsedArgs.pad_h
    pad_w = parsedArgs.pad_w
    u = parsedArgs.conv_stride_0
    v = parsedArgs.conv_stride_1
    dilation_h = parsedArgs.dilation_h
    dilation_w = parsedArgs.dilation_w
    out_c = parsedArgs.out_channels
    in_c = parsedArgs.in_channels
    in_n = parsedArgs.batchsize
    convolution = {"in_h":in_h, "in_w":in_w, "pad_h":pad_h, "pad_w":pad_w, "u":u, "v":v, "dilation_h":dilation_h, "dilation_w":dilation_w, "out_c":out_c, "in_c":in_c, "in_n":in_n}

    # input tensor definition
    # found in conv_driver.cpp ConvDriver<Tgpu, Tref, Tfile>::GetInputTensorLengthsFromCmdLine()

    input = {"in_n":in_n, "in_c":in_c, "in_h":in_h, "in_w":in_w}
    
    # output tensor definition
    # found in convolution.cpp ConvolutionDescriptor::GetForwardOutputTensor
    # wei_k = wei_n or first dimention of weight tensor
    output = {"in_n":in_n, "wei_k":wei_n}

    return input,weights,convolution,output 

def UpdateOutputMapping(mapper, problemDefinition):
    # "f","transposeA","transposeB" 
    f = problemDefinition["f"]
    transposeA = problemDefinition["transposeA"]
    transposeB = problemDefinition["transposeB"]
  
    key = (f,transposeA,transposeB) 
    
    lineDefinition = None

    if key in mapper:
        lineDefinitions = mapper[key]
    else:
        lineDefinitions = []
        mapper[key] = lineDefinitions

    if problemDefinition not in lineDefinitions:
        lineDefinitions.append(problemDefinition)

def ProcessFile(filename):

    parser = GetInceptionParser()
    rocblasParser = GetRocBLASParser()

    problemMapper = {}

    with open(filename) as logFile:
        for line in logFile:

            if "MIOpenDriver" in line:
                args=line.split(' ')
                parsedArgs, otherArgs = parser.parse_known_args(args)

                input,weight,convolution,output = ExtractProblemDefinitions(parsedArgs)
                problemDefinitionForward = GenConvolutionForwardDefinition(input,weight,convolution,output)
                UpdateOutputMapping(problemMapper, problemDefinitionForward)
                problemDefinitionBackwardData = GenConvolutionBackwardDataDefinition(input,weight,convolution,output)
                UpdateOutputMapping(problemMapper, problemDefinitionBackwardData)
                problemDefinitionBackwardWeights = GenConvolutionBackwardWeightsDefinition(input,weight,convolution,output)
                UpdateOutputMapping(problemMapper, problemDefinitionBackwardWeights)


            if "rocblas-bench" in line:
                args=line.split(' ')
                parsedArgs, otherArgs =  rocblasParser.parse_known_args(args)
                problemDefinition = vars(parsedArgs)
                UpdateOutputMapping(problemMapper, problemDefinition)

    return problemMapper

def GetOutputFileName1(outputPath, namePart, key, ext):
    function, transposeA, transposeB = key
    fileName = namePart

    if "strided" in function:
        fileName += "-strided-%s%s.%s" % (transposeA,transposeB,ext)
    else:
        fileName += "-%s%s.%s" % (transposeA,transposeB,ext)
    
    outputFileName = outputFileName = os.path.join(outputPath, fileName)
    return outputFileName

def GetTensileSize(problemDefinition):

    m = problemDefinition["m"]
    n = problemDefinition["n"]
    batch = problemDefinition["batch"]
    k = problemDefinition["k"] 

    size = "          - Exact: [ %s , %s , %s, %s ]" % (m,n,batch,k)
    return size

def BuildRocBLASBenchmarkCall(problemDefinition):
    
    f = problemDefinition["f"]
    keys = rocblas_key_mapping[f]

    rocblas_call = "./rocblas-bench"
    for key in keys:
        param = key.replace("-","")
        value = problemDefinition[param] 
        rocblas_call += " %s %s" % (key,value)

    return rocblas_call

def GetSize(problemDefinition):
    m = int(problemDefinition["m"])
    n = int(problemDefinition["n"])
    k = int(problemDefinition["k"])
    b = 1

    if "batch" in problemDefinition:
        b = int(problemDefinition["batch"])

    return [m, n, b, k]

def ClassifySize(size):
    m = size[0]
    n = size[1]
    b = size[2]
    k = size[3]

    sizeKey = "none"

    scale = m * n

    tiny = 32 * 32
    small = 128 * 128
    medium = 512 * 512

    if b > 1:
        sizeKey = "batch"

    elif (scale <= tiny):
        sizeKey = "tiny"

    elif (scale <= small):
        sizeKey = "small"

    elif (scale <= medium):
        sizeKey = "medium"

    else:
        sizeKey = "large"

    return sizeKey

def GetProblemType(key):
    _ , transposeA, transposeB = key

    initialParams = {}

    if transposeA == "N":
        initialParams["TransposeA"] = False
    else:
        initialParams["TransposeA"] = True

    if transposeB == "N":
        initialParams["TransposeB"] = False
    else:
        initialParams["TransposeB"] = True

    problemType = generateProblemType(initialParams)

    return problemType

def generateBenchmarkGroupFromScheme(scheme):
    benchmarkGroup = generateEmptyBenchmarkGroup()

    commonParams = []
    forkParams = []

    for key in scheme:
        value = scheme[key]
        if len(value) > 1:
            d = {}
            v = deepcopy(value)
            d[key] = v
            forkParams.append(d)
        else:
            d = {}
            v = deepcopy(value)
            d[key] = v
            commonParams.append(d)

    benchmarkGroup["ForkParameters"] = forkParams
    benchmarkGroup["BenchmarkCommonParameters"] = commonParams

    return benchmarkGroup

def updateProblemGroupFromKey(problemKey, sizeKey,problemGroup,sizeList):

    _ , transposeA, transposeB = problemKey
    
    transposeType = "%s%s" % (transposeA.lower(),transposeB.lower())
    benchmarkGroup = None

    scheme = generateDefaultScheme()
    
    if sizeKey == "batch":
        scheme["GlobalSplitU"] = [1]
        scheme["LdsPadA"] = [0, -1]
        scheme["LdsPadB"] = [0, -1]
        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme) 
        appendThreadTiles(benchmarkGroup, [[4,4],[4,2],[2,4],[4,8],[8,4],[8,8]])
        appendWorkGroups(benchmarkGroup, [[16,16,1],[16,8,1],[8,16,1]])
        appendSizes(benchmarkGroup, sizeList)
    elif sizeKey == "tiny":
        scheme["GlobalSplitU"] = [1,4]
        scheme["LdsPadA"] = [0, -1]
        scheme["LdsPadB"] = [0, -1]
        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme) 
        appendThreadTiles(benchmarkGroup, [[2,2],[4,2],[2,4]])
        appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[4,16,4],[16,4,4],[32,8,4],[8,32,4]])
        #appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[8,16,4],[16,8,2],[16,8,4],[8,8,1],
        #    [8,8,2],[8,8,4],[4,16,4],[16,4,4],[4,8,8],[8,4,8],[4,4,4],[4,4,8]])
        appendSizes(benchmarkGroup, sizeList)
    elif sizeKey == "small":
        scheme["GlobalSplitU"] = [1,8]
        scheme["LdsPadA"] = [0, -1]
        scheme["LdsPadB"] = [0, -1]
        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme)
        appendThreadTiles(benchmarkGroup, [[2,2],[4,2],[2,4],[4,4]])
        appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[4,16,4],[16,4,4]])
        #appendWorkGroups(benchmarkGroup, [[16,16,1],[8,8,4],[8,16,2],[8,16,4],[16,8,2],
        #    [16,8,4],[32,4,2],[4,32,2],[4,16,4],[16,4,4]])
        appendSizes(benchmarkGroup, sizeList)
    elif sizeKey == "medium":
        if transposeType == "tn":
            scheme["GlobalSplitU"] = [1,8]
            scheme["DepthU"] = [8, 16]
            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme) 
            appendThreadTiles(benchmarkGroup, [[2,2],[4,2],[2,4],[4,4]])
            appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[4,16,4],[16,4,4]])
            appendSizes(benchmarkGroup, sizeList)
        else:
            scheme["GlobalSplitU"] = [1,8]
            scheme["LdsPadA"] = [0, -1]
            scheme["LdsPadB"] = [0, -1]
            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme) 
            appendThreadTiles(benchmarkGroup, [[2,2],[4,2],[2,4],[4,4]])
            appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[4,16,4],[16,4,4]])
            #appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[8,8,4],[8,8,8],[4,16,4],[16,4,4],[32,8,2],[8,32,2]])
            appendSizes(benchmarkGroup, sizeList)
    else: #sizeKey == "large"
        scheme["GlobalSplitU"] = [1]
        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme) 
        appendThreadTiles(benchmarkGroup, [[4,4],[6,4],[4,6],[4,8],[8,4],[8,8]])
        appendWorkGroups(benchmarkGroup, [[16,16,1]])
        appendSizes(benchmarkGroup, sizeList)

    problemGroup.append(benchmarkGroup)

def OutputConfigs(problemMapper, configPath, outputName, library):

    keys = problemMapper.keys()

    configDefs = {}

    for key in keys:
        lineDefinitions = problemMapper[key]
        sizeMapper = {}
        for problemDefinition in lineDefinitions:
            size =  GetSize(problemDefinition)

            sizeKey = ClassifySize(size)
            if sizeKey not in sizeMapper:
                sizeMapper[sizeKey] = []
            sizeMapper[sizeKey].append(size)

        problemType = GetProblemType(key)
        dataType = problemType["DataType"].lower()
        operationType = problemType["OperationType"].lower()

        problemTypeName = "%s%s" % (dataType, operationType)

        _, transposeA, transposeB = key
        transpose = "%s%s" % (transposeA.lower(), transposeB.lower())
        problemKey = "%s_%s_%s" % (library, problemTypeName, transpose)
        configurationFileName = "%s_%s" % (problemKey,outputName)
        configurationFilePath = os.path.join(configPath, configurationFileName)


        newConfig = None 
        problemGroup = None

        if configurationFilePath in configDefs:
            newConfig = configDefs[configurationFilePath]
            problemGroup = newConfig.benchmarkProblems[0]
        else:
            newConfig = TuningConfiguration()
            newConfig.globalParameters = deepcopy(defaultHeader)
            newConfig.libraryLogic = deepcopy(libraryLogicMapper[library])
            newConfig.libraryClient = True
            problemGroup = [problemType]
            newConfig.benchmarkProblems = [problemGroup]
            configDefs[configurationFilePath] = newConfig

        for sizeKey in sizeMapper:
            sizeList = sizeMapper[sizeKey]
            updateProblemGroupFromKey(key,sizeKey,problemGroup,sizeList)
        
    for key in configDefs:
        newConfig = configDefs[key]
        newConfig.writeLibraryLogic(key)


def RunParseLog():

    userArgs = sys.argv[1:]

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input-file-name", "-i", help="configuration file path")
    argParser.add_argument("--output-path", "-p", help="the output path")
    argParser.add_argument("--output-file-name", "-o", help="the output file name")
    argParser.add_argument("--library", "-l", help="the library Logic name")

    args = argParser.parse_args(userArgs)

    inputFileName = args.input_file_name
    outputPath = args.output_path
    outputName = args.output_file_name
    library = args.library

    problemMapper = ProcessFile(inputFileName)

    OutputConfigs(problemMapper, outputPath, outputName, library)

    
