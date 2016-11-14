/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

const unsigned int num_gemm_params = 5;

unsigned int gemm_params[][6] = {
  // M, N, K, transA, transB, cnt
  // { 1760,   16, 1760, 0, 0, 1 },
  // { 1760,   32, 1760, 0, 0, 1 },
  // { 1760,   64, 1760, 0, 0, 1 },
  // { 1760,  128, 1760, 0, 0, 1 },
  // { 1760, 7000, 1760, 0, 0, 1 },
  // { 2048,   16, 2048, 0, 0, 1 },
  // { 2048,   32, 2048, 0, 0, 1 },
  // { 2048,   64, 2048, 0, 0, 1 },
  // { 2048,  128, 2048, 0, 0, 1 },
  // { 2048, 7000, 2048, 0, 0, 1 },
  // { 2560,   16, 2560, 0, 0, 1 },
  // { 2560,   32, 2560, 0, 0, 1 },
  // { 2560,   64, 2560, 0, 0, 1 },
  // { 2560,  128, 2560, 0, 0, 1 },
  // { 2560, 7000, 2560, 0, 0, 1 },
  { 4096,   16, 4096, 0, 0, 1 },
  { 4096,   32, 4096, 0, 0, 1 },
  { 4096,   64, 4096, 0, 0, 1 },
  { 4096,  128, 4096, 0, 0, 1 },
  { 4096, 7000, 4096, 0, 0, 1 },
  { 1760,   16, 1760, 1, 0, 1 },
  { 1760,   32, 1760, 1, 0, 1 },
  { 1760,   64, 1760, 1, 0, 1 },
  { 1760,  128, 1760, 1, 0, 1 },
  { 1760, 7000, 1760, 1, 0, 1 },
  { 2048,   16, 2048, 1, 0, 1 },
  { 2048,   32, 2048, 1, 0, 1 },
  { 2048,   64, 2048, 1, 0, 1 },
  { 2048,  128, 2048, 1, 0, 1 },
  { 2048, 7000, 2048, 1, 0, 1 },
  { 2560,   16, 2560, 1, 0, 1 },
  { 2560,   32, 2560, 1, 0, 1 },
  { 2560,   64, 2560, 1, 0, 1 },
  { 2560,  128, 2560, 1, 0, 1 },
  { 2560, 7000, 2560, 1, 0, 1 },
  { 4096,   16, 4096, 1, 0, 1 },
  { 4096,   32, 4096, 1, 0, 1 },
  { 4096,   64, 4096, 1, 0, 1 },
  { 4096,  128, 4096, 1, 0, 1 },
  { 4096, 7000, 4096, 1, 0, 1 },
  { 1760, 7133, 1760, 0, 1, 1 },
  { 2048, 7133, 2048, 0, 1, 1 },
  { 2560, 7133, 2560, 0, 1, 1 },
  { 4096, 7133, 4096, 0, 1, 1 },
  { 5124, 9124, 1760, 0, 0, 1 },
  {   35, 8457, 1760, 0, 0, 1 },
  { 5124, 9124, 2048, 0, 0, 1 },
  {   35, 8457, 2048, 0, 0, 1 },
  { 5124, 9124, 2560, 0, 0, 1 },
  {   35, 8457, 2560, 0, 0, 1 },
  { 5124, 9124, 4096, 0, 0, 1 },
  {   35, 8457, 4096, 0, 0, 1 },
  { 5124, 9124, 1760, 1, 0, 1 },
  {   35, 8457, 1760, 1, 0, 1 },
  { 5124, 9124, 2048, 1, 0, 1 },
  {   35, 8457, 2048, 1, 0, 1 },
  { 5124, 9124, 2560, 1, 0, 1 },
  {   35, 8457, 2560, 1, 0, 1 },
  { 5124, 9124, 4096, 1, 0, 1 },
  {   35, 8457, 4096, 1, 0, 1 },
  { 7680,   16, 2560, 0, 0, 1 },
  { 7680,   32, 2560, 0, 0, 1 },
  { 7680,   64, 2560, 0, 0, 1 },
  { 7680,  128, 2560, 0, 0, 1 },
  { 7680,   16, 2560, 1, 0, 1 },
  { 7680,   32, 2560, 1, 0, 1 },
  { 7680,   64, 2560, 1, 0, 1 },
  { 7680,  128, 2560, 1, 0, 1 },
  { 3072,   16, 1024, 0, 0, 1 },
  { 3072,   32, 1024, 0, 0, 1 },
  { 3072,   64, 1024, 0, 0, 1 },
  { 3072,  128, 1024, 0, 0, 1 },
  { 3072,   16, 1024, 1, 0, 1 },
  { 3072,   32, 1024, 1, 0, 1 },
  { 3072,   64, 1024, 1, 0, 1 },
  { 3072,  128, 1024, 1, 0, 1 },
  { 3072, 7435, 1024, 0, 1, 1 },
  { 7680, 5481, 2560, 0, 1, 1 }
    };
