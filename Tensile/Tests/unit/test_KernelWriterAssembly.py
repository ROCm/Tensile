################################################################################
# Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
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

from Tensile.KernelWriterAssembly import KernelWriterAssembly

def test_occupancy():
    # numThreads = 256
    assert KernelWriterAssembly.getOccupancy(256,  10, 65536,   0) == 1
    assert KernelWriterAssembly.getOccupancy(256,  10, 16384, 128) == 2
    assert KernelWriterAssembly.getOccupancy(256,  65,  8192,  64) == 3
    assert KernelWriterAssembly.getOccupancy(256,  10, 65536) == 1
    assert KernelWriterAssembly.getOccupancy(256,  10, 32768) == 2
    assert KernelWriterAssembly.getOccupancy(256, 129, 32768) == 1
    assert KernelWriterAssembly.getOccupancy(256,  10, 16384) == 4
    assert KernelWriterAssembly.getOccupancy(256, 256, 32768) == 1

    # numThreads = 512
    assert KernelWriterAssembly.getOccupancy(512,  10, 65536,   0) == 1
    assert KernelWriterAssembly.getOccupancy(512,  10, 16384, 128) == 1
    assert KernelWriterAssembly.getOccupancy(512,  65,  8192,  64) == 1
    assert KernelWriterAssembly.getOccupancy(512,  10, 65536) == 1
    assert KernelWriterAssembly.getOccupancy(512,  10, 32768) == 2
    assert KernelWriterAssembly.getOccupancy(512, 129, 32768) == 0
    assert KernelWriterAssembly.getOccupancy(512,  10, 16384) == 4
    assert KernelWriterAssembly.getOccupancy(512, 256, 32768) == 0

def test_max_regs():
    # numThreads = 256
    assert KernelWriterAssembly.getMaxRegsForOccupancy(256,  10, 65536,   0) == 256
    assert KernelWriterAssembly.getMaxRegsForOccupancy(256,  10, 16384, 128) == 128
    assert KernelWriterAssembly.getMaxRegsForOccupancy(256,  65,  8192,  64) == 84
    assert KernelWriterAssembly.getMaxRegsForOccupancy(256,  10, 65536) == 256
    assert KernelWriterAssembly.getMaxRegsForOccupancy(256,  10, 32768) == 128
    assert KernelWriterAssembly.getMaxRegsForOccupancy(256, 129, 32768) == 256
    assert KernelWriterAssembly.getMaxRegsForOccupancy(256,  10, 16384) == 64
    assert KernelWriterAssembly.getMaxRegsForOccupancy(256, 256, 32768) == 256

    # numThreads = 512
    assert KernelWriterAssembly.getMaxRegsForOccupancy(512,  10, 65536,   0) == 128
    assert KernelWriterAssembly.getMaxRegsForOccupancy(512,  10, 16384, 128) == 128
    assert KernelWriterAssembly.getMaxRegsForOccupancy(512,  65,  8192,  64) == 128
    assert KernelWriterAssembly.getMaxRegsForOccupancy(512,  10, 65536) == 128
    assert KernelWriterAssembly.getMaxRegsForOccupancy(512,  10, 32768) == 64
    assert KernelWriterAssembly.getMaxRegsForOccupancy(512, 129, 32768) == 129
    assert KernelWriterAssembly.getMaxRegsForOccupancy(512,  10, 16384) == 32
    assert KernelWriterAssembly.getMaxRegsForOccupancy(512, 256, 32768) == 256

# test_occupancy()
# test_max_regs()
