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

import math

def nTiles (MT, Size):
    nTiles = float(Size)/float(MT)
    return math.ceil(nTiles)

def TileGranularity (MT, Tiles, Size):
    tileGranularity = (float(Size) / float(MT)) / float(Tiles)
    return tileGranularity

def TilesPerCu (nBatches, Tiles0, Tiles1, nCUs, GSU, LSU):
    tilesPerCU = float(nBatches * Tiles0 * Tiles1) / (float(nCUs) / float(GSU) / float(LSU))
    return tilesPerCU

def CuGranularity (TilesPerCu):
    cuGranularity = float(TilesPerCu) / float(math.ceil(TilesPerCu))
    return cuGranularity

def ComputeGranularity(MT0, MT1, M, N, nCUs, GSU, LSU, nBatches):
    nTiles0 = nTiles(MT0, M)
    nTiles1 = nTiles(MT1, N)
    tileGranularity0 = TileGranularity(MT0,nTiles0,M)
    tileGranularity1 = TileGranularity(MT1,nTiles1,N)
    tilesPerCU = TilesPerCu(nBatches,nTiles0,nTiles1,nCUs,GSU,LSU)
    cuGranularity = CuGranularity(tilesPerCU)
    return (nTiles0,nTiles1,tileGranularity0,tileGranularity1,tilesPerCU,cuGranularity)

def WavesPerSolution(WG0,WG1):
    wavesPerSolution = (WG0*WG1)/64
    return wavesPerSolution

def WavePerSimd(tilesPerCU, wavesPerSolution):
    wavesPerSimd = (tilesPerCU * wavesPerSolution) / 4
    return wavesPerSimd

def WaveGranularity(wavesPerSimd):
    waveGranularity = float(wavesPerSimd) / float(math.ceil(wavesPerSimd))
    return waveGranularity

def ComputeWaveGranularity(WG0, WG1, MT0, MT1, M, N, nCUs, GSU, LSU, nBatches):
    nTiles0 = nTiles(MT0, M)
    nTiles1 = nTiles(MT1, N)
    tilesPerCU = TilesPerCu(nBatches,nTiles0,nTiles1,nCUs,GSU,LSU)
    wavesPerSolution = WavesPerSolution(WG0,WG1)
    wavesPerSimd = WavePerSimd(tilesPerCU, wavesPerSolution)
    waveGranularity = WaveGranularity(wavesPerSimd)
    return waveGranularity