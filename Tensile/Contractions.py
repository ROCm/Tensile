################################################################################
# Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import print_function

import itertools
import sys
import time
import yaml

import Hardware
import Properties

from SolutionStructs import Solution as OriginalSolution
from Utils import *

class ProblemType:
    StateKeys = ['operationIdentifier', 'aType', 'bType', 'cType', 'dType']
    class FreeIndex:
        StateKeys = ['a', 'b', 'ca', 'cb', 'da', 'db']

        def __init__(self, a=None, b=None, ca=None, cb=None, da=None, db=None):
            self.a = a
            self.b = b
            self.ca = ca
            self.cb = cb
            self.da = da
            self.db = db

    class BatchIndex:
        StateKeys = ['a', 'b', 'c', 'd']
        def __init__(self, a=None, b=None, c=None, d=None):
            self.a = a
            self.b = b
            self.c = c
            self.d = d

    class BoundIndex:
        StateKeys = ['a', 'b']
        def __init__(self, a=None, b=None):
            self.a = a
            self.b = b

    @classmethod
    def FromOriginalState(cls, d):
        indices = [None]*d['TotalIndices']
        freeIndices  = []
        batchIndices = []
        boundIndices = []

        for i in d['IndicesBatch']:
            bi = cls.BatchIndex(c=i, d=i)
            indices[i] = bi
            batchIndices.append(bi)

        for i in d['IndicesSummation']:
            bi = cls.BoundIndex()
            indices[i] = bi
            boundIndices.append(bi)

        for idx in range(0, len(d['IndicesFree']), 2):
            ia = d['IndicesFree'][idx]
            ib = d['IndicesFree'][idx+1]
            fi = cls.FreeIndex(ca=ia, cb=ib, da=ia, db=ib)

            indices[ia] = fi
            indices[ib] = fi
            freeIndices.append(fi)

        for ia, ic in enumerate(d['IndexAssignmentsA']):
            indices[ic].a = ia

        for ib, ic in enumerate(d['IndexAssignmentsB']):
            indices[ic].b = ib

        for idx in indices:
            assert idx is not None
            idxState = state(idx)
            for (key, value) in idxState.items():
                assert value is not None

        rv = cls()
        rv.freeIndices = freeIndices
        rv.batchIndices = batchIndices
        rv.boundIndices = boundIndices
        rv.aDims = len(d['IndexAssignmentsA'])
        rv.bDims = len(d['IndexAssignmentsB'])
        rv.cDims = d['NumIndicesC']
        rv.dDims = rv.cDims

        assert d['DataType'] == 0
        if 'DestDataType' in d:
            assert d['DestDataType'] == 0
        
        rv.aType = 'Float'
        rv.bType = 'Float'
        rv.cType = 'Float'
        rv.dType = 'Float'

        rv.batched = d['Batched']

        return rv

    def __init__(self, freeIndices=None, batchIndices=None, boundIndices=None, aDims=None, bDims=None, cDims=None, dDims=None):
        self.freeIndices  = freeIndices
        self.batchIndices = batchIndices
        self.boundIndices = boundIndices
        self.aDims = aDims
        self.bDims = bDims
        self.cDims = cDims
        self.dDims = dDims

    @property
    def indexNames(self):
        aNames = ['_'] * self.aDims
        bNames = ['_'] * self.bDims
        cNames = ['_'] * self.cDims

        allIndexNames = 'ijklmnopqrstuvwxyz'
        index = 0

        dNames = list([allIndexNames[index+i] for i in range(self.cDims)])
        index += len(dNames)

        sumNames = list([allIndexNames[index+i] for i in range(len(self.boundIndices))])
        index += len(sumNames)

        for free in self.freeIndices:
            aNames[free.a ] = dNames[free.da]
            bNames[free.b ] = dNames[free.db]
            cNames[free.ca] = dNames[free.da]
            cNames[free.cb] = dNames[free.db]

        for batch in self.batchIndices:
            name = dNames[batch.d]
            aNames[batch.a] = name
            bNames[batch.b] = name
            cNames[batch.c] = name

        for i, bound in enumerate(self.boundIndices):
            name = sumNames[i]
            aNames[bound.a] = name
            bNames[bound.b] = name

        aNames = ''.join(aNames)
        bNames = ''.join(bNames)
        cNames = ''.join(cNames)
        dNames = ''.join(dNames)
        sumNames = ''.join(sumNames)

        return (aNames, bNames, cNames, dNames, sumNames)

    @property
    def operationIdentifier(self):
        (aNames, bNames, cNames, dNames, sumNames) = self.indexNames

        return '_'.join(['Contraction', sumNames,
                         'A'+aNames,
                         'B'+bNames,
                         'C'+cNames,
                         'D'+dNames])

    def predicate(self, includeOperation=False, includeType=False):
        predicates = []

        if not self.batched:
            predicates.append(ProblemPredicate("BatchSizeEqual", index=0, value=1))

        if includeOperation:
            predicates.append(ProblemPredicate("OperationIdentifierEqual", value=self.operationIdentifier))

        if includeType:
            predicates.append(ProblemPredicate("TypesEqual", value=[self.aType, self.bType, self.cType, self.dType]))

        if len(predicates) == 0:
            return None

        if len(predicates) == 1:
            return predicates[0]

        return ProblemPredicate('And', value=predicates)


class ProblemPredicate(Properties.Predicate):
    @classmethod
    def FromOriginalKeyPair(cls, pair):
        (key, value) = pair
        if key == 'AssertMinApproxSize':
            if value == 1:
                return None
            elif value == 2:
                return cls('MaxProblemSizeGreaterThan', value=32)
            else:
                raise RuntimeError("Unknown Approx size: {}".format(value))

        if key.endswith('Multiple'):
            if value == 1:
                return None
            rv = cls(None, index=None, value=value)
            
            if key == "AssertFree0ElementMultiple":
                rv.tag = "FreeSizeAMultiple"
                rv.index = 0
                return rv
            elif key == "AssertFree1ElementMultiple":
                rv.tag = "FreeSizeBMultiple"
                rv.index = 0
                return rv
            elif key == "AssertSummationElementMultiple":
                rv.tag = "BoundSizeMultiple"
                rv.index = 0
                return rv
            else:
                raise RuntimeError("Unknown Multiple Value: {}".format(key))
        
        if key.startswith('Assert'):
            raise RuntimeError("Unknown assertion key: {}".format(key))

class SizeMapping:
    StateKeys = ['workGroup',
                 'macroTile',
                 'threadTile',
                 'depthU',
                 'staggerU',
                 'globalSplitU',
                 'staggerStrideShift',
                 'workGroupMapping']

    @classmethod
    def FromOriginalState(cls, d):
        return cls(workGroup          = d['WorkGroup'],
                   macroTile          = cls.ReadOriginalMacroTile(d),
                   threadTile         = d['ThreadTile'],
                   workGroupMapping   = d['WorkGroupMapping'],
                   staggerU           = d['StaggerU'],
                   depthU             = d['DepthU'],
                   globalSplitU       = d['GlobalSplitU'],
                   staggerStrideShift = d['_staggerStrideShift']
                   )

    @classmethod
    def ReadOriginalMacroTile(cls, d):
        rv = [1,1,1]
        rv[0] = d['MacroTile0']
        rv[1] = d['MacroTile1']
        return rv

    def __init__(self, **kwargs):
        for (key, value) in kwargs.iteritems():
            setattr(self, key, value)

class Solution:
    StateKeys = ['name',
                'problemType',
                'hardwarePredicate',
                'problemPredicate',
                'sizeMapping',
                'debugKernel',
                'info',
                'index']
    HiddenKeys = ['originalSolution']

    @classmethod
    def FromOriginalState(cls, d, deviceInfo):
        rv = cls()

        rv.name = d['SolutionNameMin']

        rv.problemType = ProblemType.FromOriginalState(d['ProblemType'])


        rv.problemPredicate = ProblemPredicate.FromOriginalState(d)

        if 'DebugKernel' in d:
            rv.debugKernel = d['DebugKernel']

        rv.index = d['SolutionIndex']

        rv.info = cls.ReadOriginalInfo(d)

        rv.sizeMapping = SizeMapping.FromOriginalState(d)

        if d['KernelLanguage'] == 'Assembly':
            d['ISA'] = tuple(map(int,deviceInfo[1][3:6]))
            #print(d['ISA'])
        else:
            d['ISA'] = (0,0,0)

        rv.originalSolution = OriginalSolution(d)

        return rv

    @classmethod
    def ReadOriginalInfo(cls, d):
        return dict([(key, str(value)) for (key, value) in d.items() if key != 'ProblemType'])

    def __init__(self, **kwargs):
        self.name = None
        self.problemType = None
        self.hardwarePredicate = Hardware.HardwarePredicate('TruePred')
        self.problemPredicate = ProblemPredicate('TruePred')
        self.sizeMapping = None
        self.debugKernel = False
        self.info = {}
        self.index = None

        for key, value in kwargs:
            if key not in Solution.StateKeys and key not in Solution.HiddenKeys:
                raise KeyError("{0} is not a property of Solution.".format(key))

            setattr(self, key, value)

