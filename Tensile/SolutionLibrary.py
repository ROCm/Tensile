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

import itertools
import Properties
import Hardware
import Contractions
from SolutionStructs import Solution as OriginalSolution
from Utils import *

class SingleSolutionLibrary:
    Tag = 'Single'

    def __init__(self, solution):
        self.solution = solution

    @property
    def tag(self):
        return self.__class__.Tag

    def state(self):
        return {'type': self.tag, 'index': self.solution.index}

class MatchingProperty(Properties.Property):
    pass

class MatchingLibrary:
    Tag = 'Matching'
    StateKeys = [('type', 'tag'), 'properties', 'table', 'distance']

    @classmethod
    def FromOriginalState(cls, d, solutions):
        indices = d[0]
        origTable = d[1]

        propertyKeys = {
                2:lambda: MatchingProperty('FreeSizeA', index=0),
                3:lambda: MatchingProperty('FreeSizeB', index=0),
                0:lambda: MatchingProperty('BatchSize', index=0),
                1:lambda: MatchingProperty('BoundSize', index=0)
            }

        properties = [propertyKeys[i]() for i in indices]

        table = []

        distance = {'type': 'Euclidean'}

        for row in origTable:
            try:
                index = row[1][0]
                value = SingleSolutionLibrary(solutions[index])
                entry = {'key': list(row[0]), 'value': value, 'speed': row[1][1]}
                table.append(entry)
            except KeyError:
                pass

        return cls(properties, table, distance)

    @property
    def tag(self):
        return self.__class__.Tag

    def __init__(self, properties, table, distance):
        self.properties = properties
        self.table = table
        self.distance = distance

class ProblemMapLibrary:
    Tag = 'ProblemMap'
    StateKeys = [('type', 'tag'), ('property', 'mappingProperty'), ('map', 'mapping')]

    def __init__(self, mappingProperty=None, mapping=None):
        self.mappingProperty = mappingProperty
        self.mapping = mapping

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        assert self.__class__ == other.__class__ and self.tag == other.tag and self.mappingProperty == other.mappingProperty

        for key,value in list(other.mapping.items()):
            if key in self.mapping:
                self.mapping[key].merge(value)
            else:
                self.mapping[key] = value

class PredicateLibrary:
    StateKeys = [('type', 'tag'), 'rows']

    def __init__(self, tag=None, rows=None):
        self.tag = tag
        self.rows = rows

    def merge(self, other):
        assert self.__class__ == other.__class__ and self.tag == other.tag

        rowdict = dict([(r['predicate'], i) for i,r in enumerate(self.rows)])

        for row in other.rows:
            if row['predicate'] in rowdict:
                myRownum = rowdict[row['predicate']]
                self.rows[myRownum]['library'].merge(row['library'])
            else:
                self.rows.append(row)

class MasterSolutionLibrary:
    StateKeys = ['solutions', 'library']

    @classmethod
    def FromOriginalState(cls, d, solutionClass=Contractions.Solution, libraryOrder = None):
        if libraryOrder is None:
            libraryOrder = ['Hardware', 'OperationIdentifier', 'Predicates', 'Matching']

        minVersion = d[0]
        deviceSection = d[1:4]
        origProblemType = d[4]
        origSolutions = d[5]
        origLibrary = d[6:8]

        problemType = Contractions.ProblemType.FromOriginalState(origProblemType)

        allSolutions = [solutionClass.FromOriginalState(s, deviceSection) for s in origSolutions]

        asmSolutions = dict([(s.index, s) for s in allSolutions if s.info['KernelLanguage'] != 'Source'])
        sourceSolutions = dict([(s.index, s) for s in allSolutions if s.info['KernelLanguage'] == 'Source'])

        matchingLibrary = MatchingLibrary.FromOriginalState(origLibrary, asmSolutions)

        for libName in reversed(libraryOrder):
            if libName == 'Matching':
                library = matchingLibrary

            elif libName == 'Hardware':
                newLib = PredicateLibrary(tag='Hardware', rows=[])
                pred = Hardware.HardwarePredicate.FromOriginalDeviceSection(deviceSection)
                newLib.rows.append({'predicate': pred, 'library': library})
                library = newLib

            elif libName == 'Predicates':
                pred = problemType.predicate(includeType=True)
                if pred is not None:
                    newLib = PredicateLibrary(tag='Problem', rows=[])
                    newLib.rows.append({'predicate': pred, 'library': library})
                    library = newLib

            elif libName == 'OperationIdentifier':
                operationID = problemType.operationIdentifier
                prop = MatchingProperty('OperationIdentifier')
                mapping = {operationID: library}
                newLib = ProblemMapLibrary(prop, mapping)
                library = newLib
            else:
                raise ValueError("Unknown value " + libName)

        rv = cls(asmSolutions, library)
        rv.sourceSolutions = sourceSolutions
        return rv

    def __init__(self, solutions, library):
        self.solutions = solutions
        self.library = library

    def state(self):
        return {'solutions': state(iter(list(self.solutions.values()))), 'library': state(self.library)}

    def applyNaming(self, naming=None):
        if naming is None:
            allSolutions = itertools.chain(iter(list(self.solutions.values())), iter(list(self.sourceSolutions.values())))
            kernels = list(itertools.chain(*[s.originalSolution.getKernels() for s in allSolutions]))
            naming = OriginalSolution.getMinNaming(kernels)

        for s in list(self.solutions.values()):
            s.name = OriginalSolution.getNameMin(s.originalSolution.getKernels()[0], naming)

    def merge(self, other):
        assert self.__class__ == other.__class__

        allIndices = itertools.chain(self.solutions, self.sourceSolutions)
        curIndex = max(allIndices) + 1

        for k,s in list(other.solutions.items()):
            s.index = curIndex
            self.solutions[curIndex] = s
            curIndex += 1

        for k,s in list(other.sourceSolutions.items()):
            s.index = curIndex
            self.sourceSolutions[curIndex] = s
            curIndex += 1

        self.library.merge(other.library)

