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

from . import Properties
from . import Hardware
from . import Contractions
from .SolutionStructs import Solution as OriginalSolution
from .Utils import state

class SingleSolutionLibrary:
    Tag = 'Single'

    def __init__(self, solution):
        self.solution = solution

    @property
    def tag(self):
        return self.__class__.Tag

    def state(self):
        return {'type': self.tag, 'index': self.solution.index}

    def remapSolutionIndices(self,indexMap):
        pass

class GranularitySelectionLibrary:
    Tag = 'GranularitySelection'
    StateKeys = [('type', 'tag'), 'indices']

    @classmethod
    def FromOriginalState(cls, indices):
        return cls(indices)

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        idList = list(set().union(self.indices, indices))
        self.indices = idList

    def __init__(self, indices):
        self.indices = indices

    def remapSolutionIndices(self,indexMap):
        for i in range(0, len(self.indices)):
            index = self.indices[i]
            if index in indexMap:
                self.indices[i] = indexMap[index]

class MatchingLibrary:
    Tag = 'Matching'
    StateKeys = [('type', 'tag'), 'properties', 'table', 'distance']

    @classmethod
    def FromOriginalState(cls, d, solutions):
        indices = d[0]
        origTable = d[1]

        propertyKeys = {
                2:lambda: Properties.Property('FreeSizeA', index=0),
                3:lambda: Properties.Property('FreeSizeB', index=0),
                0:lambda: Properties.Property('BatchSize', index=0),
                1:lambda: Properties.Property('BoundSize', index=0)
            }

        properties = list([propertyKeys[i]() for i in indices])

        table = []

        distance = {'type': 'Euclidean'}

        for row in origTable:
            try:
                index = row[1][0]
                value = SingleSolutionLibrary(solutions[index])
                key = list(row[0][0:len(properties)])
                entry = {'key': key, 'value': value, 'speed': row[1][1]}
                table.append(entry)
            except KeyError:
                pass

        return cls(properties, table, distance)

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        assert self.__class__ == other.__class__ \
                and self.properties == other.properties \
                and self.distance == other.distance

        self.table += other.table

    def remapSolutionIndices(self,indexMap):
        pass

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
 
    def remapSolutionIndices(self,indexMap):
        for key,value in list(self.mapping.items()):
            value.remapSolutionIndices(indexMap)

class PredicateLibrary:
    StateKeys = [('type', 'tag'), 'rows']

    def __init__(self, tag=None, rows=None):
        self.tag = tag
        if rows is None: rows = []
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

    def remapSolutionIndices(self,indexMap):
        for row in self.rows:
          row['library'].remapSolutionIndices(indexMap)


class MasterSolutionLibrary:
    StateKeys = ['solutions', 'library']

    @classmethod
    def FixSolutionIndices(cls, solutions):
        # fix missing and duplicate solution indices.
        try:
            maxSolutionIdx = max([s.index for s in solutions if s.index is not None])
        except ValueError:
            maxSolutionIdx = -1

        solutionsSoFar = set()
        for solution in solutions:
            if solution.index is None or solution.index in solutionsSoFar:
                maxSolutionIdx += 1
                solution.index = maxSolutionIdx
            else:
                solutionsSoFar.add(solution.index)
    @classmethod
    def FromOriginalState(cls, d, solutionClass=Contractions.Solution, libraryOrder = None):
        if libraryOrder is None:
            libraryOrder = ['Hardware', 'OperationIdentifier', 'Predicates', 'Matching']

        _ = d[0]
        deviceSection = d[1:4]
        origProblemType = d[4]
        origSolutions = d[5]
        origLibrary = d[6:8]

        problemType = Contractions.ProblemType.FromOriginalState(origProblemType)

        buildGranularity = False
        if len(d) > 9 and d[9]:
          buildGranularity = True
          assert libraryOrder[-1] == "Matching"
          libraryOrder[-1] = "Granularity"
        
        allSolutions = [solutionClass.FromOriginalState(s, deviceSection) for s in origSolutions]
        cls.FixSolutionIndices(allSolutions)

        solutions = {s.index: s for s in allSolutions}

        for libName in reversed(libraryOrder):
            if libName == 'Matching':
                matchingLibrary = MatchingLibrary.FromOriginalState(origLibrary, allSolutions)
                library = matchingLibrary
            
            elif libName == 'Granularity':
                selectionIndices = d[9]["TileSelectionIndices"]
                library = GranularitySelectionLibrary.FromOriginalState(selectionIndices)

            elif libName == 'Hardware':
                newLib = PredicateLibrary(tag='Hardware')
                devicePart = deviceSection[1]
                if devicePart == 'fallback':
                    pred = Hardware.HardwarePredicate("TruePred")
                else:
                    isa = tuple(map(int,devicePart[3:6]))
                    pred = Hardware.HardwarePredicate.FromISA(isa)

                newLib.rows.append({'predicate': pred, 'library': library})
                library = newLib

            elif libName == 'Predicates':
                predicates = problemType.predicates(includeBatch=True, includeType=True)
                predicate = Contractions.ProblemPredicate.And(predicates)

                newLib = PredicateLibrary(tag='Problem')
                newLib.rows.append({'predicate': predicate, 'library': library})
                library = newLib

            elif libName == 'OperationIdentifier':
                operationID = problemType.operationIdentifier
                prop = Properties.Property('OperationIdentifier')
                mapping = {operationID: library}

                newLib = ProblemMapLibrary(prop, mapping)
                library = newLib
            else:
                raise ValueError("Unknown value " + libName)

        rv = cls(solutions, library)
        return rv

    @classmethod
    def BenchmarkingLibrary(cls, solutions):
        solutionObjs = list([Contractions.Solution.FromOriginalState(s._state) for s in solutions])
        cls.FixSolutionIndices(solutionObjs)

        predRows = list([{'predicate': s.problemPredicate, 'library': SingleSolutionLibrary(s)} for s in solutionObjs])
        library = PredicateLibrary(tag='Problem', rows=predRows)

        solutionMap = {s.index: s for s in solutionObjs}

        return cls(solutionMap, library)


    def __init__(self, solutions, library):
        self.solutions = solutions
        self.library = library

    def state(self):
        return {'solutions': state(iter(list(self.solutions.values()))), 'library': state(self.library)}

    def applyNaming(self, naming=None):
        if naming is None:
            #allSolutions = itertools.chain(iter(list(self.solutions.values())), iter(list(self.sourceSolutions.values())))
            kernels = list(itertools.chain(*[s.originalSolution.getKernels() for s in self.solutions.values()]))
            naming = OriginalSolution.getMinNaming(kernels)

        for s in list(self.solutions.values()):
            s.name = OriginalSolution.getNameMin(s.originalSolution.getKernels()[0], naming)

    def merge(self, other):
        assert self.__class__ == other.__class__

        curIndex = max(self.solutions.keys()) + 1

        reIndexMap = {}
        for k,s in other.solutions.items():
            reIndexMap[s.index] = curIndex
            s.index = curIndex
            self.solutions[curIndex] = s
            curIndex += 1

        other.library.remapSolutionIndices(reIndexMap)

        self.library.merge(other.library)

    @property
    def cpp_base_class(self):
        return "SolutionLibrary<ContractionProblem, ContractionSolution>"

    @property
    def cpp_class(self):
        return "MasterSolutionLibrary<ContractionProblem, ContractionSolution>"

