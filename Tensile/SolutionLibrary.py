################################################################################
# Copyright 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
from typing_extensions import dataclass_transform

from . import Properties
from . import Hardware
from . import Common
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

class MatchingLibrary:
    Tag = 'Matching'
    StateKeys = [('type', 'tag'), 'properties', 'table', 'distance']

    @classmethod
    def FromOriginalState(cls, d, solutions):
        indices = d["indexOrder"]
        distance = d["distance"]
        origTable = d["table"]

        propertyKeys = {
                2:lambda: Properties.Property('FreeSizeA', index=0),
                3:lambda: Properties.Property('FreeSizeB', index=0),
                #0:lambda: Properties.Property('BatchSize', index=0),
                1:lambda: Properties.Property('BoundSize', index=0)
            }

        properties = list([propertyKeys[i]() for i in indices if i in propertyKeys])
        keyOrder = [i for i,j in enumerate(indices) if j in propertyKeys]

        table = []

        for row in origTable:
            try:
                index = row[1][0]
                value = SingleSolutionLibrary(solutions[index])
                key = list([row[0][i] for i in keyOrder])
                #key = list(row[0][0:len(properties)])
                entry = {'key': key, 'value': value, 'speed': row[1][1]}
                table.append(entry)
            except KeyError:
                pass

        table.sort(key=lambda r: r['key'])

        return cls(properties, table, distance)

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        assert self.__class__ == other.__class__ \
                and self.properties == other.properties \
                and self.distance == other.distance

        self.table += other.table

        self.table.sort(key=lambda r: r['key'])

    def remapSolutionIndices(self,indexMap):
        pass

    def __init__(self, properties, table, distance):
        self.properties = properties
        self.table = table
        self.distance = distance

class DecisionTreeLibrary:
    Tag= 'DecisionTree'
    StateKeys = [('type', 'tag'), 'properties', 'trees']

    @classmethod
    def FromOriginalState(cls, d, solutions):
        properties = d["properties"]
        origTrees = d["trees"]

        trees = []

        for tree in origTrees:
            index = tree["solution"]
            value = SingleSolutionLibrary(solutions[index])

            entry = {'tree': tree["tree"], 'value': value}
            trees.append(entry)

        return cls(properties, trees)

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        raise RuntimeError("DecisionTreeLibrary does not support merging; ensure each library row has a unique predicate")

    def remapSolutionIndices(self,indexMap):
        pass

    def __init__(self, properties, trees):
        self.properties = properties
        self.trees = trees

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

        rowPreds = [r['predicate'] for r in self.rows]

        for row in other.rows:
            if row['predicate'] in rowPreds:
                myRownum = rowPreds.index(row['predicate'])
                self.rows[myRownum]['library'].merge(row['library'])
            else:
                self.rows.append(row)

        # Sort to ensure consistent fallback logic.
        self.rows.sort(key=lambda x: x['predicate'])

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
    def FromOriginalState(cls, d, origSolutions, solutionClass=Contractions.Solution, libraryOrder = None):

        # functions for creating each "level" of the library
        def hardware(d, problemType, solutions, library):
            devicePart = d["ArchitectureName"]
            cuCount = d["CUCount"]

            newLib = PredicateLibrary(tag='Hardware')
            if devicePart == 'fallback':
                pred = Hardware.HardwarePredicate('TruePred')
            else:
                pred = Hardware.HardwarePredicate.FromHardware(Common.gfxArch(devicePart), cuCount)

            newLib.rows.append({'predicate': pred, 'library': library})
            return newLib

        def operationIdentifier(d, problemType, solutions, library):
            operationID = problemType.operationIdentifier
            prop = Properties.Property('OperationIdentifier')
            mapping = {operationID: library}

            newLib = ProblemMapLibrary(prop, mapping)
            return newLib

        def performanceMetric(d, problemType, solutions, library):
            if d.get("PerfMetric", "DeviceEfficiency") != 'DeviceEfficiency':
                predicate = Properties.Predicate(tag=d["PerfMetric"])
            else:
                predicate = Properties.Predicate(tag='TruePred')
            newLib = PredicateLibrary(tag='Problem')
            newLib.rows.append({'predicate': predicate, 'library': library})
            return newLib

        def fp16AltImpl(d, problemType, solutions, library):
            if d.get("Fp16AltImpl"):
                predicate = Properties.Predicate(tag='Fp16AltImpl')
            else:
                predicate = Properties.Predicate(tag='TruePred')
            newLib = PredicateLibrary(tag='Problem')
            newLib.rows.append({'predicate': predicate, 'library': library})
            return newLib

        def predicates(d, problemType, solutions, library):
            predicates = problemType.predicates(includeBatch=True, includeType=True)
            predicate = Contractions.ProblemPredicate.And(predicates)

            newLib = PredicateLibrary(tag='Problem')
            newLib.rows.append({'predicate': predicate, 'library': library})
            return newLib

        def selection(d, problemType, solutions, library):
            if d["LibraryType"] == "Matching":
                if d["Library"]["distance"] == 'Equality':
                    predicate = Properties.Predicate(tag='EqualityMatching')
                else:
                    predicate = Properties.Predicate(tag='TruePred')

                matchingLib = MatchingLibrary.FromOriginalState(d["Library"], solutions)
                library = PredicateLibrary(tag='Problem')
                library.rows.append({'predicate': predicate, 'library': matchingLib})

            elif d["LibraryType"] == "DecisionTree":
                library = PredicateLibrary(tag='Problem')
                for lib in d["Library"]:
                    preds = lib["region"]
                    predObjs = [Properties.Predicate.FromOriginalState(p) for p in preds]

                    if len(predObjs) == 1:
                        predicate = predObjs[0]
                    else:
                        predicate = Properties.Predicate.And(predObjs)

                    treeLib = DecisionTreeLibrary.FromOriginalState(lib, solutions)
                    library.rows.append({'predicate': predicate, 'library': treeLib})

            return library
        # end library creation functions

        if libraryOrder is None:
            libraryOrder = [hardware, operationIdentifier, performanceMetric, fp16AltImpl, predicates, selection]
        assert libraryOrder[-1] == selection

        problemType = Contractions.ProblemType.FromOriginalState(d["ProblemType"])
        allSolutions = [solutionClass.FromSolutionStruct(s) for s in origSolutions]
        cls.FixSolutionIndices(allSolutions)

        # library is constructed in reverse order i.e. bottom-up
        library = None
        for libName in reversed(libraryOrder):
            library = libName(d, problemType, allSolutions, library)

        solutions = {s.index: s for s in allSolutions}
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

    def __init__(self, solutions, library, version=None):
        self.solutions = solutions
        self.library = library
        self.version = version

    def state(self):
        rv = {'solutions': state(iter(list(self.solutions.values()))),
              'library': state(self.library)}

        if self.version is not None:
            rv['version'] = self.version
        return rv

    def applyNaming(self, naming=None):
        if naming is None:
            #allSolutions = itertools.chain(iter(list(self.solutions.values())), iter(list(self.sourceSolutions.values())))
            kernels = list(itertools.chain(*[s.originalSolution.getKernels() for s in self.solutions.values()]))
            naming = OriginalSolution.getMinNaming(kernels)

        for s in list(self.solutions.values()):
            s.name = OriginalSolution.getNameMin(s.originalSolution.getKernels()[0], naming)

    def remapSolutionIndicesStartingFrom(self, curIndex):
        reIndexMap = {}
        solutionCopy = self.solutions
        self.solutions = dict()
        for k,s in solutionCopy.items():
            reIndexMap[s.index] = curIndex
            s.index = curIndex
            self.solutions[curIndex] = s
            curIndex += 1

        self.library.remapSolutionIndices(reIndexMap)

    def merge(self, other, startIndex=0):
        assert self.__class__ == other.__class__

        curIndex = max(startIndex, max(self.solutions.keys()) + 1)

        reIndexMap = {}
        for k,s in other.solutions.items():
            reIndexMap[s.index] = curIndex
            s.index = curIndex
            self.solutions[curIndex] = s
            curIndex += 1

        other.library.remapSolutionIndices(reIndexMap)

        self.library.merge(other.library)

        return curIndex    #Next unused index

    @property
    def cpp_base_class(self):
        return 'SolutionLibrary<ContractionProblem, ContractionSolution>'

    @property
    def cpp_class(self):
        return 'MasterSolutionLibrary<ContractionProblem, ContractionSolution>'
