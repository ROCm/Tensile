from __future__ import print_function

import sys
import yaml

def to_dict(obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()

    if hasattr(obj.__class__, 'DictKeys'):
        rv = {}
        for key in obj.__class__.DictKeys:
            attr = key
            if isinstance(key, tuple):
                (key, attr) = key
            rv[key] = to_dict(getattr(obj, attr))
        return rv

    if isinstance(obj, dict):
        return dict([(key, to_dict(value)) for key,value in obj.items()])

    if any([isinstance(obj, cls) for cls in [str, int, float, unicode]]):
        return obj

    try:
        obj = [to_dict(i) for i in obj]
        return obj
    except TypeError:
        pass

    return obj

class ProblemType:
    DictKeys = ['operationIdentifier', 'aType', 'bType', 'cType', 'dType']
    class FreeIndex:
        DictKeys = ['a', 'b', 'ca', 'cb', 'da', 'db']

        def __init__(self, a=None, b=None, ca=None, cb=None, da=None, db=None):
            self.a = a
            self.b = b
            self.ca = ca
            self.cb = cb
            self.da = da
            self.db = db

    class BatchIndex:
        DictKeys = ['a', 'b', 'c', 'd']
        def __init__(self, a=None, b=None, c=None, d=None):
            self.a = a
            self.b = b
            self.c = c
            self.d = d

    class BoundIndex:
        DictKeys = ['a', 'b']
        def __init__(self, a=None, b=None):
            self.a = a
            self.b = b

    @classmethod
    def FromOriginalDict(cls, d):
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
            idxDict = to_dict(idx)
            for (key, value) in idxDict.items():
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

class Predicate:
    @classmethod
    def FromOriginalDict(cls, d):
        predicates = list([p for p in map(cls.FromOriginalKeyPair, d.items()) if p is not None])
        if len(predicates) == 0:
            return cls('TruePred')
        if len(predicates) == 1:
            return predicates[0]

        return cls('And', value=predicates)

    def __init__(self, tag=None, index=None, value=None):
        self.tag = tag
        self.index = index
        self.value = value

    def to_dict(self):
        rv = {'type': self.tag}
        if self.index is not None: rv['index'] = to_dict(self.index)
        if self.value is not None: rv['value'] = to_dict(self.value)
        return rv

    def __eq__(self, other):
        return self.tag   == other.tag   and \
               self.value == other.value and \
               self.index == other.index

class ProblemPredicate(Predicate):
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

class HardwarePredicate(Predicate):
    @classmethod
    def FromOriginalDeviceSection(cls, d):
        gfxArch = d[1]
        return cls("AMDGPU", value=cls("Processor", value=gfxArch))

class ContractionSizeMapping:
    DictKeys = ['workGroup', 'macroTile', 'threadTile']

    @classmethod
    def FromOriginalDict(cls, d):
        return cls(workGroup = d['WorkGroup'],
                   macroTile = cls.ReadOriginalMacroTile(d),
                   threadTile = d['ThreadTile'])

    @classmethod
    def ReadOriginalMacroTile(cls, d):
        rv = [1,1,1]
        rv[0] = d['MacroTile0']
        rv[1] = d['MacroTile1']
        return rv

    def __init__(self, workGroup = None, threadTile = None, macroTile = None):
        self.workGroup  = workGroup
        self.threadTile = threadTile
        self.macroTile  = macroTile

class ContractionSolution:
    DictKeys = ['name',
                'problemType',
                'hardwarePredicate',
                'problemPredicate',
                'sizeMapping',
                'debugKernel',
                'info',
                'index']

    @classmethod
    def FromOriginalDict(cls, d):
        rv = cls()

        rv.name = d['SolutionNameMin']

        rv.problemType = ProblemType.FromOriginalDict(d['ProblemType'])


        rv.problemPredicate = ProblemPredicate.FromOriginalDict(d)

        if 'DebugKernel' in d:
            rv.debugKernel = d['DebugKernel']

        rv.index = d['SolutionIndex']

        rv.info = cls.ReadOriginalInfo(d)

        rv.sizeMapping = ContractionSizeMapping.FromOriginalDict(d)

        return rv

    @classmethod
    def ReadOriginalInfo(cls, d):
        return dict([(key, str(value)) for (key, value) in d.items() if key != 'ProblemType'])

    def __init__(self, **kwargs):
        self.name = None
        self.problemType = None
        self.hardwarePredicate = HardwarePredicate('TruePred')
        self.problemPredicate = ProblemPredicate('TruePred')
        self.sizeMapping = None
        self.debugKernel = False
        self.info = {}
        self.index = None

        for key, value in kwargs:
            if key not in ContractionSolution.DictKeys:
                raise KeyError("{0} is not a property of ContractionSolution.".format(key))

            setattr(self, key, value)

class SingleSolutionLibrary:
    Tag = 'Single'

    def __init__(self, solution):
        self.solution = solution

    @property
    def tag(self):
        return self.__class__.Tag

    def to_dict(self):
        return {'type': self.tag, 'index': self.solution.index}

class MatchingProperty:
    def __init__(self, tag, index=None, value=None):
        self.tag = tag
        self.index = index
        self.value = value

    def to_dict(self):
        rv = {'type': self.tag}
        if self.index is not None: rv['index'] = to_dict(self.index)
        if self.value is not None: rv['value'] = to_dict(self.value)
        return rv

class MatchingLibrary:
    Tag = 'Matching'
    DictKeys = [('type', 'tag'), 'properties', 'table', 'distance']

    @classmethod
    def FromOriginalDict(cls, d, solutions):
        indices = d[0]
        table = d[1]

        propertyKeys = {
                2:lambda: MatchingProperty('FreeSizeA', index=0),
                3:lambda: MatchingProperty('FreeSizeB', index=0),
                0:lambda: MatchingProperty('BatchSize', index=0),
                1:lambda: MatchingProperty('BoundSize', index=0)
            }

        properties = [propertyKeys[i]() for i in indices]

        table = []

        distance = {'type': 'Euclidean'}

        for row in table:
            index = row[1][0]
            value = SingleSolutionLibrary(solutions[index])
            entry = {'key': list(row[0]), 'value': value, 'speed': row[1][1]}
            table.append(entry)

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
    DictKeys = [('type', 'tag'), ('property', 'mappingProperty'), ('map', 'mapping')]

    def __init__(self, mappingProperty=None, mapping=None):
        self.mappingProperty = mappingProperty
        self.mapping = mapping

    @property
    def tag(self):
        return self.__class__.Tag

class PredicateLibrary:
    DictKeys = [('type', 'tag'), 'rows']

    def __init__(self, tag=None, rows=None):
        self.tag = tag
        self.rows = rows

class MasterSolutionLibrary:
    DictKeys = ['solutions', 'library']

    @classmethod
    def FromOriginalDict(cls, d, libraryOrder = None):
        if libraryOrder is None:
            libraryOrder = ['Hardware', 'OperationIdentifier', 'Predicates', 'Matching']

        minVersion = d[0]
        deviceSection = d[1:4]
        origProblemType = d[4]
        origSolutions = d[5]
        origLibrary = d[6:8]

        problemType = ProblemType.FromOriginalDict(origProblemType)

        solutions = dict([(solution.index, solution) for solution in map(ContractionSolution.FromOriginalDict, origSolutions)])
        print(type(solutions))
        matchingLibrary = MatchingLibrary.FromOriginalDict(origLibrary, solutions)

        for libName in reversed(libraryOrder):
            if libName == 'Matching':
                library = matchingLibrary

            elif libName == 'Hardware':
                newLib = PredicateLibrary(tag='Hardware', rows=[])
                pred = HardwarePredicate.FromOriginalDeviceSection(deviceSection)
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

        return cls(solutions, library)

    def __init__(self, solutions, library):
        self.solutions = solutions
        self.library = library

    def to_dict(self):
        return {'solutions': to_dict(self.solutions.itervalues()), 'library': to_dict(self.library)}

def main(args):
    with open(args[0]) as inFile:
        data = yaml.load(inFile)

    if True:
        masterLibrary = MasterSolutionLibrary.FromOriginalDict(data)
        #import pdb
        #pdb.set_trace()
        outData = to_dict(masterLibrary)

    else:
        originalSolutions = data[5]
        #print(originalSolutions)
        newSolutions = []
        for s in originalSolutions:
            newSolutions.append(ContractionSolution.FromOriginalDict(s))

        outData = [to_dict(s) for s in newSolutions]

    with open(args[1], 'w') as outFile:
        if True:
            yaml.dump(outData, outFile)
        else:
            import json
            json.dump(outData, outFile, sort_keys=True, indent=2, separators=(",", ": "))

if __name__ == "__main__":
    main(sys.argv[1:])

