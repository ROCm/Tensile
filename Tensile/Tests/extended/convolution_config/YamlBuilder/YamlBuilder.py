import copy, operator
from functools import reduce
import yaml

class Solutions:

    # If adding new solutions, update 'solutions' in conftest.py

    @classmethod
    def commonSetup(cls):
        return {
            "InitialSolutionParameters": None,
            "BenchmarkCommonParameters": [{"EdgeType": ["ShiftPtr"]}],
            "ForkParameters": None,
            "BenchmarkForkParameters": None,
            "JoinParameters": None,
            "BenchmarkJoinParameters": None,
            "BenchmarkFinalParameters": None
        }

    @classmethod
    def src1(cls):
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [0]},
                    {"KernelLanguage": ["Source"]},
                    {"ThreadTile": [
                        [ 2, 2 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1 ]
                        ]},
                    {"DepthU": [8]},
                    {"PackBatchDims": [1]}, # required to handle some Backward-Weights cases
                    {"GlobalReadVectorWidth": [1]},
                    {"VectorWidth": [1]},
                ]

        return s

    @classmethod
    def src5_gsu(cls):
        # Has GSU configs + some other options.
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [0]},
                    {"KernelLanguage": ["Source"]},
                    {"ThreadTile": [
                        [ 8, 8 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 16, 1 ]
                        ]},
                    {"DepthU": [4]},
                    {"PackBatchDims": [0,1]},
                    {"PackSummationDims": [0,1]},
                    {"GlobalSplitU": [1,2,4]},
                    {"GlobalReadVectorWidth": [1,-1]},
                    {"VectorWidth": [1,-1]},
                ]

        return s

    @classmethod
    def asm1(cls):
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [0]},
                    {"KernelLanguage": ["Assembly"]},
                    {"ThreadTile": [
                        [ 2, 2 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1 ]
                        ]},
                    {"DepthU": [8]},
                    {"GlobalReadVectorWidth": [-1]},
                    {"VectorWidth": [1]},
                    {"FractionalLoad": [0]}
                ]

        return s

    @classmethod
    def asm3_pbd(cls):
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [1]},
                    {"KernelLanguage": ["Assembly"]},
                    {"ThreadTile": [
                        [ 2, 2 ],
                        [ 4, 8 ],
                        [ 8, 8 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1],
                        [ 16, 8, 1]
                        ]},
                    {"DepthU": [8]},
                    {"GlobalReadVectorWidth": [-1]},
                    {"VectorWidth": [1,4]},
                    {"FractionalLoad": [0,1]},
                    {"PackBatchDims": [0,1]},
                ]

        return s

    @classmethod
    def asm3_splitu(cls):
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [1]},
                    {"KernelLanguage": ["Assembly"]},
                    {"ThreadTile": [
                        [ 4, 4 ],
                        [ 4, 8 ],
                        [ 8, 8],
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1],
                        #[  8, 8, 2],
                        #[  8, 8, 4],
                        #[ 16, 8, 2],
                        #[ 16, 16, 2],
                        ]},
                    {"DepthU": [8]},
                    {"GlobalReadVectorWidth": [-1]},
                    {"GlobalSplitU": [1,2,3,4,8]},
                    {"VectorWidth": [1,4]},
                    {"PackBatchDims": [1]},
                    {"FractionalLoad": [0,1]}
                ]

        return s

    @classmethod
    def defaultSolution(cls):
        return cls.asm3_pbd

class YamlBuilder:

    def __init__(self, doc):
        self.doc = doc

    def write(self, fname):
        with open(str(fname), "w") as f:
            yaml.dump(self.doc, f)

    @classmethod
    def Header(cls,debug):
        rv= \
        {
            "GlobalParameters":
            {
                "MinimumRequiredVersion": "4.2.0",
                "ForceRedoBenchmarkProblems": True,
                "ForceRedoLibraryLogic": True,
                "ForceRedoLibraryClient": True,
                "CMakeBuildType": "Release",
                "EnqueuesPerSync": 1,
                "SyncsPerBenchmark": 1,
                "LibraryPrintDebug": False,
                "NumElementsToValidate": 1000,
                "ValidationMaxToPrint": 4,
                "ValidationPrintValids": False,
                "ShortNames": False,
                "MergeFiles": True,
                "KernelTime": True,
                "SolutionSelectionAlg": 1,
                "ProblemFromConvolution": True,
                "NewClient": 2,
                #"CpuThreads": 0,
            }
        }

        if debug:
          rv["GlobalParameters"]["CMakeBuildType"] = "Debug"
          rv["GlobalParameters"]["CpuThreads"] = 0

        return rv


    @classmethod
    def genSpatials(cls, conv, spatialRange):
        spatials = []
        for h in spatialRange:
            for w in spatialRange:
                if conv.formatNumSpatialDims==2:
                    spatials.append((h,w))
                elif conv.formatNumSpatialDims==3:
                    for d in [7]:
                        spatials.append((h,w,d))
                else:
                    raise RuntimeError('unknown formatNumSpatialDims=%d'%conv.formatNumSpatialDims)
        return spatials


    @classmethod
    def genExacts(cls, conv, nRange, ckRange, spatialRange):
        exactSizes = []
        spatials = cls.genSpatials(conv, spatialRange)
        for n in nRange:
            for c in ckRange:
                for k in ckRange:
                    for s in spatials:
                        (problemSizes,problemStrides) = conv.makeProblem(False, n, c, k, s)
                        exactSizes.append(problemSizes)
        return exactSizes

    @staticmethod
    def memSize(indexAssignments, exactSizes):
        """
        Return max memory required for specified index assignments in list of exacts
        """
        maxSize=0
        for exact in exactSizes:
            size = reduce(operator.mul,[exact[idx] for idx in indexAssignments], 1)
            if size > maxSize:
                maxSize = size

        return maxSize

    @classmethod
    def ProblemSizesResNet(cls, conv, problemType, problemLevel):
        exactSizes = []
        n=64
        for (c,h,w,k) in (
                [1024,14,14,2048],
                [1024,14,14,256],
                [1024,14,14,512],
                [128,28,28,128],
                [128,28,28,512],
                [2048,7,7,512],
                [256,14,14,1024],
                [256,14,14,256],
                [256,56,56,128],
                [256,56,56,512],
                [256,56,56,64],
                [3,230,230,64],
                [512,28,28,1024],
                [512,28,28,128],
                [512,28,28,256],
                [512,7,7,2048],
                [512,7,7,512],
                [64,56,56,256],
                [64,56,56,64],
                [64,56,56,64],
             ):
            (problemSizes,problemStrides) = conv.makeProblem(False, n, c, k, spatialIn=[h,w])
            exactSizes.append(problemSizes)
        return [{"ProblemSizes": [ {"Exact": e} for e in exactSizes]}]


    @classmethod
    def ProblemSizes(cls, conv, problemType, problemLevel):
        if conv.spatial:
            spatialIn = conv.spatial
        else:
            spatialIn = [14]*conv.formatNumSpatialDims

        if -1 in spatialIn:
            raise RuntimeError('Spatial must be completely specified, not "%s"'%spatialIn)
        if -1 in conv.filter:
            raise RuntimeError('Filter must be completely specified, not "%s"'%conv.config['Filter'])

        exactSizes = []
        (problemSizes,problemStrides) = conv.makeProblem(False, n=8, c=32, k=16, spatialIn=spatialIn)
        exactSizes.append(problemSizes)

        if problemLevel==2:
            exactSizes += cls.genExacts(conv, nRange=(1,2,8), ckRange=[64], spatialRange=(7,14,56))
        elif problemLevel==3:
            exactSizes += cls.genExacts(conv, nRange=(1,2,8), ckRange=range(127,129), spatialRange=(7,14,56))
        elif problemLevel==4:
            exactSizes += cls.genExacts(conv, nRange=(1,2,8), ckRange=range(127,129), spatialRange=(7,56,73,111,194))

        try:
            asize = cls.memSize(problemType["IndexAssignmentsA"], exactSizes)
            bsize = cls.memSize(problemType["IndexAssignmentsB"], exactSizes)
            dsize = cls.memSize(range(0,problemType["NumIndicesC"]), exactSizes)
            print ("generated %d exact sizes.  ElementSizes: A=%d B=%d D=%d Total=%d" % \
                    (len(exactSizes), asize, bsize, dsize, asize+bsize+dsize))
        except KeyError:
            None

        return [{"ProblemSizes": [ {"Exact": e} for e in exactSizes]}]

    @classmethod
    def ConvolutionVsContraction(cls, conv, solution, dataType):
        """
        Generates a YamlBuilder object that will run in
        ConvolutionVsContraction mode.
        """
        obj = cls.ConvolutionContraction(conv, {}, solution, problemFunc=cls.ProblemSizes, problemLevel=1, dataType=dataType)
        obj.doc["GlobalParameters"]["ConvolutionVsContraction"] = 1
        obj.doc["GlobalParameters"]["ProblemFromConvolution"] = 1
        for problem in obj.doc["BenchmarkProblems"]:
            problem[0]["OperationType"] = conv.convolutionType
            problem[0]["ConvolutionConfig"] = [copy.deepcopy(conv.config)]

        return obj

    @classmethod
    def ConvolutionContraction(cls, conv, problemType, solution, dataType, \
                                problemFunc, problemLevel=1):
        """
        Generates a YamlBuilder object that will run a convolution, in normal
        contraction mode.
        """
        doc = cls.Header(debug=False)

        tensileProblemType = {
            "OperationType": "TensorContraction",
            "DataType": dataType
        }

        tensileProblemType.update(problemType)

        benchmarkParams = solution()
        for (key,value) in conv.solutionParms.items():
            benchmarkParams["ForkParameters"].append({key:[value]})
        benchmarkParams["BenchmarkFinalParameters"] = problemFunc(conv, problemType, problemLevel)

        doc["BenchmarkProblems"] = [[tensileProblemType, benchmarkParams]]

        return cls(doc)
