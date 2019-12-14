import copy
import yaml

class YamlBuilder:

    def __init__(self, doc):
        self.doc = doc

    def write(self, fname):
        with open(str(fname), "w") as f:
            yaml.dump(self.doc, f)

    @classmethod
    def Header(cls):
        return \
        {
            "GlobalParameters":
            {
                "MinimumRequiredVersion": "4.2.0",
                "ForceRedoBenchmarkProblems": True,
                "ForceRedoLibraryLogic": True,
                "ForceRedoLibraryClient": True,
                "CMakeBuildType": "Debug",
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
                "NewClient": 2,
            }
        }

    @classmethod
    def src1(cls):
        return {
            "InitialSolutionParameters": None,
            "BenchmarkCommonParameters": [{"EdgeType": ["ShiftPtr"]}],
            "ForkParameters":
                [
                    {"PrefetchGlobalRead": [0]},
                    {"KernelLanguage": ["Source"]},
                    {"ThreadTile": [
                        [ 2, 2 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1 ]
                        #[ 16, 8, 1]
                        ]},
                    {"DepthU": [8]},
                    {"GlobalReadVectorWidth": [1]},
                    {"VectorWidth": [1]},
                    {"FractionalLoad": [0]}
                ],
            "BenchmarkForkParameters": None,
            "JoinParameters": None,
            "BenchmarkJoinParameters": None,
            "BenchmarkFinalParameters": None
        }

    @classmethod
    def asm3(cls):
        return {
            "InitialSolutionParameters": None,
            "BenchmarkCommonParameters": [{"EdgeType": ["ShiftPtr"]}],
            "ForkParameters":
                [
                    {"PrefetchGlobalRead": [0]},
                    {"KernelLanguage": ["Assembly"]},
                    {"ThreadTile": [
                        [ 2, 2 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1 ]
                        #[ 16, 8, 1]
                        ]},
                    {"DepthU": [8]},
                    {"GlobalReadVectorWidth": [1]},
                    {"VectorWidth": [1]},
                    {"FractionalLoad": [0]}
                ],
            "BenchmarkForkParameters": None,
            "JoinParameters": None,
            "BenchmarkJoinParameters": None,
            "BenchmarkFinalParameters": None
        }

    @classmethod
    def ProblemSizes(cls, conv):
        if conv.spatial:
            spatialIn = conv.spatial
        else:
            spatialIn = [14]*conv.formatNumSpatialDims
        # replace any TBD spatials with something:
        spatialIn = [x if x>0 else 42 for x in spatialIn]

        if -1 in conv.filter:
            raise RuntimeError('Filter must be completely specified, not "%s"'%conv.config['Filter'])

        (problemSizes,problemStrides) = conv.makeProblem(False, 8, 32, 16, spatialIn)

        return [{"ProblemSizes": [{"Exact": problemSizes}]}]

    @classmethod
    def ConvolutionVsContraction(cls, conv, dataType='s'):
        """
        Generates a YamlBuilder object that will run in
        ConvolutionVsContraction mode.
        """
        obj = cls.ConvolutionContraction(conv, {}, dataType)
        obj.doc["GlobalParameters"]["ConvolutionVsContraction"] = 1
        obj.doc["GlobalParameters"]["ProblemFromConvolution"] = 1
        for problem in obj.doc["BenchmarkProblems"]:
            problem[0]["OperationType"] = conv.convolutionType
            problem[0]["ConvolutionConfig"] = [copy.deepcopy(conv.config)]

        return obj

    @classmethod
    def ConvolutionContraction(cls, conv, problemType, dataType='s'):
        """
        Generates a YamlBuilder object that will run a convolution, in normal
        contraction mode.
        """
        doc = cls.Header()

        tensileProblemType = {
            "OperationType": "TensorContraction",
            "DataType": dataType
        }

        tensileProblemType.update(problemType)

        benchmarkParams = cls.asm3()
        benchmarkParams["BenchmarkFinalParameters"] = cls.ProblemSizes(conv)

        doc["BenchmarkProblems"] = [[tensileProblemType, benchmarkParams]]

        return cls(doc)
