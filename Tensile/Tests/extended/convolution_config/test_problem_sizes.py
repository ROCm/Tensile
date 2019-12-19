import pytest,logging
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import YamlBuilder

@pytest.mark.parametrize(
        "problemSizes",
        [pytest.param((YamlBuilder.ProblemSizes,level), id="default-lvl=%d"%level) for level in [1,2,3,4]] +
        [pytest.resnetSizes]
        )
def test_2d_stride1(problemSizes):
    "Test number of problems generated"
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': '1x1',
                      'Filter': '1x1',
                      })

    exacts = problemSizes[0](conv, z, problemSizes[1])
    #print ("exacts=", exacts)


@pytest.mark.parametrize("problemSizes", [pytest.defaultSizes, pytest.resnetSizes])
@pytest.mark.parametrize("problemLevel", [1,2,3,4])
def test_2d_stride2(problemSizes, problemLevel):
    "Test number of problems generated"
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': '2x2',
                      'Filter': '3x3',
                      })

    exacts = problemSizes[0](conv, z, problemLevel)
    #print ("exacts=", exacts)


@pytest.mark.parametrize("problem_level", [1,2,3,4])
def test_3d(problem_level):
    "Test number of problems generated"
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCDHW',
                      'Stride': '2x2x2',
                      'Filter': '3x3x3',
                      })

    exacts = YamlBuilder.ProblemSizes(conv, z, problem_level)
