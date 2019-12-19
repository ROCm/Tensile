import pytest,logging
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import YamlBuilder


@pytest.mark.parametrize("problem_level", [1,2,3,4])
def test_2d(problem_level):
    "Test number of problems generated"
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': '2x2',
                      'Filter': '3x3',
                      })

    exacts = YamlBuilder.ProblemSizes(conv, z, problem_level)


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
