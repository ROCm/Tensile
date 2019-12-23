import logging,pytest
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import YamlBuilder

log =logging.getLogger("testlog")

@pytest.mark.parametrize("problemSizes", [pytest.defaultSizes, pytest.resnetSizes])
def test_ckyx_1x1(run_convolution_level,problemSizes):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'CKYX',
                      'Filter': '1x1',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[0, 3, 2])
    assert(z['IndexAssignmentsB']==[1, 3, 2])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")

    solutionName = run_convolution_level.solution.__name__
    if (solutionName == "asm3_pbd" or solutionName=="asm3_splitu") and \
       problemSizes[0] == YamlBuilder.ProblemSizesResNet:
        pytest.skip("bug with asm splitu")

    run_convolution_level.func(conv, z, run_convolution_level.solution, problemSizes[0], problemSizes[1])

def test_ckyx_1x1_nopack(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'CKYX',
                      'PackedSpatialDims': 0,
                      'Filter': '1x1',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
    assert(z['IndexAssignmentsB']==[2, 4, 3])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")

    if run_convolution_level.solution.__name__ == "asm3_splitu":
        pytest.skip("bug with asm splitu")

    run_convolution_level.func(conv, z, run_convolution_level.solution)


def test_ckyx_2x2(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'CKYX',
                      'Filter': '2x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[5, 4, 0, 3, 2])
    assert(z['IndexAssignmentsB']==[5, 4, 1, 3, 2])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1,2:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,4:0")

    run_convolution_level.func(conv, z, run_convolution_level.solution)
