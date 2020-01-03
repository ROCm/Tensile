import logging,pytest
from Tensile.SolutionStructs import Convolution
log =logging.getLogger("testlog")

def test_nhwc_defaults(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[3, 1, 2])
    assert(z['IndexAssignmentsB']==[3, 0, 2])
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")
    assert(conv.solutionParms["AssertSizeEqual"] == {})

    solutionName = run_convolution_level.solution.__name__
    if solutionName.startswith("asm"):
        pytest.skip("bug with asm NHWC")
    #run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nhwc_filter2x2(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      'Filter': '3x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[3, 5, 4, 1, 2])
    assert(z['IndexAssignmentsB']==[5, 4, 3, 0, 2])
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,4:0")
    assert(conv.solutionParms["AssertSizeEqual"] == {5:2, 4:3})
    #skip since bug in asm output swap required by NHWC
    solutionName = run_convolution_level.solution.__name__
    if solutionName.startswith("asm"):
        pytest.skip("bug with asm NHWC")
    #run_convolution_level.func(conv, z, run_convolution_level.solution)

