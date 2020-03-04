import logging,pytest
from Tensile.SolutionStructs import Convolution
log =logging.getLogger("testlog")

def test_nhwc_defaults(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[3, 1, 2])
        assert(z['IndexAssignmentsB']==[3, 0, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})

    solutionName = run_convolution_level.solution.__name__
    if solutionName.startswith("asm"):
        pytest.skip("bug with asm NHWC")
    #run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nhwc_filter2x2(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      'Filter': '3x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [4,3] if conv.unrollOnChannel else [5,4]
        cdim = 5 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[cdim] + filterDims + [1, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 0, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,4:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2, filterDims[1]:3})
    #skip since bug in asm output swap required by NHWC, impacts both source and asm
    solutionName = run_convolution_level.solution.__name__
    if solutionName.startswith("asm"):
        pytest.skip("bug with asm NHWC")
    #run_convolution_level.func(conv, z, run_convolution_level.solution)

