import logging,pytest
from Tensile.SolutionStructs import Convolution
log =logging.getLogger("testlog")

@pytest.mark.parametrize("unrollOnChannel", [0,1], ids=["unrollOnChannel0", "unrollOnChannel1"])
def test_nchw_backwardweights_defaults(tensile_state, run_convolution_level, unrollOnChannel):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'UnrollOnChannel': unrollOnChannel,
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        (cdim,filterDims) = (3,[2]) if conv.unrollOnChannel else (2,[3])
        assert(z['NumIndicesC']==2)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim])
        assert(z['IndexAssignmentsB']==filterDims + [1, cdim, 0])
        #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,0:1")
        #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_backwardweights_filter3x1(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x1',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        (cdim,filterDims) = (4,[3]) if conv.unrollOnChannel else (3,[4])
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, 1, cdim])
        assert(z['IndexAssignmentsB']==filterDims + [2, cdim, 1])
        #assert(conv.solutionParms["AssertStrideAEqual"] == "2:1,0:1")
        #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
        assert(conv.solutionParms["AssertSizeEqual"] == {0:3})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_backwardweights_filter1x3(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x3',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        (cdim,filterDims) = (4,[3]) if conv.unrollOnChannel else (3,[4])
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, 1, cdim])
        assert(z['IndexAssignmentsB']==filterDims + [2, cdim, 1])
        #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,2:1,0:1")
        #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
        assert(conv.solutionParms["AssertSizeEqual"] == {0:3})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_backwardweights_filter3x5(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x5',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        # TODO - need to expand filter dims
        (cdim,filterDims) = (5,[4]) if conv.unrollOnChannel else (4,[5])
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA'] == filterDims + [0, 1, 2, cdim])
        assert(z['IndexAssignmentsB'] == filterDims + [3, cdim, 2])
        assert(z['SetConstStrideA'] == [[0,1], [filterDims[0],1]])
        assert(z['SetConstStrideB'] == [[2,0], [filterDims[0],1]])
        #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,3:1,0:1")
        #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
        assert(conv.solutionParms["AssertSizeEqual"] == {0:5, 1:3})
    run_convolution_level.func(conv, z, run_convolution_level.solution)


def test_nchw_backwardweights_filter3x5_nopack(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x5',
                      'PackedSpatialDims': 0,
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        # TODO - need to expand filter dims
        (cdim,filterDims) = (6,[5,4]) if conv.unrollOnChannel else (4,[6,5])
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA'] == filterDims + [0, 1, 2, cdim])
        assert(z['IndexAssignmentsB'] == filterDims + [3, cdim, 2])
        assert(z['SetConstStrideA'] == [[0,1], [filterDims[0],1]])
        assert(z['SetConstStrideB'] == [[2,0], [filterDims[0],1]])

        #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,3:1,0:1")
        #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
        assert(conv.solutionParms["AssertSizeEqual"] == {0:5, 1:3})
    run_convolution_level.func(conv, z, run_convolution_level.solution)
