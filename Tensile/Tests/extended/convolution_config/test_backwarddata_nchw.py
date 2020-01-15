import logging,pytest
from pytest import args
from Tensile.SolutionStructs import Convolution
log =logging.getLogger("testlog")

@pytest.mark.skip(reason="backward_data under construction")
def test_nchw_backwarddata_defaults(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardData',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '14x14',
                      })
    log.debug(conv.printUsage(z))
    if not args["no_conv_assertions"]:
        assert(z['NumIndicesC']==2)
        assert(z['IndexAssignmentsA']==[3, 0, 2])
        assert(z['IndexAssignmentsB']==[3, 1, 2])
        #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,0:1")
        #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

@pytest.mark.skip(reason="backward_data under construction")
def test_nchw_backwarddata_filter3x1(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardData',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x1',
                      })
    log.debug(conv.printUsage(z))
    if not args["no_conv_assertions"]:
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[4, 0, 1, 3])
        assert(z['IndexAssignmentsB']==[4, 2, 3])
        #assert(conv.solutionParms["AssertStrideAEqual"] == "2:1,0:1")
        #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
        assert(conv.solutionParms["AssertSizeEqual"] == {4:1})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

@pytest.mark.skip(reason="backward_data under construction")
def test_nchw_backwarddata_filter1x3(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardData',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x3',
                      })
    log.debug(conv.printUsage(z))
    if not args["no_conv_assertions"]:
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[4, 0, 1, 3])
        assert(z['IndexAssignmentsB']==[4, 2, 3])
        #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,2:1,0:1")
        #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
        assert(conv.solutionParms["AssertSizeEqual"] == {4:1})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

@pytest.mark.skip(reason="backward_data under construction")
def test_nchw_backwarddata_filter3x5(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardData',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x5',
                      })
    log.debug(conv.printUsage(z))
    if not args["no_conv_assertions"]:
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA']==[5, 0, 1, 2, 4])
        assert(z['IndexAssignmentsB']==[5, 3, 4])
        #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,3:1,0:1")
        #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
        assert(conv.solutionParms["AssertSizeEqual"] == {4:3,5:5})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_backwarddata_filter3x5_nopack(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardData',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x5',
                      'PackedSpatialDims': 0,
                      })
    log.debug(conv.printUsage(z))
    if not args["no_conv_assertions"]:
        (cdim,filterDims) = (6,[5,4]) if conv.unrollOnChannel else (4,[6,5])
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA'] == filterDims + [0, 1, cdim, 3])
        assert(z['IndexAssignmentsB'] == filterDims + [2, cdim, 3])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1, 2:1})
        #assert(conv.solutionParms["AssertStrideBEqual"] == {0:1, filterDims[1]:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[1]:3,filterDims[0]:5})
    run_convolution_level.func(conv, z, run_convolution_level.solution)
