import logging,pytest
from Tensile.SolutionStructs import Convolution
log =logging.getLogger("testlog")

def test_nchw_backwardweights_defaults(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '14x14',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==2)
    assert(z['IndexAssignmentsA']==[3, 0, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2, 0])
    #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,0:1")
    #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
    assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_backwardweights_filter3x1(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x1',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[4, 0, 1, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3, 1])
    #assert(conv.solutionParms["AssertStrideAEqual"] == "2:1,0:1")
    #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
    assert(conv.solutionParms["AssertSizeEqual"] == {0:3})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_backwardweights_filter1x3(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x3',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[4, 0, 1, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3, 1])
    #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,2:1,0:1")
    #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
    assert(conv.solutionParms["AssertSizeEqual"] == {0:3})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_backwardweights_filter3x5(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x5',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[5, 0, 1, 2, 4])
    assert(z['IndexAssignmentsB']==[5, 3, 4, 2])
    assert(z['SetConstStrideA'] == [[0, 1], [5, 1]])
    assert(z['SetConstStrideB'] == [[2,0], [5,1]])
    #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,3:1,0:1")
    #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
    assert(conv.solutionParms["AssertSizeEqual"] == {0:5, 1:3})
    run_convolution_level.func(conv, z, run_convolution_level.solution)


def test_nchw_backwardweights_filter3x5_nopack(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x5',
                      'PackedSpatialDims': 0,
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[6, 5, 0, 1, 2, 4])
    assert(z['SetConstStrideA'] == [[0, 1], [6, 1]])
    assert(z['SetConstStrideB'] == [[2,0], [6,1]])

    assert(z['IndexAssignmentsB']==[6, 5, 3, 4, 2])
    #assert(conv.solutionParms["AssertStrideAEqual"] == "1:1,3:1,0:1")
    #assert(conv.solutionParms["AssertStrideBEqual"] == "1:1")
    assert(conv.solutionParms["AssertSizeEqual"] == {0:5, 1:3})
    run_convolution_level.func(conv, z, run_convolution_level.solution)
