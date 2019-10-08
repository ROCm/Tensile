import pytest, logging
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import YamlBuilder
log =logging.getLogger("testlog")

def test_nchw_backwardweights_defaults(request, tensile_client):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==2)
    assert(z['IndexAssignmentsA']==[3, 0, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2])
    assert(z['SetConstStrideA']==[[3,1]])
    assert(z['SetConstStrideB']==[])
    if request.config.getoption("--run_client"):
        YamlBuilder.run_tensile_client(request, conv, z, tensile_client)

def test_nchw_backwardweights_filter3x1(request, tensile_client):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x1',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[4, 0, 1, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3])
    #assert(z['SetConstStrideA']==[[3,1]])
    #assert(z['SetConstStrideB']==[])
    if request.config.getoption("--run_client"):
        YamlBuilder.run_tensile_client(request, conv, z, tensile_client)

def test_nchw_backwardweights_filter1x3(request, tensile_client):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x3',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[4, 0, 1, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3])
    #assert(z['SetConstStrideA']==[[3,1]])
    #assert(z['SetConstStrideB']==[])
    if request.config.getoption("--run_client"):
        YamlBuilder.run_tensile_client(request, conv, z, tensile_client)

def test_nchw_backwardweights_filter3x5(request, tensile_client):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionBackwardWeights',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '3x5',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[5, 0, 1, 2, 4])
    assert(z['IndexAssignmentsB']==[5, 3, 4])
    #assert(z['SetConstStrideA']==[[3,1]])
    #assert(z['SetConstStrideB']==[])
    if request.config.getoption("--run_client"):
        YamlBuilder.run_tensile_client(request, conv, z, tensile_client)
