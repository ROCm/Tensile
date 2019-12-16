import logging
from Tensile.SolutionStructs import Convolution

log =logging.getLogger("testlog")

def test_yaml(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[0, 3, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2])
    assert(z['SetConstStrideA']==[[0,1]])
    assert(z['SetConstStrideB']==[[2,0]])
    assert(z['UseInitialStridesAB']==0)

    run_convolution_level.func(conv, z, run_convolution_level.solution)
