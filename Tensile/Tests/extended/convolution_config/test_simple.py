import logging
from Tensile.SolutionStructs import Convolution

log =logging.getLogger("testlog")

def test_simple(tensile_state, run_convolution_level):
    "Isolated simple test for developement testing"


    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x1',
                      })

    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[0, 3, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2])
    assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
    assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})

    run_convolution_level.func(conv, z, run_convolution_level.solution)
