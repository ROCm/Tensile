import logging,pytest
from Tensile.SolutionStructs import Convolution
log =logging.getLogger("testlog")

def test_cnhw_defaults(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'CNHW',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[0, 1, 3])
    assert(z['IndexAssignmentsB']==[3, 2, 1])
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")

    solutionName = run_convolution_level.solution.__name__
    if solutionName=="asm3_splitu":
        pytest.skip("bug with asm splitu")

    run_convolution_level.func(conv, z, run_convolution_level.solution)

