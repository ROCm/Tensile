import pytest, logging
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import YamlBuilder

log =logging.getLogger("testlog")

def test_yaml(request, tensile_client_dir, tmp_path):
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
    assert(z['UseInitialStrides']==False)

    if request.config.getoption("--run_client"):
        YamlBuilder.run_tensile_client(request, conv, z, tensile_client_dir, tmp_path)
