import pytest
from Tensile.SolutionStructs import Convolution

def test_bad_config():
    z={} # problemType definition
    with pytest.raises(Exception):
        conv = Convolution(z, 'ConvolutionForward',
                  config={'FUBAR': '0',
                      })

def test_bad_tensoraformat():
    z={} # problemType definition
    with pytest.raises(Exception):
        conv = Convolution(z, 'ConvolutionForward',
                  config={'TensorAFormat': 'FUBAR',
                      })

def test_bad_tensorbformat():
    z={} # problemType definition
    with pytest.raises(Exception):
        conv = Convolution(z, 'ConvolutionForward',
                  config={'TensorBFormat': 'FUBAR',
                      })

def test_bad_tensordformat():
    z={} # problemType definition
    with pytest.raises(Exception):
        conv = Convolution(z, 'ConvolutionForward',
                  config={'TensorDFormat': 'FUBAR',
                      })
