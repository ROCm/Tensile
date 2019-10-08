import pytest
from Tensile.SolutionStructs import Convolution

# content of test_sample.py
def test_nchw_defaults():
    z={} # problemType definition
    Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      })
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[0, 3, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2])
    assert(z['SetConstStrideA']==[])
    assert(z['SetConstStrideB']==[[2, 0]])

def test_cnhw_defaults():
    z={} # problemType definition
    Convolution(z, 'ConvolutionForward',
              config={
                      'TensorAFormat': 'CNHW',
                      })
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[0, 1, 3])
    assert(z['IndexAssignmentsB']==[3, 2, 1])
    assert(z['SetConstStrideA']==[])
    assert(z['SetConstStrideB']==[[1, 0]])

def test_nhwc_defaults():
    z={} # problemType definition
    Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      })
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[3, 0, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2])
    assert(z['SetConstStrideA']==[])
    assert(z['SetConstStrideB']==[[2, 0]])

def test_nchw_packed_spatial0():
    z={} # problemType definition
    Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'PackedSpatialDims': 0
                      })
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3])
    assert(z['SetConstStrideA']==[])
    assert(z['SetConstStrideB']==[[3, 0]])

def test_nchw_packed_strides():
    z={} # problemType definition
    Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': 'NxN',
                      })
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3])
    assert(z['SetConstStrideA']==[])
    assert(z['SetConstStrideB']==[[3, 0]])

def test_nchw_packed_strides3D():
    z={} # problemType definition
    Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCDHW',
                      'TensorBFormat': 'KCZYX',
                      'TensorDFormat': 'NCDHW',
                      'Stride': 'NxNxN',
                      })
    assert(z['NumIndicesC']==5)
    assert(z['IndexAssignmentsA']==[0, 1, 2, 5, 4])
    assert(z['IndexAssignmentsB']==[5, 3, 4])
    assert(z['SetConstStrideA']==[])
    assert(z['SetConstStrideB']==[[4, 0]])

def test_bad_config():
    z={} # problemType definition
    with pytest.raises(Exception):
        Convolution(z, 'ConvolutionForward',
                  config={'FUBAR': '0',
                      })
