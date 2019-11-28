import logging
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import YamlBuilder
log =logging.getLogger("testlog")

# content of test_sample.py
def test_nchw_defaults(run_convolution_level):
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
    run_convolution_level(conv, z)

def test_cnhw_defaults(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'CNHW',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[0, 1, 3])
    assert(z['IndexAssignmentsB']==[3, 2, 1])
    assert(z['SetConstStrideA']==[[0,1]])
    assert(z['SetConstStrideB']==[[1, 0]])
    assert(z['UseInitialStrides']==False)
    run_convolution_level(conv, z)

def test_nhwc_defaults(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[3, 0, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2])
    assert(z['SetConstStrideA']==[[0,1]])
    assert(z['SetConstStrideB']==[[2, 0]])
    assert(z['UseInitialStrides']==False)
    run_convolution_level(conv, z)

def test_nchw_packed_spatial0(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'PackedSpatialDims': 0
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3])
    assert(z['SetConstStrideA']==[[0,1]])
    assert(z['SetConstStrideB']==[[3, 0]])
    assert(z['UseInitialStrides']==False)
    run_convolution_level(conv, z)

def test_nchw_tbd_strides(run_convolution_level):

    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': 'NxN',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3])
    assert(z['SetConstStrideA']==[])
    assert(z['SetConstStrideB']==[[3, 0]])
    assert(z['UseInitialStrides']==True)
    run_convolution_level(conv, z)

def test_nchw_const_strides(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': '2x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3])
    assert(z['SetConstStrideA']==[[0,2]])
    assert(z['SetConstStrideB']==[[3, 0]])
    assert(z['UseInitialStrides']==True)

    run_convolution_level(conv, z)

def test_nchw_const_use_initial_strides(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': '2x3',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
    assert(z['IndexAssignmentsB']==[4, 2, 3])
    assert(z['SetConstStrideA']==[[0,3]])
    assert(z['SetConstStrideB']==[[3, 0]])
    assert(z['UseInitialStrides']==True)
    run_convolution_level(conv, z)

def test_nchw_filter(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '2x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[5, 4, 0, 3, 2])
    assert(z['IndexAssignmentsB']==[5, 4, 3, 1, 2])
    assert(z['SetConstStrideA']==[[0,1], [5,1]])
    assert(z['SetConstStrideB']==[[2,0]])
    run_convolution_level(conv, z)

def test_nchw_dilation(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Dilation': '2x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[0, 3, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2])
    assert(z['SetConstStrideA']==[[0,1]])
    assert(z['SetConstStrideB']==[[2, 0]])
    run_convolution_level(conv, z)

def test_nchw_stride_filter(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': 'NxN',
                      'Filter': '2x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[6, 5, 0, 1, 4, 3])
    assert(z['IndexAssignmentsB']==[6, 5, 4, 2, 3])
    assert(z['SetConstStrideA']==[[6,1]])
    assert(z['SetConstStrideB']==[[3, 0]])
    run_convolution_level(conv, z)

def test_ncdhw_packed_strides3d_defaults(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCDHW',
                      'Stride': 'NxNxN',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==5)
    assert(z['IndexAssignmentsA']==[0, 1, 2, 5, 4])
    assert(z['IndexAssignmentsB']==[5, 3, 4])
    assert(z['SetConstStrideA']==[])
    assert(z['SetConstStrideB']==[[4, 0]])
    run_convolution_level(conv, z)

def test_ncdhw_packed_strides_filter3d(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCDHW',
                      'TensorBFormat': 'KCZYX',
                      'TensorDFormat': 'NCDHW',
                      'Stride': 'NxNxN',
                      'Filter': '3x3x3',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==5)
    assert(z['IndexAssignmentsA']==[8,7,6, 0, 1, 2, 5, 4])
    assert(z['IndexAssignmentsB']==[8,7,6, 5, 3, 4])
    assert(z['SetConstStrideA']==[[8,1]])
    assert(z['SetConstStrideB']==[[4, 0]])
    run_convolution_level(conv, z)

def test_ncdhw_packed_strides3d(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCDHW',
                      'TensorBFormat': 'KCZYX',
                      'TensorDFormat': 'NCDHW',
                      'Stride': 'NxNxN',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==5)
    assert(z['IndexAssignmentsA']==[0, 1, 2, 5, 4])
    assert(z['IndexAssignmentsB']==[5, 3, 4])
    assert(z['SetConstStrideA']==[])
    assert(z['SetConstStrideB']==[[4, 0]])
    run_convolution_level(conv, z)
