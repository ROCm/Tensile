import logging
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import YamlBuilder
log =logging.getLogger("testlog")

"""
These tests run the convolution-vs-contraction mode always, 
"""

def test_simple(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x1',
                      'Stride': '1x1',
                      'Dilation': '1x1',
                      'Spatial': '17x31',
                      })
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_stride1x2(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x1',
                      'Stride': '1x2',
                      'Dilation': '1x1',
                      'Spatial': '17x31',
                      })
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_stride2x1(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x1',
                      'Stride': '2x1',
                      'Dilation': '1x1',
                      'Spatial': '17x31',
                      })
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_stride2x3(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x1',
                      'Stride': '2x3',
                      'Dilation': '1x1',
                      'Spatial': '17x31',
                      })
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_filter1x2(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x2',
                      'Stride': '1x1',
                      'Dilation': '1x1',
                      'Spatial': '17x31',
                      })

    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)


def test_filter2x1(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '2x1',
                      'Stride': '1x1',
                      'Dilation': '1x1',
                      'Spatial': '17x31',
                      })
    log.debug(conv.printUsage(z))
    #TODO - re-enable this test
    #YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_filter2x3(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '2x3',
                      'Stride': '1x1',
                      'Dilation': '1x1',
                      'Spatial': '17x31',
                      })
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_dilation1x2(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x1',
                      'Stride': '1x1',
                      'Dilation': '1x2',
                      'Spatial': '17x31',
                      })
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_dilation2x1(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x1',
                      'Stride': '1x1',
                      'Dilation': '2x1',
                      'Spatial': '17x31',
                      })
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_dilation2x3(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x1',
                      'Stride': '1x1',
                      'Dilation': '2x3',
                      'Spatial': '17x31',
                      })
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_filter_stride_dilation_0(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'TensorDFormat': 'NCHW',
                      'Filter': '2x3',
                      'Stride': '2x3',
                      'Dilation': '2x3',
                      'Spatial': '17x31',
                      })
    assert(z['NumIndicesC']==4)
    assert(z['IndexAssignmentsA']==[6,5, 0,1, 4,3])
    assert(z['IndexAssignmentsB']==[6,5, 4, 2, 3])
    assert(z['SetConstStrideA']==[[0,3], [6,3]])
    assert(z['SetConstStrideB']==[[3, 0]])
    assert(z['UseInitialStridesAB'])
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)

def test_filter_stride_dilation_1(request, tensile_client_dir, tmp_path):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '6x7',
                      'Stride': '2x3',
                      'Dilation': '4x5',
                      'Spatial': '27x51',
                      })
    log.debug(conv.printUsage(z))
    YamlBuilder.run_convolution_vs_contraction(request, conv, z, tensile_client_dir, tmp_path)
