import logging
from Tensile.SolutionStructs import Convolution
log =logging.getLogger("testlog")

def test_spatial_in():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW'
                     })

    log.debug(conv.printUsage(z))
    p = conv.makeProblem(False, n=64, c=1024, k=256, spatialIn=[14,15])
    assert(p[0] == [210, 256, 64, 1024])
    assert(p[1] == [1, -1, -1])

def test_spatial_parm():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '13x14',
                     })

    log.debug(conv.printUsage(z))
    p = conv.makeProblem(False, n=64, c=1024, k=256)
    assert(p[0] == [182, 256, 64, 1024])
    assert(p[1] == [1, -1, -1])


def test_stride():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '13x14',
                      'Stride' : '2x3',
                     })

    log.debug(conv.printUsage(z))
    p = conv.makeProblem(False, n=64, c=1024, k=256)
    assert(p[0] == [4, 6, 256, 64, 1024])
    assert(p[1] == [3, 28, -1, -1])


def test_stride_filter():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '13x14',
                      'Stride' : '2x3',
                      'Filter' : '3x4',
                     })

    log.debug(conv.printUsage(z))
    p = conv.makeProblem(False, n=64, c=1024, k=256)
    assert(p[0] == [3, 5, 256, 64, 3, 4, 1024])
    assert(p[1] == [1, 14, 3, 28, -1, -1])

def test_stride_filter_dilated():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '13x14',
                      'Stride' : '2x3',
                      'Filter' : '3x4',
                      'Dilation': '2x3',
                     })

    log.debug(conv.printUsage(z))
    p = conv.makeProblem(False, n=64, c=1024, k=256)
    assert(p[0] == [3, 5, 256, 64, 3, 4, 1024])
    assert(p[1] == [3, 28, 3, 28, -1, -1])

def test_spatial_unspecified():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                     })

    log.debug(conv.printUsage(z))
    p = conv.makeProblem(True, n=64, c=1024, k=256)
    assert(p[0] == [-1, 256, 64, 1024])
    assert(p[1] == [1, -1, -1])

