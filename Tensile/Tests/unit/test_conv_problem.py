import logging,pytest
from Tensile.SolutionStructs import Convolution,ConvProblem
log =logging.getLogger("testlog")

def test_stride2x3():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
                          config={'TensorAFormat': 'NCHW',
                                  'Stride': '2x3',
                           })
    log.debug(conv.printUsage(z))
    e= { 'n':64, 'c':256, 'h':20, 'w':14, 'k':1024, 'x':1, 'y':1, 'u':2, 'v':3 }
    ec = ConvProblem(e, conv)
    assert (ec.sizes == (4, 10, e['k'], e['n'], e['c']))
    assert (ec.stridesA == (3, 28, -1, -1))

def test_stride2x3_defaults():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
                          config={'TensorAFormat': 'NCHW',
                                  'Stride': '2x3',
                           })
    log.debug(conv.printUsage(z))
    e= { 'n':64, 'c':256, 'h':20, 'w':14, 'k':1024}
    ec = ConvProblem(e, conv)
    assert (ec.sizes == (4, 10, e['k'], e['n'], e['c']))
    assert (ec.stridesA == (3, 28, -1, -1))

def test_bad_filter():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
                          config={'TensorAFormat': 'NCHW',
                           })
    log.debug(conv.printUsage(z))
    e= { 'n':64, 'c':256, 'h':20, 'w':14, 'k':1024, 'x':2, 'y':1, 'u':1, 'v':1 }
    with pytest.raises(RuntimeError, \
            #match="Mismatch between ConvolutionConfig value \(1\) and specified value \(2\) for filter[0]."):
            match="Mismatch between ConvolutionConfig value"):
        ConvProblem(e, conv)


def test_bad_stride():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
                          config={'TensorAFormat': 'NCHW',
                                  'Stride': '2x3',
                           })
    log.debug(conv.printUsage(z))
    e= { 'n':64, 'c':256, 'h':20, 'w':14, 'k':1024, 'x':1, 'y':1, 'u':7, 'v':2 }
    with pytest.raises(RuntimeError, \
            match="Mismatch between ConvolutionConfig value.*stride"):
        ConvProblem(e, conv)

def test_bad_missing_field():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
                          config={'TensorAFormat': 'NCHW',
                                  'Stride': '2x3',
                           })
    log.debug(conv.printUsage(z))
    e= { 'c':256, 'h':20, 'w':14, 'k':1024, 'x':1, 'y':1, 'u':1, 'v':1 }
    with pytest.raises(Exception, match="required ConvProblem field 'n' not present in ConvProblem"):
        ConvProblem(e, conv)

def test_bad_invalid_field():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
                          config={'TensorAFormat': 'NCHW',
                                  'Stride': '2x3',
                           })
    log.debug(conv.printUsage(z))
    e= { 'A': 123, 'n':64, 'c':256, 'h':20, 'w':14, 'k':1024, 'x':1, 'y':1, 'u':1, 'v':1 }
    with pytest.raises(Exception, match="unknown ConvProblem field 'A'"):
        ConvProblem(e, conv)

def test_bad_invalid_list():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
                          config={'TensorAFormat': 'NCHW',
                                  'Stride': '2x3',
                           })
    log.debug(conv.printUsage(z))
    e= [1,2,3,4]
    with pytest.raises(Exception, match="ConvProblem must be a dictionary"):
        ConvProblem(e, conv)
