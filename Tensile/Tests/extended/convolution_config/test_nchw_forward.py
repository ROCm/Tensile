import logging,pytest
from Tensile.SolutionStructs import Convolution
log =logging.getLogger("testlog")

def test_nchw_defaults(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[0, 3, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2])
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

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

def test_nhwc_defaults(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[3, 0, 2])
    assert(z['IndexAssignmentsB']==[3, 1, 2])
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nhwc_filter2x2(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      'Filter': '2x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[3, 5, 4, 0, 2])
    assert(z['IndexAssignmentsB']==[5, 4, 3, 1, 2])
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,4:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

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
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")

    solutionName = run_convolution_level.solution.__name__
    if solutionName=="asm3_splitu":
        pytest.skip("bug with asm splitu")

    run_convolution_level.func(conv, z, run_convolution_level.solution)

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
    assert(z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

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
    assert(z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:2")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

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
    assert(z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:3")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_filter2x2(run_convolution_level):
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
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1,2:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,4:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_filter2x1(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '2x1',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[4, 0, 3, 2])
    assert(z['IndexAssignmentsB']==[4, 3, 1, 2])
    assert(z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "1:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "3:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_filter2x1_dilation(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '2x1',
                      'Dilation': '1x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[4, 0, 3, 2])
    assert(z['IndexAssignmentsB']==[4, 3, 1, 2])
    assert(z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "1:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "3:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_filter1x2(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '1x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[4, 0, 3, 2])
    assert(z['IndexAssignmentsB']==[4, 3, 1, 2])
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1,1:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,3:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_filter1x2_dilation(run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '1x2',
                      'Dilation': '1x2',
                      })
    log.debug(conv.printUsage(z))
    assert(z['NumIndicesC']==3)
    assert(z['IndexAssignmentsA']==[4, 0, 3, 2])
    assert(z['IndexAssignmentsB']==[4, 3, 1, 2])
    assert(z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:2,1:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,3:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

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
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

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
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,4:0")

    solutionName = run_convolution_level.solution.__name__
    if solutionName=="asm3_splitu":
        pytest.skip("bug with asm splitu")

    run_convolution_level.func(conv, z, run_convolution_level.solution)

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
    assert(z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

@pytest.mark.skip(reason="out of registers in asm runs")
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
    assert(not z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "0:1")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,5:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)

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
    assert(z['UseInitialStridesAB'])
    assert(conv.solutionParms["AssertStrideAEqual"] == "")
    assert(conv.solutionParms["AssertStrideBEqual"] == "0:1,2:0")
    run_convolution_level.func(conv, z, run_convolution_level.solution)
