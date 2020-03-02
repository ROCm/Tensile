import logging,pytest
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import defaultSizes, resnetSizes, inceptionSizes
log =logging.getLogger("testlog")

@pytest.mark.parametrize("problemSizes", [defaultSizes, resnetSizes, inceptionSizes])
def test_nchw_filter2x2(tensile_state, run_convolution_level, problemSizes):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '2x2',
                      'PadStart': 'NxN',
                      'PadEnd':   'NxN',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2})

    run_convolution_level.func(conv, z, run_convolution_level.solution, problemSizes[0], problemSizes[1])

@pytest.mark.parametrize("problemSizes", [defaultSizes, resnetSizes, inceptionSizes])
def test_nchw_filter7x1(tensile_state, run_convolution_level, problemSizes):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '7x1',
                      'PadStart': 'Nx0',
                      'PadEnd':   'Nx0',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:7})
    run_convolution_level.func(conv, z, run_convolution_level.solution, problemSizes[0], problemSizes[1])

def test_nchw_filter2x1_dilation(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '2x1',
                      'Dilation': '1x2',
                      'PadStart': 'Nx0',
                      'PadEnd':   'Nx0',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_filter1x2(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '1x2',
                      'PadStart': '0xN',
                      'PadEnd':   '0xN',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1,1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_filter1x2_dilation(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '1x2',
                      'Dilation': '1x2',
                      'PadStart': '0xN',
                      'PadEnd':   '0xN',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:2,1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_dilation_filter4x4_pad(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '4x4',
                      'PadStart': '2x2',
                      'PadEnd':   '2x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        (cdim, filterDims) = (5,[4,3]) if conv.unrollOnChannel else (5,[4,3])
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:2,2:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,filterDims[0]:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:1, filterDims[1]:1})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_stride_filter(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': 'NxN',
                      'Filter': '2x2',
                      'PadStart': 'NxN',
                      'PadEnd':   'NxN',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [5,4] if conv.unrollOnChannel else [6,5]
        cdim = 6 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA']==filterDims + [0, 1, cdim, 3])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 2, 3])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,4:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2, filterDims[1]:2})

    run_convolution_level.func(conv, z, run_convolution_level.solution)
