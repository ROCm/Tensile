################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import sys
import operator
from collections import namedtuple,OrderedDict
from warnings import warn
from functools import reduce
from .Common import globalParameters, defaultProblemType, assignParameterWithDefault, printExit, assignParameterRequired, defaultSolution, validParameters, print1
from .Common import validActivationFormats, validWeightFormats, validConvolutionConfig, validMFMA
from copy import deepcopy
import math
from .Utils import roundUpToNearestMultiple
from .DataType import DataType
from enum import Enum

########################################
# Print a reject message :
def reject(state, *args):
  if globalParameters["PrintSolutionRejectionReason"]:
    sys.stdout.write("\nreject: ")
    for a in args:
      print(a)
    #traceback.print_stack(None, 2)
  if state != None:
    state["Valid"] = False

# print a labled variable
def pvar(state, field):
  return field + "=" + str(state[field])

def roundupRatio(dividend, divisor):
  return int(math.ceil(float(dividend) / float(divisor)))

class Fbs(Enum):
  Free=0     # Expect to be free dimension
  Batch=1    # Expect to be batch dimension
  Sum=2      # Expect to be summation dimension

################################################################################
RegDim=namedtuple("RegDim", ["idx", "fbs", "dim"])

class ConvolutionConfig:
  def __init__(self, fil=None, stride=None, dilation=None, \
                  spatial=None, groupCount=1, \
                  padStart=None, padEnd=None):
    self.fil = fil
    self.stride = stride
    self.dilation = dilation
    self.spatial = spatial
    self.groupCount = groupCount
    self.padStart = padStart
    self.padEnd = padEnd

  @staticmethod
  def copyField(tag, selfValues, refValues):
    """ Use selfValues if specified (after validating they match cc), or conv config values if not"""
    if selfValues:
      if refValues:
          assert(len(selfValues) == len(refValues))
          for (i,(selfVal, refVal)) in enumerate(zip(selfValues, refValues)):
            if selfVal == -1:
              selfValues[i] = refVal
            if selfVal != -1 and refVal != -1 and refVal != selfVal:
              raise RuntimeError("Mismatch between ConvolutionConfig value (%d) and ConvProblem value (%d) for %s[%d]." %
                        (refVal, selfVal, tag, i))
      return selfValues
    else:
      return refValues

  def copyFromRef(self, ref):
    """
    For all fields which are -1 in self, copy from reference implementation.
    For any fields that are specified in self (not -1), ensure they match reference
    """
    self.fil = self.copyField("filter", self.fil, ref.fil)
    self.stride = self.copyField("stride", self.stride, ref.stride)
    self.dilation = self.copyField("dilation", self.dilation, ref.dilation)
    self.spatial = self.copyField("spatial", self.spatial, ref.spatial)
    self.padStart = self.copyField("padStart", self.padStart, ref.padStart)
    self.padEnd = self.copyField("padEnd", self.padEnd, ref.padEnd)

    if self.groupCount == -1:
      self.groupCount = ref.groupCount

  def checkFullySpecified(self, ref):
    """
    Throw exception if the config is not fully specified.
    """
    for field in ('fil', 'stride', 'dilation', 'spatial', 'groupCount', 'padStart', 'padEnd'):
        val = getattr(self,field)
        if val==None:
          raise RuntimeError("ConvolutionConfig field '%s' == None and must be specified'" % field)
        elif isinstance(val,int) and val==-1 or type(val) in (tuple,list) and -1 in val:
          raise RuntimeError("ConvolutionConfig field '%s' == %s must be fully specified.'" % (field,val))


  def __str__ (self):
      return("filter:%s stride:%s dilation:%s spatial:%s group:%d padStart:%s padEnd:%s" \
            % (str(self.fil) if self.fil else "tbd",
               self.stride, self.dilation, \
               str(self.spatial) if self.spatial else "tbd", \
               self.groupCount, self.padStart, self.padEnd))

class Convolution:
  class Dimension:
    """
    A description of the dimension - short char, usage, and const strides
    Dimensions are later assigned tensile indices and assigned to A/B
    based on the desired formats.
    """
    # stride=-1 indicates TBD stride; >=0 indicates a compile-time constant
    def __init__(self, shortChar, description, size=-1, strideA=-1, strideB=-1):
      self.shortChar = shortChar
      self.description = description
      self.size=size
      self.strideA=strideA
      self.strideB=strideB

    def __str__(self):
      s = "%5s : %s" % ("'%s'"%self.shortChar, self.description)
      if self.size != -1:
        s+=" [size:%d]"%self.size
      if self.strideA != -1:
        s+=" [strideA:%d]"%self.strideA
      if self.strideB != -1:
        s+=" [strideB:%d]"%self.strideB
      return s
    def __repr__(self):
      return self.shortChar

  SummaryProblemProperties=[\
        'OperationType','DestDataType','DataType','HighPrecisionAccumulate',\
        'TensorAFormat','TensorBFormat','TensorDFormat',\
        'Filter', 'Stride','Dilation','PadStart','PadEnd','GroupCount',\
        'NumIndicesC', 'IndexAssignmentsA','IndexAssignmentsB',\
        'IndicesFree', 'IndicesBatch', 'IndicesSummation',\
        'SetConstStrideA', 'SetConstStrideB', 'ZeroPadA', \
        'UseBeta', 'UseInitialStridesAB', "AllowNoFreeDims", \
        ]
  SummarySolutionProperties=[\
        'AssertSizeEqual', 'AssertStrideAEqual', 'AssertStrideBEqual', \
        ]

  # valid lowest filter dimensions, these we can attach compile-time constant strides:
  ValidLowestFilterDim= ('X','XY', 'XYZ', 'W', 'HW', 'DHW')

  def initForwardConvolution(self, problemTypeOut, config, \
                             formatA, formatB, formatD,
                             ndim, cdim, kdim, sdims, fdims, \
                             fil, stride, dilation):
    """
    Output : registerA and registerB
    """
    # Make index assignments following standard Tensile Index assignment rules (see Common.py)
    # - Indices < NumCindices are batch or free indices and are present in TensorD
    # - Indices >= NumCindices are summation indices.  cidx is cin / summation so must be after nidx
    # - Memory order for TensorD is NumCindices...0, with 0 the fastest-moving dim.

    # The index assignment is captured in the RegDim.idx parm. This idx is in global space.

    # Specific assignments to A and B (and associated impact on memory order of those tensors) is
    # specified by order of parms to registerA and registerB below.

    # Control output space dimension order:S
    # Note:
    #   - 'C' in output format refers to output channels, ie 'K'
    #   - backward-weight formats swap (C,K)
    #   - fastest moving in format is rightmost and should have idx=0, then
    #     increase index moving left through the format.
    #print ("formatA=", formatA, "formatB=", formatB, "formatD=", formatD)

    self.normalizedCC = ConvolutionConfig(fil=fil, stride=stride, dilation=dilation, \
                                          spatial=None)
    if formatD in ('NCHW','NCDHW'):
      sidx = 0
      i = len(sdims)
      kidx = i ; i+=1
      nidx = i ; i+=1
      sumIdx = i
    elif formatD in ('NHWC','NDHWC'):
      i = 0
      kidx = i ; i+=1
      sidx = i ; i+=len(sdims)
      nidx = i ; i+=1
      sumIdx = i
    elif formatD in ("CNHW", "CNDHW"):
      sidx = 0
      # re-order batch dim to control memory order in output space
      i = len(sdims)
      nidx = i ; i+=1
      kidx = i ; i+=1
      sumIdx = i
    else:
      raise RuntimeError ("unknown formatD '%s'"%formatD)

    if not self.unrollOnChannel:
      # place cidx at lowest summation then filters -> filters are unroll
      cidx = sumIdx
      sumIdx = sumIdx+1

    self.filterRegDims = []
    for filterDim in fdims:
      self.filterRegDims.append( RegDim(sumIdx, Fbs.Sum, filterDim) )
      sumIdx = sumIdx+1


    if self.unrollOnChannel:
      # place cidx at highest summation, after filters
      cidx = sumIdx

    self.spatialRegDims = []
    # reverse dims  so can pass spatialRegDims to register functions in 'convolution' order
    for si,sdim in enumerate(sdims):
      self.spatialRegDims.insert(0, RegDim(sidx+si, Fbs.Free, sdim))

    chinRegDim = [RegDim(cidx,Fbs.Sum,cdim)]
    choutRegDim= [RegDim(kidx,Fbs.Free,kdim)]

    if formatA in ("NCHW", "NCDHW"):
      self.registerA( [RegDim(nidx,Fbs.Batch,ndim)] + chinRegDim + self.spatialRegDims + self.filterRegDims )
    elif formatA in ("NHWC", "NDHWC"):
      self.registerA( [RegDim(nidx,Fbs.Batch,ndim)] + self.spatialRegDims + self.filterRegDims + chinRegDim )
    elif formatA in ("CNHW", "CNDHW"):
      self.registerA( chinRegDim + [RegDim(nidx,Fbs.Batch,ndim)] + self.spatialRegDims + self.filterRegDims )
    else:
      raise RuntimeError ("unknown formatA '%s'"%formatA)

    ndim.strideB = 0
    if formatB in ("KCYX",'KCZYX') :
      self.registerB( [RegDim(nidx,Fbs.Batch,ndim)] + choutRegDim + chinRegDim + self.filterRegDims )
    elif formatB in ("CKYX",'CKZYX'):
      self.registerB( [RegDim(nidx,Fbs.Batch,ndim)] + chinRegDim + choutRegDim + self.filterRegDims )
    elif formatB in ("CYXK",'CZYXK'):
      self.registerB( [RegDim(nidx,Fbs.Batch,ndim)] + chinRegDim + self.filterRegDims + choutRegDim )
    elif formatB in ("KYXC",'KZYXC'):
      self.registerB( [RegDim(nidx,Fbs.Batch,ndim)] + choutRegDim + self.filterRegDims + chinRegDim )
    else:
      raise RuntimeError ("unknown formatB '%s'"%formatB)

    problemTypeOut["NumIndicesC"] = 2+len(self.spatialRegDims)

    problemTypeOut["ZeroPadA"] = self.makeZeroPadConvProblemType(self.cc.padStart, self.cc.padEnd)

    # Attach constant strides to A, if possible:
    nonFilterDims = [dim for dim in self.regDimsA if dim not in self.filterRegDims]
    setStride=False
    if sdims and nonFilterDims[-1].dim == sdims[0]:
      setStride = True
      sdims[0].strideA = self.cc.stride[0]
    elif nonFilterDims[-1].dim==cdim:
      setStride = True
      cdim.strideA = 1

    if self.filterRegDims and self.regDimsA[-1].dim == self.filterRegDims[-1].dim:
      if self.filterRegDims[-1].dim.shortChar in self.ValidLowestFilterDim:
        self.filterRegDims[-1].dim.strideA = self.cc.dilation[0]
    elif not setStride:
      raise RuntimeError ("unexpected lowest dimension in tensorAFormat(%s)"%self.tensorAFormat)

    # Attach constant strides to B, if possible:
    if self.regDimsB[-1].dim == cdim:
      cdim.strideB=1
    elif self.regDimsB[-1].dim == kdim:
      kdim.strideB=1
    elif self.filterRegDims and self.regDimsB[-1].dim == self.filterRegDims[-1].dim:
      if self.filterRegDims[-1].dim.shortChar in self.ValidLowestFilterDim:
        self.filterRegDims[-1].dim.strideB = 1
    else:
      raise RuntimeError ("unexpected lowest dimension in tensorBFormat(%s)"%self.tensorAFormat)

  @staticmethod
  def swap(targetStr, replStr1, replStr2):
    tmp = '$'
    assert(tmp not in targetStr)
    for (char1,char2) in zip(replStr1,replStr2):
      if char1==',':
        continue
      targetStr = targetStr.replace(char1, tmp).replace(char2, char1).replace(tmp, char2)
    return targetStr

  def makeZeroPadConvProblemType(self, padStart, padEnd):
    """
    Convert padStart/padEnd into the format expected by ProblemType ZeroPad*
    Tensile drops any compile-time padding info here; this must be provided with each problem.
    """
    rv = []
    spatialChars='WHD'
    filterChars='XYZ'
    for i in range(self.numSpatialDims):
      if padStart[i] or padEnd[i]:
          anchorIdx = self.convolutionDims[spatialChars[i]].idx
          sumIdx    = self.convolutionDims[filterChars[i]].idx
          rv.append([anchorIdx, sumIdx, -1, -1])
    return rv

  def makeZeroPadProblemType(self, zps, padStart, padEnd, cc):
    """ Convert padStart/padEnd into the format expected by ProblemType ZeroPad* """
    rv = []
    ss = 1
    for (i,zp) in enumerate(zps):
      (anchorIdx, sumIdx) = zp[:2]
      rv.append([anchorIdx, sumIdx, padStart[i]*ss, padEnd[i]*ss])
      assert(cc.spatial[i] != -1)
      ss *= cc.spatial[i]
    return rv

  def __init__(self, problemTypeOut, convolutionType, config):
    """
    problemTypeOut contains problem type parms created by this constructor.
    """

    self.convolutionDims={};
    self.convolutionType = convolutionType
    self.config = config # input configuraiton
    self.problemTypeOut = problemTypeOut
    self.cc = ConvolutionConfig() # parsed configuration

    for k in config:
      if k not in validConvolutionConfig:
        raise RuntimeError ("unknown convolution config field '%s'"%k)

    self.tensorAFormat = config.get("TensorAFormat", "NCHW")
    assert self.tensorAFormat in validActivationFormats
    self.formatNumSpatialDims = len(self.tensorAFormat)-2
    assert (self.formatNumSpatialDims>=2 and self.formatNumSpatialDims<=3)

    if convolutionType in ('ConvolutionForward', 'ConvolutionBackwardData'):
      defaultFormatB = "KCYX" if self.formatNumSpatialDims==2 else 'KCZYX'
      defaultFormatD = self.tensorAFormat
    elif convolutionType == 'ConvolutionBackwardWeights':
      defaultFormatB = "NCHW" if self.formatNumSpatialDims==2 else 'NCDHW'
      defaultFormatD = "KCYX" if self.formatNumSpatialDims==2 else 'KCZYX'

    self.tensorBFormat = config.get("TensorBFormat", defaultFormatB)
    self.tensorDFormat = config.get("TensorDFormat", defaultFormatD)

    if convolutionType in ('ConvolutionForward', 'ConvolutionBackwardData'):
      assert self.tensorBFormat in validWeightFormats
      assert self.tensorDFormat in validActivationFormats
    elif convolutionType in ('ConvolutionBackwardWeights'):
      assert self.tensorBFormat in validActivationFormats
      assert self.tensorDFormat in validWeightFormats

    if self.tensorDFormat == 0:
      self.tensorDFormat = self.tensorAFormat
    assert len(self.tensorAFormat) == len(self.tensorBFormat) == len(self.tensorDFormat)

    # index 0,1,2 = W,H,D = X,Y,Z
    if config.get("Spatial",None):
      self.cc.spatial  = self.dimxParm(config, "Spatial",-1)
    else:
      self.cc.spatial = None
    self.cc.fil   = self.dimxParm(config, "Filter",1)
    self.cc.stride   = self.dimxParm(config, "Stride",1)
    self.cc.dilation = self.dimxParm(config, "Dilation",1)
    self.cc.padStart = self.dimxParm(config, "PadStart",0)
    self.cc.padEnd   = self.dimxParm(config, "PadEnd",0)
    self.packedSpatialDims = config.get("PackedSpatialDims", 1)
    self.packedFilterDims  = config.get("PackedFilterDims", 1)
    self.unrollOnChannel = config.get("UnrollOnChannel", 1)

    assert(type(self.packedFilterDims) == int)
    assert(type(self.packedSpatialDims) == int)
    assert(type(self.unrollOnChannel) == int)
    if not all(i==1 for i in self.cc.dilation[1:]):
      self.packedFilterDims = 0
    if not (\
       all(i==1 for i in self.cc.stride[1:]) and \
       all(i==0 for i in self.cc.padStart) and \
       all(i==0 for i in self.cc.padEnd) \
       ):
      self.packedSpatialDims = 0

    assert (len(self.cc.fil)==len(self.cc.stride)==len(self.cc.dilation) \
            ==len(self.cc.padStart)==len(self.cc.padEnd))

    self.groupCount = config.get("GroupCount", 1)
    self.indexAssignments = []

    # Index assignment have fastest-moving first
    ndim = Convolution.Dimension('N',   'Minibatch dimension. size#T=N.')
    kdim = Convolution.Dimension('K',   'Cout. size#T=Cout.')
    cdim = Convolution.Dimension('C', 'Cin.  size#T=Cin.')

    if self.packedSpatialDims:
      if self.formatNumSpatialDims==2:
        sdims = [Convolution.Dimension('HW', \
            'Spatially packed HW. size#T=H_o*W_o. strideA#T=strideW(#S0).')]
      elif self.formatNumSpatialDims==3:
        sdims = [Convolution.Dimension('DHW', \
            'Spatially packed DHW. size#T=D_o*H_o*W_o. strideA#T=strideW(#S0).')]
      else:
        raise RuntimeError ("unsupported formatNumSpatialDims")
    else:
      sdims = []
      schars = [1,'W','H','D']
      # sdims[0] is W
      for si in range(self.formatNumSpatialDims):
        sc=schars[si+1]
        if si==0:
            strideMsg = "stride%s(#S0)"%sc
        else:
            strideMsg = "%s_in*stride%s(#S%d)"%(schars[si],sc,si)
        sdims.append(Convolution.Dimension(sc,  \
            'Spatial %s. size#T=%s_o strideA#T=%s.'%(sc,sc,strideMsg)))

    # dims actually used in the tensor.
    self.numSpatialDims = len(sdims)
    if self.packedSpatialDims:
      assert (self.numSpatialDims <= self.formatNumSpatialDims)
    else:
      assert (self.numSpatialDims == self.formatNumSpatialDims)

    fdims = []
    for (rfi,filterValue) in enumerate(self.cc.fil[::-1]):
      if not self.packedFilterDims or filterValue != 1:
        fi = self.formatNumSpatialDims - rfi - 1 # forward filter index, 0...
        filterChar = chr(ord('X')+fi)
        filterValueStr = "TBD" if filterValue==-1 else str(filterValue)
        prevChar = ['1', 'W', 'W*H']
        # TODO - stride setconst maybe applies only for NCHW/CNHW format not NHWC
        # can modify message here based on format or position of indices?
        filterMsg = "Filter%s. size#T=Filter%s(%s). strideA#T=Dilation%s(#D%d)*%s." \
            % (filterChar, filterChar, filterValueStr, filterChar, fi, \
               prevChar[fi])
        fdims.append(Convolution.Dimension(filterChar, filterMsg, size=filterValue))

    # Create summation dimensions for non-unit filters and assign summation indices
    assert(len(self.cc.fil)) == self.formatNumSpatialDims

    # Output format: C->K -> doesn't matter
    # what about if both C and K present in output (ie weights)  CK
    if convolutionType in ("ConvolutionForward"):
      self.initForwardConvolution(problemTypeOut, config, \
                                self.tensorAFormat, self.tensorBFormat, self.tensorDFormat,
                                ndim=ndim, cdim=cdim, kdim=kdim, sdims=sdims, fdims=fdims, \
                                fil=self.cc.fil, stride=self.cc.stride, dilation=self.cc.dilation)
    elif convolutionType in ("ConvolutionBackwardData"):
      # swaps cdim and kdim
      formatB=self.swap(self.tensorBFormat, 'C', 'K')
      self.initForwardConvolution(problemTypeOut, config, \
                                self.tensorAFormat, formatB, self.tensorDFormat,
                                ndim=ndim, cdim=kdim, kdim=cdim, sdims=sdims, fdims=fdims, \
                                fil=self.cc.fil, stride=self.cc.dilation, dilation=self.cc.stride)
    elif convolutionType in ("ConvolutionBackwardWeights"):
      # swaps ndim and cdim; filter and spatial
      formatA=self.swap(self.tensorBFormat, 'C', 'N')
      # convert activation->weight format, ie NCHW->KCYX
      formatB=self.swap(self.tensorBFormat, 'WHD', 'XYZ').replace('C','K').replace('N','C')
      formatD=self.swap(self.tensorDFormat, 'C,K,XYZ', 'N,C,WHD')  # ie CKYX -> NCHW
      self.initForwardConvolution(problemTypeOut, config, \
                                  formatA, formatB, formatD,
                                  ndim=cdim, cdim=ndim, kdim=kdim, sdims=list(reversed(fdims)), \
                                  fdims=list(reversed(sdims)), \
                                  fil=self.cc.spatial, stride=self.cc.dilation, \
                                  dilation=self.cc.stride)
      # fdims (filter dims) become the free dims for backward-weights.
      # if these dims have size==1 and stride==default they may be collapsed to empty list.
      # set AllowNoFreeDims to tell Tensile to use the batch dim as a virtual free dims
      # this forces PackBatchDims and sets Index* appropriately.
      if not fdims:
        problemTypeOut["AllowNoFreeDims"] = True

    # convert from convolution order to tensor order:
    self.regDimsA.reverse()
    self.regDimsB.reverse()

    problemTypeOut["IndexAssignmentsA"] = [x[0] for x in self.regDimsA]
    problemTypeOut["IndexAssignmentsB"] = [x[0] for x in self.regDimsB]
    problemTypeOut["UseBeta"] = False # MI kernels don't use beta


    self.solutionParms = {}

    stridea=[]
    for (idx,fbs,dim) in self.regDimsA:
      if dim.strideA != -1:
        stridea.append([idx, dim.strideA])
    stridea.sort()
    self.solutionParms["AssertStrideAEqual"] = \
            {problemTypeOut["IndexAssignmentsA"].index(s[0]) : s[1] for s in stridea}
    problemTypeOut["SetConstStrideA"] = stridea

    strideb=[]
    for (idx,fbs,dim) in self.regDimsB:
      if dim.strideB != -1:
        strideb.append([idx,dim.strideB])
    strideb.sort()

    self.solutionParms["AssertStrideBEqual"] = \
            {problemTypeOut["IndexAssignmentsB"].index(s[0]) : s[1] for s in strideb}
    problemTypeOut["SetConstStrideB"] = strideb

    self.solutionParms["AssertSizeEqual"] = {regDim.idx:regDim.dim.size for regDim in self.indexAssignments if regDim.dim.size != -1}

    if self.solutionParms["AssertStrideAEqual"].get(0,-1) == 1 and \
       self.solutionParms["AssertStrideBEqual"].get(0,-1) == 1:
      # optimize if no initial stride needed in A or B
      # allow yaml to override this for testing UseInitialStridesAB
      if "UseInitialStridesAB" not in problemTypeOut:
        problemTypeOut["UseInitialStridesAB"] = False
    else:
      problemTypeOut["UseInitialStridesAB"] = True

    iaa = problemTypeOut["IndexAssignmentsA"]
    iab = problemTypeOut["IndexAssignmentsB"]
    allIndices = list(range(len(self.indexAssignments)))
    self.sumIndices  = list(range(problemTypeOut["NumIndicesC"], len(self.indexAssignments)))
    self.batchIndices = [idx for idx in allIndices \
                   if idx in iaa and idx in iab and idx not in self.sumIndices]
    self.freeIndices = [idx for idx in allIndices \
                   if idx not in self.sumIndices and idx not in self.batchIndices]
    self.checkDims(self.freeIndices, self.batchIndices, self.sumIndices)


  def dimIdx(self, convolutionChar):
    return self.convolutionDims[convolutionChar].idx

  def convolutionChar(self, dimIdx):
    return self.indexAssignments[dimIdx].dim.shortChar

  def markedConvolutionChar(self, dimIdx, tc):
    regDim = self.indexAssignments[dimIdx]
    assert tc in ('A','B')
    if tc=='A' and regDim.fbs==Fbs.Sum and regDim.dim.shortChar not in ['C','K','N']:
        return "_" + regDim.dim.shortChar
    else:
        return regDim.dim.shortChar

  def makeProblem(self, n, c, k, pcc):
    """
    Generate valid problem dims for specified convolution
    pcc is a ConvolutionConfig class with specified values for this problem.
    The function will attempt to initialize TBD values in pcc from the convolution base class,
    and will then check to ensure the problem is fully specified.

    Return [ [sizes], [stridesA] ]
    """
    numDims = 1 + max(max([x[0] for x in self.regDimsA]), max([x[0] for x in self.regDimsB]))
    sizes = [-1]*numDims
    astrides = [-1]*numDims
    bstrides = [-1]*numDims

    pcc.copyFromRef(self.cc)
    pcc.checkFullySpecified(self.cc)

    sizes[self.convolutionDims['N'].idx]=n
    sizes[self.convolutionDims['C'].idx]=c
    sizes[self.convolutionDims['K'].idx]=k

    bstrides[self.convolutionDims['N'].idx] = 0 # broadcast b matrix

    if len(pcc.spatial) != self.formatNumSpatialDims:
      raise RuntimeError ("len(pcc.spatial=", pcc.spatial, ") must match formatNumSpatialDims(%d)"%self.formatNumSpatialDims)

    # convert to Output dimensions:
    spatialOut=[0]*len(pcc.spatial)
    for i in range(self.formatNumSpatialDims):
      spatialOut[i] = int((pcc.spatial[i] - pcc.fil[i] + 1 + pcc.padStart[i] + pcc.padEnd[i]) / pcc.stride[i])

    #print ("spatialOut=", spatialOut, "padStart=", pcc.padStart, "padEnd=", pcc.padEnd)

    for fi,filterValue in enumerate(pcc.fil):
      try:
        pos = self.convolutionDims[chr(ord('X')+fi)].idx
        sizes[pos] = filterValue
        astrides[pos] = pcc.dilation[0] if fi==0 else pcc.spatial[fi-1]*pcc.dilation[fi]
      except KeyError:
        None

    if self.numSpatialDims==1:
      spatialName="DHW"[3-self.formatNumSpatialDims:]
      pos=self.convolutionDims[spatialName].idx
      sizes[pos] = reduce((lambda x, y: x * y), spatialOut) # product of all spatial dimes
      astrides[pos] = pcc.stride[0]
    else:
      for si,sout in enumerate(spatialOut):
        spatialChars=['W','H','D']
        pos = self.convolutionDims[spatialChars[si]].idx
        sizes[pos] = sout

        astrides[pos]=pcc.stride[0] if si==0 else pcc.spatial[si-1]*pcc.stride[si]

    assert all(i!=-1 for i in sizes)

    # translate to strides for A tensor in IndexAssignmentsA order:
    orderedStridesA = []
    orderedStridesB = []
    for (idx,fbs,dim) in self.regDimsA:
      orderedStridesA.append(astrides[idx])

    for (idx,fbs,dim) in self.regDimsB:
      orderedStridesB.append(bstrides[idx])

    #print("ordered=A", orderedStridesA, "b=", orderedStridesB)

    return (sizes, orderedStridesA, orderedStridesB)

  def registerA(self, regDimList):
    """
    Provide a list of indices in convolution order - these will be reversed when assigned to IndexAssignmentsAB
    The order of items in the list determines the IndexAssignment order.
    Each tuple in the list is a RegDim class.
     - idx is the tensor index
     - fbs indicates if the tensor is expected to be Free, Sum, or Batch.  This is used for later check.
     - dim is Convolution.Dimension class that describes the dimension (for Usage info)
    """
    for regDim in regDimList:
      try:
        self.indexAssignments[regDim.idx]
      except IndexError:
        self.indexAssignments.extend([None]*(1+regDim.idx-len(self.indexAssignments)))
      assert(self.indexAssignments[regDim.idx] == None or \
             self.indexAssignments[regDim.idx] == regDim)
      self.indexAssignments[regDim.idx] = regDim
      self.convolutionDims[regDim.dim.shortChar] = regDim
    self.regDimsA = regDimList

  def registerB(self, regDimList):
    """
    See registerA
    """
    for regDim in regDimList:
      try:
        self.indexAssignments[regDim.idx]
      except IndexError:
        self.indexAssignments.extend([None]*(1+regDim.idx-len(self.indexAssignments)))
      assert(self.indexAssignments[regDim.idx] == None or \
             self.indexAssignments[regDim.idx] == regDim)
      self.indexAssignments[regDim.idx] = regDim
      self.convolutionDims[regDim.dim.shortChar] = regDim

    self.regDimsB = regDimList

  def dimxParm(self, config, parmName, default):
    parm =config.get(parmName)
    if not parm:
      rv = [default ] * self.formatNumSpatialDims
    else:
      rv=[]
      for x in parm.split('x'):
        if x.upper()=='N':
          rv.append(-1)
        else:
          rv.append(int(x))
      rv.reverse() # rightmost number is 0
    if len(rv) != self.formatNumSpatialDims:
        raise RuntimeError ("%s parm '%s' must have %d spatial dims'"%(parmName, parm, self.formatNumSpatialDims))
    return rv

  def printUsage(self, problemType, details=False):
    print()
    print("Tensor Formats: A:%s B:%s D:%s\n" % (self.tensorAFormat, self.tensorBFormat, self.tensorDFormat))
    print("Input Conv: %s packedFilter:%d packedSpatiol:%d unrollOnChannel:%d\n" % \
            (str(self.cc), self.packedFilterDims, self.packedSpatialDims, \
             self.unrollOnChannel))
    print("Normalized Conv: %s unrollOnChannel:%d\n" % (str(self.normalizedCC), self.unrollOnChannel))
    print("Tensile Index Assignments and Usage:")
    print("   Tensile    : ConvChar: Explanation/Usage")
    for (idx,regDim) in enumerate(self.indexAssignments):
        tensileChar = globalParameters['IndexChars'][idx]
        usage = str(regDim.dim)
        usage = usage.replace('#T', tensileChar)
        for i in range(len(self.cc.stride)):
            usage = usage.replace('#S%d'%i, str(self.cc.stride[i]) if self.cc.stride[i]>=0 else 'TBD')
        for i in range(len(self.cc.dilation)):
            usage = usage.replace('#D%d'%i, str(self.cc.dilation[i]) if self.cc.dilation[i]>=0 else 'TBD')
        print("  %d('%c') %-5s:   %s" % (idx, tensileChar, str(regDim.fbs).split('.')[1], usage))

    if 0:
      print ()
      print ("  FreeIndices:", ','.join([str(x) for x in self.freeIndices]))
      print ("  BatchIndices:", ','.join([str(x) for x in self.batchIndices]))
      print ("  SumIndices:", ','.join([str(x) for x in self.sumIndices]))

    if details:
      print ()
      print ("- Spatial sizes D_i, H_i, W_i refer to size of INPUT dimension.")
      print ("- Spatial sizes D_o, H_o, W_o refer to size of OUTPUT dimension.")
      print ("     For example W_o =  (W_i - X - padStart - padEnd + 1)/stride")
      print ("- (TBD)' indicates the parm is flexible and must be specified at runtime.")
      print ("- (i)' where i is an integer constant, indicates the parm is hard-coded at compile time.")
      print ("  The runtime value must match the compile-time value.")
      print ("- Unspecified strides use default stride value:")
      print ("    stride[i] = (stride[i-1]*size[i]) for i>0 ; 1 for i==0.")
      print ("- [stride*,size*] in brackets list required values to run the generated solutions.")
      print ("- Tensile IndexAssignments list the fastest-moving (in memory) index first.")
      print ("- Dimension collapsing:")
      print ("    - spatial dims with default strides and no zero-pad are collapsed with adjacent dims.")
      print ("    - Nx1 filter with dilationY=1 collapse into a single filter.")
      print ("    - 1xN filter with dilationX=1 collapse into a single filter.")
      print ("    - PackSpatialDims=0 / PackFilterDims=0 forcibly disables collapsing.")
      print ("- Overlapping / Hidden summation dimensions shown below with leading '_'.")

    print ()
    if problemType:
      print ("ProblemType Definition:")
      for k in Convolution.SummaryProblemProperties:
        try:
          if k in ['IndexAssignmentsA', 'IndexAssignmentsB']:
              comment = "# [" + ",".join([self.markedConvolutionChar(idx,k[-1]) for idx in problemType[k]]) + "]"
          elif k == 'NumIndicesC':
              comment = "# [" + ",".join([self.convolutionChar(idx) for idx in range(0,problemType[k])] ) + "]"
          else:
              comment = ""
          print ("  ", k, ":", problemType[k], comment)
        except KeyError:
          pass

    print ()
    print ("Solution Assertions:")
    for k in Convolution.SummarySolutionProperties:
      try:
        print ("  ", k, ":", self.solutionParms[k])
      except KeyError:
        pass


  def checkDims(self, freeIndices, batchIndices, sumIndices):
    for dimList in (self.regDimsA, self.regDimsB):
      for (idx,fbs,dim) in dimList:
        if fbs==Fbs.Free and idx not in freeIndices:
          raise RuntimeError ("dimension %d('%s') expected to be free dimension" % (idx, dim.shortChar))
        elif fbs==Fbs.Batch and idx not in batchIndices:
          raise RuntimeError ("dimension %d('%s') expected to be batch dimension" % (idx, dim.shortChar))
        elif fbs==Fbs.Sum and idx not in sumIndices:
          raise RuntimeError ("dimension %d('%s') expected to be summation dimension" % (idx, dim.shortChar))


  def identifier(self):
    id = self.convolutionType
    id += "_" + self.tensorAFormat
    id += "_" + self.tensorBFormat
    id += "_" + self.tensorDFormat
    id += "_spatialDims:" + str(self.numSpatialDims)
    id += "_indices:" + '.'.join([x.dim.shortChar for x in self.indexAssignments])
    if self.cc.spatial:
      id += "_spatial:" + "x".join([str(x) for x in self.cc.spatial[::-1]])
    id += "_filter:" + "x".join([str(x) for x in self.cc.fil[::-1]])
    id += "_stride:" + "x".join([str(x) for x in self.cc.stride[::-1]])
    id += "_dilation:" + "x".join([str(x) for x in self.cc.dilation[::-1]])
    id += "_padStart:" + "x".join([str(x) for x in self.cc.padStart[::-1]])
    id += "_padEnd:" + "x".join([str(x) for x in self.cc.padEnd[::-1]])
    return id


################################################################################
# ProblemType
# name of solution should begin with name of problemType, and arguments can be listed out explicitly
class ProblemType:
  ########################################
  def __init__(self, config):
    self.state = {}

    for key in defaultProblemType:
      assignParameterWithDefault(self.state, key, config, defaultProblemType)

    if "DataType" in config:
      self["DataType"] = DataType(config["DataType"])
    else:
      printExit("NO data type specified")
      self["DataType"] = DataType(0)

    if "DestDataType" in config:
      self["DestDataType"] = DataType(config["DestDataType"])
    else:
      if "DataType" in config:
        self["DestDataType"] = DataType(config["DataType"])
      else:
        printExit("NO dest data type or data type specified")
        self["DataType"] = DataType(0)


    if "ComputeDataType" in config:
      self["ComputeDataType"] = DataType(config["ComputeDataType"])
    else:
      if "DestDataType" in config:
        self["ComputeDataType"] = DataType(config["DestDataType"])
      else:
        if "DataType" in config:
          self["ComputeDataType"] = DataType(config["DataType"])
        else:
          printExit("NO compute data type, or dest data type, or data type specified")
          self["DataType"] = DataType(0)

    self.convolution = None
    if self["OperationType"] == "GEMM":
      self.initGEMM(config)
    elif self["OperationType"] == "TensorContraction":
      self.initTensorContraction(self.state)
    elif self["OperationType"] in ("ConvolutionForward", "ConvolutionBackwardData", "ConvolutionBackwardWeights"):
      self.initConvolution(config, self["OperationType"])
    else:
      printExit("Unsupported OperationType = %s" % self["OperationType"])

    self.state["AssignedDerivedParameters"] = False
    ProblemType.assignDerivedParameters(self.state)

    if self.convolution:
      if globalParameters["PrintConvolutionUsage"] & 0x3 :
        print()
        self.convolution.printUsage(self, globalParameters["PrintConvolutionUsage"]&0x2)
        print()
      self.convolution.checkDims(self.state["IndicesFree"], self.state["IndicesBatch"], self.state["IndicesSummation"])


    for tc in ('A', 'B'):
      freeDims={}
      sumDims={}
      for zp in self["ZeroPad%s"%tc] :
        (freeDim, sumDim, leading, trailing) = zp
        if freeDim not in self.state["IndicesFree"]:
          printExit("ZeroPad%s=%s dim=%u is not a free index"%(tc, zp, freeDim))
        if freeDim not in self.state["IndexAssignments%s"%tc]:
          printExit("ZeroPad%s=%s dim=%u is not in IndexAssignments%s"%(tc, zp, freeDim, tc))
        if sumDim not in self.state["IndicesSummation"]:
          printExit("ZeroPad%s=%s dim=%u is not a summation index"%(tc, zp, sumDim))
        if freeDim in freeDims:
          printExit("ZeroPad%s=%s freeDim=%u occurs in more than one tuple (prev:%s)"%(tc, zp, freeDim,freeDims[freeDim]))
        freeDims[freeDim] = zp
        if sumDim in sumDims:
          printExit("ZeroPad%s=%s sumDim=%u occurs in more than one tuple"%(tc, zp, sumDim))
        sumDims[sumDim] = zp

    for tc in ('A', 'B'):
      for sc in self["SetConstStride%s"%tc] :
          (anchorDim, stride) = sc[:2]
          if anchorDim not in self.state["IndexAssignments%s"%tc]:
              printExit("SetConstStride%s=%s anchorDim=%u is not in IndexAssignments%s"%(tc, sc, anchorDim, tc))



  ########################################
  def initGEMM(self, config):
    sumIdx = 3 if self["Batched"] else 2
    self["IndexAssignmentsA"] = [0, sumIdx] # N
    self["IndexAssignmentsB"] = [sumIdx, 1] # N
    if self["TransposeA"]:
      self["IndexAssignmentsA"] = [sumIdx, 0] # T
    if self["TransposeB"]:
      self["IndexAssignmentsB"] = [1, sumIdx] # T
    if self["Batched"]:
      self["IndexAssignmentsA"].append(2)
      self["IndexAssignmentsB"].append(2)
      self["NumIndicesC"] = 3
    else:
      self["NumIndicesC"] = 2

    self["NumIndicesLD"] = 4
    self["IndexAssignmentsLD"][0] = self["NumIndicesC"] + 1
    for i in range(1, len(self["IndexAssignmentsLD"])):
      self["IndexAssignmentsLD"][i] = self["IndexAssignmentsLD"][i-1] + 1

  ########################################
  def initTensorContraction(self, config):
    assignParameterRequired(self.state, "NumIndicesC", config)
    assignParameterRequired(self.state, "IndexAssignmentsA", config)
    assignParameterRequired(self.state, "IndexAssignmentsB", config)
    self["NumIndicesLD"] = 0

  ########################################
  def initConvolution(self, config, convolutionType):
    convolutionConfig = {}
    try:
      if config['ConvolutionConfig'] != None:
        for dict in config['ConvolutionConfig']:
          for k,v in dict.items():
            convolutionConfig[k] = v
    except KeyError:
      raise RuntimeError ("OperationType %s must include ConvolutioConfig section in ProblemType"%convolutionType)

    self.convolution = Convolution(self, convolutionType, convolutionConfig)
    self["NumIndicesLD"] = 0
    self["UseBeta"] = False

  ########################################
  def isGEMM(self):
    return self.operationType == 0

  ########################################
  def isTensorContraction(self):
    return self.operationType == 1

  ########################################
  # determine d0, d1, dU
  @staticmethod
  def assignDerivedParameters(state):
    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    state["TotalIndices"] = max(max(state["IndexAssignmentsA"])+1, \
        max(state["IndexAssignmentsB"])+1)

    # determine num free, batch
    state["IndicesFree"] = []
    state["IndicesBatch"] = []
    state["IndicesSummation"] = []

    for i in range(0, state["NumIndicesC"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        state["IndicesBatch"].append(i)

      elif inA or inB:
        state["IndicesFree"].append(i)
      else:
        printExit("invalid index %u (inC but not (inA or inB))" % i)

    # determine num summation
    for i in range(state["NumIndicesC"], state["TotalIndices"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        state["IndicesSummation"].append(i)
      else:
        printExit("invalid index %u (expected summation but not (inA and inB))" % i)
    # print index assignments
    if globalParameters["PrintIndexAssignments"]:
      print("IndicesFree:  %s" % state["IndicesFree"])
      print("IndicesBatch: %s" % state["IndicesBatch"])
      print("IndicesSum:   %s" % state["IndicesSummation"])
      print("IndexAssignmentsA:   %s" % state["IndexAssignmentsA"])
      print("IndexAssignmentsB:   %s" % state["IndexAssignmentsB"])
      print("NumIndicesC:  %s" % state["NumIndicesC"])

    for k in ('IndexAssignmentsA','IndexAssignmentsB'):
      if len(state[k]) != len(set(state[k])):
        printExit("duplicate index in %s=%s"% (k,state[k]))

    state["NumIndicesFree"] = len(state["IndicesFree"])
    state["NumIndicesBatch"] = len(state["IndicesBatch"])
    state["NumIndicesSummation"] = len(state["IndicesSummation"])
    if not state["AllowNoFreeDims"] and state["NumIndicesFree"] < 2 :
      printExit("Tensile requires >= 2 free indices or set AllowNoFreeDims; FreeIndices=%s."% state["IndicesFree"])

    # by default, unroll index will be the last/inner summation index
    state["IndexUnroll"] = state["IndicesSummation"][len(state["IndicesSummation"])-1]
    for i in range(0, len(state["IndexAssignmentsA"])):
      if state["IndexAssignmentsA"][i] == state["IndexUnroll"]:
        state["IndexUnrollA"] = i
        break
    for i in range(0, len(state["IndexAssignmentsB"])):
      if state["IndexAssignmentsB"][i] == state["IndexUnroll"]:
        state["IndexUnrollB"] = i
        break
    #print2("IndexUnrollA: %u" % state["IndexUnrollA"])
    #print2("IndexUnrollB: %u" % state["IndexUnrollB"])

    # assign d0, d1
    if state["AllowNoFreeDims"]:
      dimList = state["IndicesFree"] + state["IndicesBatch"]
    else:
      dimList = state["IndicesFree"]
    state["Index01A"] = [i for i in state["IndexAssignmentsA"] if i in dimList][0]
    state["Index01B"] = [i for i in state["IndexAssignmentsB"] if i in dimList][0]
    #print2("Index01A: %u" % state["Index01A"])
    #print2("Index01B: %u" % state["Index01B"])
    # Store code is optimized for 0 as the fastest-moving in memory
    # whichever has lower stride in C (lower value), is 0, other is 1
    if state["Index01A"] < state["Index01B"]:
      state["Index0"]  = state["Index01A"]
      state["Index1"]  = state["Index01B"]
      state["Tensor0"] = 0
      state["Tensor1"] = 1
      state["TileA"] = 0
      state["TileB"] = 1
    else:
      state["Index0"]  = state["Index01B"]
      state["Index1"]  = state["Index01A"]
      state["Tensor0"] = 1
      state["Tensor1"] = 0
      state["TileA"] = 1
      state["TileB"] = 0

    # generalize transpose
    strideIdxA = state["IndexAssignmentsA"].index(state["Index01A"])
    strideIdxB = state["IndexAssignmentsB"].index(state["Index01B"])
    unrollIdxA = state["IndexAssignmentsA"].index(state["IndexUnroll"])
    unrollIdxB = state["IndexAssignmentsB"].index(state["IndexUnroll"])
    state["TLUA"] = strideIdxA < unrollIdxA
    state["TLUB"] = strideIdxB < unrollIdxB
    #state["TLUB"] = True # hack

    if globalParameters["PrintIndexAssignments"]:
      print("TLUA:  %s (stridePosA(%d) <? unrollIdxA(%d)" % \
			(state["TLUA"], strideIdxA, unrollIdxA))
      print("TLUB:  %s (stridePosB(%d) <? unrollIdxB(%d)" % \
	  		(state["TLUB"], strideIdxB, unrollIdxB))
      print("Index01A:  %s" % state["Index01A"])
      print("Index01B:  %s" % state["Index01B"])
    #unrollDimStrideGreaterThanTileDimStrideA = TLUA = !transA = fast
    #!unrollDimStrideLessThanTileDimStrideB   = TLUB =  transB = fast
    state["AssignedDerivedParameters"] = True


  ########################################
  def __str__(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self["NumIndicesC"]):
      name += indexChars[i].lower()
    # A dimensions
    name += "_A"
    for i in self["IndexAssignmentsA"]:
      name += indexChars[i].lower()
    if self["ComplexConjugateA"]:
      name += "C"
    # B dimensions
    name += "_B"
    for i in self["IndexAssignmentsB"]:
      name += indexChars[i].lower()
    if self["ComplexConjugateB"]:
      name += "C"

    # precision and other
    name += "_"
    name += self["DataType"].toChar()
    if self["UseBeta"]: name += "B"
    if self["HighPrecisionAccumulate"] and not self["SilentHighPrecisionAccumulate"]: name += "H"
    if self["UseInitialStridesAB"]: name += "I"
    if self["UseInitialStridesCD"]: name += "Ic"
    return name

  def keys(self):
    return list(self.state.keys())
  def __len__(self):
    return len(self.state)
  def __iter__(self):
    return iter(self.state)
  def __getitem__(self, key):
    return self.state[key]
  def __setitem__(self, key, value):
    self.state[key] = value
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return self.state
  def __hash__(self):
    return hash(str(self))
  def __eq__(self, other):
    return isinstance(other, ProblemType) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

  def get(self, key, default=None):
    try:
      return self.state[key]
    except:
      return default



################################################################################
# ProblemSizeRange
################################################################################
class ProblemSizeRange:

  ########################################
  def __init__(self, problemType, config):
    self.totalIndices = 1+max(problemType["IndexAssignmentsA"]) + problemType["NumIndicesLD"]
    if len(config) < self.totalIndices:
      for i in range(len(config), self.totalIndices):
        if i < self.totalIndices - problemType["NumIndicesLD"]:
          config.append(0)
        else:
          config.append([0])

    self.indexMax = []
    self.indexIsSized = []
    self.indicesSized = []
    self.indicesMapped = []
    for i in range(0, self.totalIndices):
      dim = deepcopy(config[i])
      if isinstance(dim, list):
        if len(dim) == 1:
          self.indicesSized.append([dim[0], 1, 0, dim[0]])
        elif len(dim) == 2:
          self.indicesSized.append([dim[0], dim[0], 0, dim[1]])
        elif len(dim) == 3:
          self.indicesSized.append([dim[0], dim[1], 0, dim[2]])
        elif len(dim) == 4:
          self.indicesSized.append([dim[0], dim[1], dim[2], dim[3]])
        else:
          printExit("dimension[%u] config (%s) has %u descriptors rather than 1-4."
              % ( i, dim, len(dim) ))
        self.indexIsSized.append(True)
        self.indexMax.append(self.indicesSized[len(self.indicesSized)-1][3])

      elif isinstance(dim, int):
        self.indicesMapped.append(dim)
        self.indexIsSized.append(False)
        self.indexMax.append(self.indicesSized[self.indicesMapped[ \
            len(self.indicesMapped)-1]][3])

    # max num elements in each tensor
    self.maxNumElements = [ 1, 1, 1 ]
    for i in range(0, problemType["NumIndicesC"]):
      self.maxNumElements[0] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsA"]:
      self.maxNumElements[1] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsB"]:
      self.maxNumElements[2] *= self.indexMax[i]

    self.totalProblemSizes = 1
    self.numProblemSizes = [] # per index
    self.problemSizeToIndex = []
    self.problemIndexToSize = []
    sizedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      self.problemSizeToIndex.append({})
      self.problemIndexToSize.append({})
      if self.indexIsSized[i]:
        self.numProblemSizes.append(0)
        index = self.indicesSized[sizedIdx]
        sizedIdx += 1
        currentSize = index[0]
        currentIncrement = index[1]
        while currentSize <= index[3]:
          currentSize += currentIncrement
          currentIncrement += index[2]
          self.numProblemSizes[i] += 1
      else:
        self.numProblemSizes.append(1)
      self.totalProblemSizes *= self.numProblemSizes[i]

    ########################################
    # enumerate problem sizes
    currentSizedIndexSizes = []
    currentSizedIndexIncrements = []
    for i in range(0, len(self.indicesSized)):
      currentSizedIndexSizes.append(self.indicesSized[i][0])
      currentSizedIndexIncrements.append(self.indicesSized[i][1])

    # iterate over all problem sizes
    self.problemSizes = []
    moreProblemSizes = True
    problemIdx = 0
    problemSize = [0]*self.totalIndices
    while moreProblemSizes:
      #/ convert current sized and mapped indices to full sizes
      currentSizedIdx = 0
      currentMappedIdx = 0
      for i in range(0, self.totalIndices):
        if self.indexIsSized[i]:
          problemSize[i] = currentSizedIndexSizes[currentSizedIdx]
          currentSizedIdx+=1
        else:
          problemSize[i] = problemSize[self.indicesMapped[currentMappedIdx]]
          currentMappedIdx+=1
      self.problemSizes.append(tuple(problemSize))

      #/ increment sizes for next benchmark
      currentSizedIndexSizes[0] += currentSizedIndexIncrements[0]
      currentSizedIndexIncrements[0] += self.indicesSized[0][2]
      for i in range(1, len(self.indicesSized)+1):
        # if prior index past max, reset to min and increment next index
        if currentSizedIndexSizes[i-1] > self.indicesSized[i-1][3]:
          #/ reset prior index
          currentSizedIndexSizes[i-1] = self.indicesSized[i-1][0]
          currentSizedIndexIncrements[i-1] = self.indicesSized[i-1][1]
          # increment next index
          if i >= len(self.indicesSized):
            moreProblemSizes = False
          else:
            currentSizedIndexSizes[i] += currentSizedIndexIncrements[i]
            currentSizedIndexIncrements[i] += self.indicesSized[i][2]

      problemIdx+=1

  ########################################
  # YAML format
  def __str__(self):
    state = "[ "
    sizedIdx = 0
    mappedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      if self.indexIsSized[i]:
        indices = self.indicesSized[sizedIdx]
        state += "[ %u, %u, %u, %u ]" \
            % (indices[0], indices[1], indices[2], indices[3])
        sizedIdx += 1
      else:
        indices = self.indicesSized[self.indicesMapped[mappedIdx]]
        state += str(self.indicesMapped[mappedIdx])
        mappedIdx += 1
      if i < len(self.indexIsSized)-1:
        state += ", "
    state += " ]"
    return state

class Problem:
  """ Problem sizes, strides, padding and other info"""
  def __init__(self, sizes=None, stridesA=None, stridesB=None, stridesC=None, stridesD=None, zeroPadA=None, zeroPadB=None, count=None):
    self.sizes = tuple(sizes) if sizes else None
    self.stridesA = tuple(stridesA) if stridesA else None
    self.stridesB = tuple(stridesB) if stridesB else None
    self.stridesC = tuple(stridesC) if stridesC else None
    self.stridesD = tuple(stridesD) if stridesD else None

    self.zeroPadA = zeroPadA
    self.zeroPadB = zeroPadB
    self.convConfig = None
    self.count = count

  def __str__(self):
    rv= "sizes:" + str(self.sizes)
    if self.stridesA:
      rv += "stridesA:" + str(self.stridesA)
    return rv


class ConvProblem(Problem):
  ConvField = namedtuple ("ConvField", ('shortChar', 'descrip', 'default'))
  AllowedConvFields = [ ConvField('n', 'Batch Count', None),
                        ConvField('c', 'Channel In', None),
                        ConvField('k', 'Channel Out',  None),

                        ConvField('d', 'Spatial Depth', -1),
                        ConvField('h', 'Spatial Height',-1),
                        ConvField('w', 'Spatial Width', -1),

                        ConvField('z', 'Filter Z',  -1),
                        ConvField('y', 'Filter Y',  -1),
                        ConvField('x', 'Filter X',  -1),

                        ConvField('#', 'Stride for Depth', -1),
                        ConvField('u', 'Stride for Height', -1),
                        ConvField('v', 'Stride for Width', -1),

                        ConvField('^', 'Dilation for filter Depth Z', -1),
                        ConvField('l', 'Dilation for filter Height Y', -1),
                        ConvField('j', 'Dilation for filter Width X', -1),

                        ConvField('$', 'Pad for Depth', -1),
                        ConvField('p', 'Pad for Height', -1),
                        ConvField('q', 'Pad for WidthX', -1),

                        ConvField('$_', 'Pad End for Depth (overrides $ for end)', -1),
                        ConvField('p_', 'Pad End for Height (overrides p for end)', -1),
                        ConvField('q_', 'Pad End for Width (overrides q for end)', -1),

                        ConvField('g', 'Group Count',  1),

                        ConvField('count', 'Layer execution Count',  -1),
                        ]
  AllowedConfFieldsDict = {field.shortChar : field for field in AllowedConvFields}

  @staticmethod
  def initParm(e, chars, skipFields):
    fields = []
    for s in (chars):
      if s not in skipFields:
        fields.append(e[s])
    return fields

  def __init__(self, e, convolution):

    self.inputConfig = deepcopy(e)

    if convolution.formatNumSpatialDims==2:
      skipFields = ('d', 'z', '#', '^', '$')
    else:
      skipFields = ()

    if not isinstance(e,dict):
        raise RuntimeError ("ConvProblem must be a dictionary, for example '{n: 64, ...}' not '[n: 64, ...]'")

    for k in e:
      if k not in ConvProblem.AllowedConfFieldsDict:
        # TODO  - detect and print message for common error n:32 w/o space
        raise RuntimeError ("unknown ConvProblem field '%s'"%k)

    for (k,field) in ConvProblem.AllowedConfFieldsDict.items():
      if k not in e and k not in skipFields:
        if field.default == None:
          raise RuntimeError ("required ConvProblem field '%s' not present in ConvProblem:%s"%(k,e))
        elif isinstance(field.default, int):
          e[k] = field.default

    padStart = self.initParm(e, ('q','p','$'), skipFields)
    padEnd = self.initParm(e, ('q_','p_','$_'), skipFields)
    padEnd = [ps if pe==-1 else pe for (ps,pe) in zip(padStart,padEnd) ] # use padStart as default
    self.convConfig = ConvolutionConfig(
                fil = self.initParm(e, ('x','y','z'), skipFields),
                stride = self.initParm(e, ('v','u','#'), skipFields),
                dilation   = self.initParm(e, ('j','l','^'), skipFields),
                spatial =    self.initParm(e, ('w','h','d'), skipFields),
                padStart = padStart,
                padEnd = padEnd,
                groupCount = e['g']
              )

    (sizes, stridesA, stridesB) = convolution.makeProblem(e['n'], e['c'], e['k'], self.convConfig)
    zeroPadA = convolution.makeZeroPadProblemType(convolution.problemTypeOut["ZeroPadA"],
        self.convConfig.padStart, self.convConfig.padEnd, self.convConfig)

    Problem.__init__(self, sizes, stridesA, stridesB=stridesB, zeroPadA=zeroPadA, count=e['count'])

    #print ("sizes=", self.sizes, "stridesA=", self.stridesA, "stridesB=", self.stridesB, "zeroPadA=", self.zeroPadA)


  def toExactDict(self):
    """ Return a dict with ExactDict fields, after converting the ConvProblem to tensor sizes and strides"""
    padStartA = [zp[2] for zp in self.zeroPadA]
    padEndA = [zp[3] for zp in self.zeroPadA]
    exactFields = OrderedDict()

    exactFields['count'] = self.count
    exactFields['sizes'] = list(self.sizes)
    exactFields['stridesA'] = list(self.stridesA)

    if padStartA:
      exactFields['padStartA'] = padStartA
    if padEndA:
      exactFields['padEndA'] = padEndA

    return exactFields


class ExactList(Problem):
  def __init__(self, e, problemType):
    if len(e) == problemType["TotalIndices"]:
      if -1 in e:
        printExit("ExactSize %s contains -1" % (e))
      if problemType["OperationType"] == "GEMM":
        e += [-1, -1, -1, -1]
        e = ExactList.convertLeadingDims(problemType, tuple(e))
      sizes=e

    elif len(e) == (problemType["TotalIndices"] + problemType["NumIndicesLD"]):
      sizes = ExactList.convertLeadingDims(problemType, tuple(e))
    else:
      printExit("ExactSize %s doesn't match indices of ProblemType %s, totalIndices=%d" \
          % (e, problemType, problemType["TotalIndices"]) )

    # TODO- pass strides here, remove calls to convertLeadingDims
    Problem.__init__(self, sizes=sizes, zeroPadA=problemType["ZeroPadA"], zeroPadB=problemType["ZeroPadB"])

  @staticmethod
  def convertLeadingDims(problemType, problemSize):
    # FIXME-problem: refactor to eliminate max, pass strides in strideB parm rather than hacked
    # onto the end of the sizes list
    return problemSize[:problemType["NumIndicesC"]+1] + \
           (max(problemSize[0], problemSize[problemType["IndexAssignmentsLD"][0]]),) + \
           (max(problemSize[0], problemSize[problemType["IndexAssignmentsLD"][1]]),) + \
           (max(problemSize[problemType["IndexAssignmentsLD"][2]],
                problemSize[problemType["IndexAssignmentsA"][0]]),) + \
           (max(problemSize[problemType["IndexAssignmentsLD"][3]],
                problemSize[problemType["IndexAssignmentsB"][0]]),)


class ExactDict(Problem):
  # padStartA is list of pad starts for A dimension in order of ZeroPadA list.
  # padEndA is list of pad ends for A dimension in order of ZeroPadA list.
  AllowedFields = [ 'count', 'sizes', 'stridesA', 'stridesB', 'stridesC', 'stridesD', 'padStartA', 'padEndA', 'padStartB', 'padEndB']

  def __init__(self, e, problemType):
    Problem.__init__(self)

    for f in e:
      if f in ExactDict.AllowedFields:
        setattr(self, f, e[f])
      else:
        raise RuntimeError ("specified field '%s' is not a valid Exact dict field"%f)


    if problemType:
      zp={}
      zp['A'] = deepcopy(problemType["ZeroPadA"])
      zp['B'] = deepcopy(problemType["ZeroPadB"])

      for (tc, padName, zpField) in (
          ("A", "padStartA",2), ("A", "padEndA", 3),
          ("B", "padStartB",2), ("B", "padEndB", 3) ):
          try:
            problemPad = getattr(self, padName)
            if len(problemPad) != len (zp[tc]):
                raise RuntimeError ("problem-specified %s==%s does not match length of problem-type pad==%s." % (padName, problemPad, zp[tc]))
            for (i,p) in enumerate(problemPad):
              if not (zp[tc][i][zpField] == -1 or zp[tc][i][zpField] == p):
                raise RuntimeError ("problem-specified %s==%d does not match problem-type==%d." % (padName, p, zp[tc][i][zpField]))
              zp[tc][i][zpField] = p
          except AttributeError:
            None

      for (tc) in ("A", "B"):
        for p in zp[tc]:
          if p[2] == -1 or p[3]==-1:
            raise RuntimeError ("padStart/padEnd for %s must be specified in problem-type or problem - can't be left -1/TBD" % zp[tc])

      self.zeroPadA = zp['A']
      self.zeroPadB = zp['B']
    else:
      self.zeroPadA = self.zeroPadB = []

    if problemType:
      if len(self.sizes) != problemType["TotalIndices"]:
        raise RuntimeError ("specified size=%s does not have enough indices for problem (expected %d, got %d)" \
                % (self.sizes, problemType["TotalIndices"], len(self.sizes)))




################################################################################
# ProblemSizes
################################################################################
class ProblemSizes:

  ########################################
  def __init__(self, problemType, config):
    self.problemType = problemType
    self.ranges = []
    self.exacts = []
    self.minStrides = None
    if config:
      for dictionary in config:
        for sizeTypeKey in dictionary:
          #print ("PROBLEM parsed:", sizeTypeKey, dictionary[sizeTypeKey])
          if sizeTypeKey == "Range":
            psr = ProblemSizeRange(problemType, dictionary[sizeTypeKey])
            self.ranges.append( psr )
          elif sizeTypeKey == "Exact":
            e= dictionary[sizeTypeKey]
            if isinstance(e,list):
              self.exacts.append(ExactList(e, problemType))
            elif isinstance(e,dict):
              self.exacts.append(ExactDict(e, problemType))
            else:
              printExit("Unsupported Exact type==%s"%type(e))

          elif sizeTypeKey == "Conv":
            if problemType.convolution == None:
              printExit("ConvProblem requires OperationType==Convolution*")
            else:
              self.exacts.append(ConvProblem(dictionary[sizeTypeKey], problemType.convolution))

          elif sizeTypeKey == "MinStride":
            e = dictionary[sizeTypeKey]
            if len(e) != problemType["TotalIndices"]:
              printExit("MinStride %s doesn't match indices of ProblemType %s" \
                  % (e, problemType) )
            if self.minStrides:
              printExit("Only one MinStride command is allowed in a ProblemsSizes definition.  Previous minStrides:%s, New minstride:%s" \
                  % (self.minStrides, e) )

            self.minStrides=(tuple(e))
          else:
            printExit("ProblemSize Type %s not supported"%sizeTypeKey)

    if not self.minStrides:
      # set harmless default mins of 0
      self.minStrides = ([0]* problemType["TotalIndices"])

    # not the ideal spot, but convert leading dims that are below the minimum size
    if problemType["OperationType"] == "GEMM":
      for i in range(0, len(self.ranges)):
        self.ranges[i].problemSizes[:] = \
          [ExactList.convertLeadingDims(self.problemType, problemSize) for problemSize in self.ranges[i].problemSizes]

    self.problems = OrderedDict()
    for sizeRange in self.ranges:
        for rangeSize in sizeRange.problemSizes:
            self.problems.update({Problem(rangeSize, zeroPadA=problemType["ZeroPadA"]) : 1 })
    for e in self.exacts:
        self.problems.update({e : 1})
    if globalParameters["SortProblems"]:
      self.problems =  sorted(list( self.problems.keys()), key=operator.attrgetter("sizes"))
    else:
      self.problems =  list(self.problems.keys())
    self.totalProblemSizes = len(self.problems)

    # max sizes
    self.maxD = 0
    self.maxC = 0
    self.maxA = 0
    self.maxB = 0
    for problem in self.problems:
      problemSize = problem.sizes # FIXME-problem.   This should use problem.strides*

      sizeLdd = problemSize[self.problemType["IndexAssignmentsLD"][0]] if problemType["OperationType"] == "GEMM" else problemSize[0]
      sizeD = max(self.minStrides[0], sizeLdd)
      for i in range(1, problemType["NumIndicesC"]):
        sizeD *= max(self.minStrides[i], problemSize[i])

      sizeLdc = problemSize[self.problemType["IndexAssignmentsLD"][1]] if problemType["OperationType"] == "GEMM" else problemSize[0]
      sizeC = max(self.minStrides[0], sizeLdc)
      for i in range(1, problemType["NumIndicesC"]):
        sizeC *= max(self.minStrides[i], problemSize[i])

      sizeLda = problemSize[self.problemType["IndexAssignmentsLD"][2]] \
                if problemType["OperationType"] == "GEMM" \
                else problemSize[self.problemType["IndexAssignmentsA"][0]]
      sizeA = max(self.minStrides[self.problemType["IndexAssignmentsA"][0]], sizeLda)
      for i in self.problemType["IndexAssignmentsA"][1:]:
        sizeA *= max(self.minStrides[i], problemSize[i])

      sizeLdb = problemSize[self.problemType["IndexAssignmentsLD"][3]] \
                if problemType["OperationType"] == "GEMM" \
                else problemSize[self.problemType["IndexAssignmentsB"][0]]
      sizeB = max(self.minStrides[self.problemType["IndexAssignmentsB"][0]], sizeLdb)
      for i in self.problemType["IndexAssignmentsB"][1:]:
        sizeB *= max(self.minStrides[i], problemSize[i])

      self.maxD = max(self.maxD, sizeD)
      self.maxC = max(self.maxC, sizeC)
      self.maxA = max(self.maxA, sizeA)
      self.maxB = max(self.maxB, sizeB)

    if globalParameters["PrintConvolutionUsage"] & 0x4:
      for problem in self.problems:
        if isinstance(problem, ConvProblem):
          print (problem.inputConfig, '->\n  ', ", ".join(["%s: %s"%(k,v) for (k,v) in problem.toExactDict().items()]))


  def __str__(self):
    s = "ProblemSizes\n"
    for sizeRange in self.ranges:
      s += "  %s" % sizeRange
    return s

# kds is class Solution or class Kernel
# If PackFreeDims=1 then all free dims are packed ; else only 1 free dim/matrix is supported
# PackBatchDims can pack batches into A or B (has stride==0 requirements for non-packed tensor);
# batchMask controls which bit in PackBatchDims detects batch index
def isPackedIndex(ks, index, batchMask=0x3):
  problemType = ks["ProblemType"]
  return index in problemType["IndicesFree"] and ks["PackFreeDims"] or \
         index in problemType["IndicesBatch"] and (ks["PackBatchDims"] & batchMask)

def isExtractableIndex(ks, index, tc='x'):
  xA = index in ks['PackedC0IndicesX'][:-1]
  xB = index in ks['PackedC1IndicesX'][:-1]
  if tc=='A':
    return xA
  elif tc=='B':
    return xB
  else:
    return xA or xB

################################################################################
# Solution
################################################################################
class Solution:

  ########################################
  def __init__(self, config):
    self._name = None
    config = deepcopy(config)

    self._state = {}
    # problem type
    if "ProblemType" in config:
      self["ProblemType"] = ProblemType(config["ProblemType"])
    else:
      self["ProblemType"] = ProblemType(defaultProblemType)

    # assign parameters with defaults
    for key in defaultSolution:
      assignParameterWithDefault(self._state, key, config, defaultSolution)

    if 'ISA' not in self._state:
      if 'ISA' in config:
        self._state['ISA'] = config['ISA']
      elif config['KernelLanguage'] == 'Assembly':
        self._state['ISA'] = list(globalParameters["CurrentISA"])
      else:
        self._state['ISA'] = [0,0,0]

    # assign parameters without defaults
    for key in config:
      if key != "ProblemType" and key not in self._state:
        self._state[key] = config[key]
    self["Valid"] = True
    self["AssignedProblemIndependentDerivedParameters"] = False
    self["AssignedDerivedParameters"] = False

    if self["ProblemType"].convolution:
        for (key,value) in self["ProblemType"].convolution.solutionParms.items():
            self._state[key]=value
    Solution.assignDerivedParameters(self._state)
    self._name = None

  # these keys are copied from ProblemType to internal that may be overridden
  InternalKeys = ["UseSgprForGRO","VectorStore"]

  ########################################
  # get a list of kernel parameters for this solution
  def getKernels(self):
    kernel = deepcopy(self._state)
    kernel.update({"Kernel": True})
    kernels = []
    kernels.append(kernel)
    return kernels

  @staticmethod
  def getKernelsBetaOnlyFromProblem(problemType, gsu):
    kernels = []
    if gsu < 2:
      return kernels
    betas = [False]
    if problemType["UseBeta"]:
      betas.append(True)
    for beta in betas:
      kernel = {}
      kernel["ProblemType"] = deepcopy(problemType)
      kernel["ProblemType"]["UseBeta"] = beta
      kernel["KernelLanguage"] = "Source"
      kernels.append(kernel)
    return kernels

  ########################################
  # get a list of kernel parameters for this solution
  def getKernelsBetaOnly(self):
    return self.getKernelsBetaOnlyFromProblem( \
            self["ProblemType"], \
            self["GlobalSplitU"])

  ########################################
  # assign tile sizes
  @staticmethod
  def assignProblemIndependentDerivedParameters(state):

    if "AssignedProblemIndependentDerivedParameters" in state:
      if state["AssignedProblemIndependentDerivedParameters"]:
        return
    state["AssignedProblemIndependentDerivedParameters"] = False
    if "Valid" not in state:
      state["Valid"] = True

    EnableMatrixInstruction = state["EnableMatrixInstruction"] if "EnableMatrixInstruction" in state else None
    if EnableMatrixInstruction == None:
      if  ("MIBlock" in state and len(state["MIBlock"]) == 6) \
          and ("MIWaveGroup" in state and len(state["MIWaveGroup"]) == 2) \
          and ("MIWaveTile" in state and len(state["MIWaveTile"]) == 2):
        EnableMatrixInstruction = True
      elif ("WorkGroup" in state and len(state["WorkGroup"]) == 3) \
          and ("ThreadTile" in state and len(state["ThreadTile"]) == 2) :
        EnableMatrixInstruction = False
      else:
        reject(state, "EnableMatrixInstruction undetermined")

    if EnableMatrixInstruction == True:
      state["MatrixInstM"]         = state["MIBlock"][0]
      state["MatrixInstN"]         = state["MIBlock"][1]
      state["MatrixInstK"]         = state["MIBlock"][2]
      state["MatrixInstB"]         = state["MIBlock"][3]
      state["MatrixInstBM"]        = state["MIBlock"][4]
      state["MatrixInstBN"]        = state["MIBlock"][5]

      state["LocalSplitU"]         = 1
      state["MIOutputVectorWidth"] = 4

      if state["MatrixInstM"] == 4:
        state["ThreadTile0"] = state["MIWaveTile"][0] * state["MIOutputVectorWidth"]
        state["ThreadTile1"] = state["MIWaveTile"][1]
        state["SubGroup0"]   = state["MIWaveGroup"][0] * state["MatrixInstM"] * state["MatrixInstBM"] // state["MIOutputVectorWidth"]
        state["SubGroup1"]   = state["MIWaveGroup"][1] * state["MatrixInstN"] * state["MatrixInstBN"]
      else:
        state["ThreadTile0"] = state["MatrixInstBM"] * state["MIWaveTile"][0] * (state["MatrixInstM"] * state["MatrixInstN"] // globalParameters["WavefrontWidth"])
        state["ThreadTile1"] = state["MatrixInstBN"] * state["MIWaveTile"][1]
        state["SubGroup0"]   = state["MIWaveGroup"][0] * (globalParameters["WavefrontWidth"] // state["MatrixInstN"])
        state["SubGroup1"]   = state["MIWaveGroup"][1] * state["MatrixInstN"]

    elif EnableMatrixInstruction == False:
      state["ThreadTile0"] = state["ThreadTile"][0]
      state["ThreadTile1"] = state["ThreadTile"][1]

      state["SubGroup0"]   = state["WorkGroup"][0]
      state["SubGroup1"]   = state["WorkGroup"][1]
      state["LocalSplitU"] = state["WorkGroup"][2]

    if "SubGroup0" in state and "SubGroup1" in state and "LocalSplitU" in state:
      state["NumThreads"]  = state["SubGroup0"] * state["SubGroup1"] * state["LocalSplitU"]

    # macro tile sizes
    if "SubGroup0" in state and "ThreadTile0" in state:
      state["MacroTile0"] = state["SubGroup0"]*state["ThreadTile0"]
    if "SubGroup1" in state and "ThreadTile1" in state:
      state["MacroTile1"] = state["SubGroup1"]*state["ThreadTile1"]
    if "MacroTile" in state:
      if state["MacroTile0"] != state["MacroTile"][0] \
          or state["MacroTile1"] != state["MacroTile"][1]:
        reject(state, "MacroTile mismatch")

    if state["Valid"] and "MacroTileShapeMax" in state \
        and "MacroTileShapeMin" in state:
      macroTileShape = max(state["MacroTile0"]//state["MacroTile1"], \
          state["MacroTile1"]//state["MacroTile0"])
      if macroTileShape > state["MacroTileShapeMax"] \
          or macroTileShape < state["MacroTileShapeMin"]:
        reject(state, "rejecting MacroTile Shape %u:%u for Min:Max %u:%u" \
            % (state["MacroTile0"], state["MacroTile1"], \
            state["MacroTileShapeMin"], state["MacroTileShapeMax"]))

    if "WorkGroupMappingType" in state:
      if state["WorkGroupMappingType"] == "Z":
        if abs(state["WorkGroupMapping"]) > 2:
          reject(state, "WorkGroupMappingType=Z only supports WorkGroupMapping=1, 2")

    # done
    state["AssignedProblemIndependentDerivedParameters"] = True

  ########################################
  # This is the "classic" algorithm which requires that each threads load the same number of bytes
  # Called with tc=A and then with tc=B
  # totalVectors is totalElements/GRVW, this is #vectors loaded by the LoadTile
  # Reduces the GlobalLoadVectorWidth if necessary if each thread has a small amount of work to do.
  # Output from this function:
  #  state[GlobalLoadVectorWidth*]
  #  state[NumLoads*] # only used in SolutionStructs, with classic alg
  @staticmethod
  def setGlobalLoadVectorWidth(state, tc, totalVectors):
    validDepthU = True
    if totalVectors < state["NumThreads"]:
      # Try to reduce size of vector so every thread has a load to do
      pv = state["NumThreads"]//totalVectors
      if not state["FractionalLoad"]:
        if state["NumThreads"] % totalVectors != 0:
          reject(None, "NumThreads %u %% totalVectors %u != 0" \
              % (state["NumThreads"], totalVectors))
          validDepthU = False
        if pv * totalVectors != state["NumThreads"]:
          reject(None, "pv %u * totalVectors %u != NumThreads %u" \
              % (pv, totalVectors, state["NumThreads"]))
          validDepthU = False
        if state["GlobalReadVectorWidth"] % pv != 0:
          reject(None, "NumThreads %u %% totalVectors %u != 0" \
              % (state["NumThreads"], totalVectors))
          validDepthU = False
    else:
      pv = 1 # no partial vector required
      if totalVectors % state["NumThreads"] != 0:
        if not state["FractionalLoad"]:
          reject(None, "totalVectors %u %% NumThreads %u != 0" \
              % (totalVectors, state["NumThreads"]))
          validDepthU = False

    state["GlobalLoadVectorWidth%s"%tc] = state["GlobalReadVectorWidth"]//pv

    # NumLoads is NOT used on the fractional path
    # NumLoads is number of vector loads per-thread
    state["NumLoads%s"%tc] = totalVectors * pv // state["NumThreads"]
    #print "result: ", pvar(state, "GlobalLoadVectorWidth%s"%tc), \
    #        pvar(state, "NumLoads%s"%tc)

    return validDepthU

  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalescedA]
  #   state[NumLoadsPerpendicularA]
  #   state[LSCA]
  #   state[LSPA]
  @staticmethod
  def setGlobalLoadTileDimClassic(state, tc, numLoads, totalVectorsCoalesced, totalElementsPerp):
    # nlc = 1
    if state["NumLoadsCoalesced%s"%tc] == 1 :
      foundValid = False
      for nlc in range(1, int(state["NumLoads%s"%tc]+1)):
        nlp = state["NumLoads%s"%tc] // nlc
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          #print("NumLoadsCoalesced",state["NumLoadsCoalesced%s"%tc])
          #print("NumLoadsPerpendicular",state["NumLoadsPerpendicular%s"%tc])
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=1 found"%tc)
        return False

    # nlc = -1
    elif state["NumLoadsCoalesced%s"%tc] == -1:
      foundValid = False
      for nlc in range(state["NumLoads%s"%tc], 0, -1):
        nlp = state["NumLoads%s"%tc] // nlc
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=-1 found"%tc)
        return False

    # nlc = other
    else:
      if state["NumLoadsCoalesced%s"%tc] > state["NumLoads%s"%tc]:
        reject(state, "%s nlc > numLoads"%tc)
        return False

      state["NumLoadsPerpendicular%s"%tc] = state["NumLoads%s"%tc] \
          // state["NumLoadsCoalesced%s"%tc]

      if state["NumLoads%s"%tc] % state["NumLoadsCoalesced%s"%tc] != 0:
        reject(state, "%s: numLoads %u %% numLoadsCoalesced %u != 0" \
            % (tc, state["NumLoads%s"%tc], state["NumLoadsCoalesced%s"%tc]))
        return False

      if totalVectorsCoalesced % state["NumLoadsCoalesced%s"%tc] != 0 :
        reject(state, "%s: totalVectorsCoalesced %u %% numLoadsPara %u != 0" \
              % (tc, totalVectorsCoalesced, state["NumLoadsCoalesced%s"%tc]))
        return False
      if totalElementsPerp % state["NumLoadsPerpendicular%s"%tc] != 0:
        reject(state, "%s: totalElementsPerp %u %% numLoadsPerp %u != 0" \
              % (tc, totalElementsPerp, state["NumLoadsPerpendicular%s"%tc]))
        return False

    if state["ProblemType"]["TLU%s"%tc]:
      state["LSC%s"%tc] = state["MacroTile%s"%tc] \
          // state["NumLoadsCoalesced%s"%tc]
      state["LSP%s"%tc] = int(math.ceil(float(state["DepthU"]) / state["NumLoadsPerpendicular%s"%tc]))
    else:
      state["LSC%s"%tc] = int(math.ceil(float(state["DepthU"]) / state["NumLoadsCoalesced%s"%tc]))
      state["LSP%s"%tc] = state["MacroTile%s"%tc] \
         // state["NumLoadsPerpendicular%s"%tc]

    return True


  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalesced*]
  #   state[NumLoadsPerpendicular*]
  #   state[LSC*]
  #   state[LSP*]
  #   state[GlobalReadVectorWidth]
  #
  # LSC and LSP define the shape of the PerLoadTile, measured in elements.
  #   LSC*LSP is the elements loaded by a single instruction across all
  #   threads in the group.
  #   LSC is the number of elements loaded in the para(coalesced) dimension
  #   LSP is the number of elements loaded in the perp(noncoalesced) dimension
  #   PerLoadTile is always rectangular.
  #   When BufferLoad=1, the area (LSC*LSP) can be larger than NumThreads.
  #   In this case, some threads will generate a dummy OOB GRO.
  #   Related fields:
  #     LVC = LSC/GRVW  (LVCA = LSCA/GLVWA)
  #     LVP = LSP/GRVW  (LVPA = LSPA/GLVWA)
  #
  # NumLoadsCoalesced and NumLoadsPerpendicular define the number of times the
  #   PerLoadTile is loaded in each dimension to fetch the LoadTile
  # LoadTile = (LSC * NumLoadsCoalesced) * (LSP * NumLoadsPerpendicular).
  #   For Fractional, the LoadTile can be larger than the MacroTile. Buffer
  #   loads will clip any OOB references to 0 and will also avoid writing these
  #   into LDS.

  # Fractional load algorithm:
  #  - Each load instruction loads one or more (complete) rows of the load tile.
  #     - Each row is LSC elements wide
  #     - Rows are complete and do not wrap. This allows a single base GRO VGPR
  #       to be used for all loads in the tile.
  #     - Some work-items in the load may not perform useful work. These WI will
  #       set their GRO to a large OOB number so as to do no harm
  #     - Some G2L registers space may be unused as well.
  #     - The 'used' message at the bottom of this routine computes and prints the
  #       wasted register space.
  #     - The wasted space is removed when the data is written to LDS- the LWO
  #       for work-items beyond the valid ones are set to safely write to OOB locations.

  #     - In cases where each load is loading multiple rows (multiple lines of lsc
  #       elements), the last load is allowed to load fewer lines than the others.
  #       The KernelWriterAssembly will modify the LWO for the last load.  This allows
  #       flexibility in the unroll factors for example.
  @staticmethod
  def setGlobalLoadTileDimFractional(state, tc, depthU):

    assert(depthU > 0)
    dbFract = 0

    # parDim, perpDim define the LoadTile and are measured in elements
    if state["ProblemType"]["TLU%s"%tc]:
      parDim  = state["MacroTile%s"%tc]
      perpDim = depthU
    else:
      parDim  = depthU
      perpDim = state["MacroTile%s"%tc]

    if dbFract:
        print("\ninfo: %s Fractional MT%u_%u_%u Par=%u Perp=%u WG%02u_%02u_%02u NumThreads=%u GRWV=%u" \
          % (tc, state["MacroTile0"], state["MacroTile1"], depthU, \
            parDim, perpDim, \
            state["WorkGroup"][0], state["WorkGroup"][1], state["LocalSplitU"], \
            state["NumThreads"], state["GlobalReadVectorWidth"]))

    # Try to find a GRVW which is smaller than the LSC and also does not force
    # the LSC to wrap - both of these conditions can be tested with lsc % grvw ==0.
    # Each iteration divides GRWV by 2 which provides finer granularity
    # and a possible opportunity to handle the lsc
    grvw = state["GlobalReadVectorWidth"]
    minGrvw = 2 if state["ProblemType"]["DataType"].isHalf() and \
                globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"] else 1
    bestVw = -1
    while grvw >= minGrvw:
      # Per instruction across the entire group:
      elementsLoadedPerInst = state["NumThreads"]*grvw
      # LSC, LSP - #elements loaded along specified dim with each load
      if parDim >= elementsLoadedPerInst:
        # entire work-group can work on (part) of the same row
        state["LSC%s"%tc] = elementsLoadedPerInst
        state["LSP%s"%tc] = 1
        state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim , state["LSC%s"%tc])
        state["NumLoadsPerpendicular%s"%tc] = 1
      else:
        # work-group exceeds read dimension so wraps to multiple rows
        state["LSC%s"%tc] = parDim
        state["LSP%s"%tc] = min(perpDim, elementsLoadedPerInst // parDim)
        state["NumLoadsCoalesced%s"%tc] = 1
        state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

      # Vector loads can't wrap to next P dim, so LSC must be divisible by vector elements;
      if dbFract:
        print("  lsc search : lsc(%u) %% grvw(%u) = %u (?0)" % (state["LSC%s"%tc], grvw, state["LSC%s"%tc] % grvw))
      if state["LSC%s"%tc] % grvw == 0:
        bestVw = grvw
        # Try to shrink GRVW if possible while keeping same LSC and LSP:
        # For example, avoid cases where we use a GRVW=4 with many empty addresses
        # when a GRVW=1 will do instead.
        validElementsLoadedPerInst = state["LSC%s"%tc] * state["LSP%s"%tc]
        grvw //= 2
        while grvw >= minGrvw:
          elementsLoadedPerInst = state["NumThreads"]*grvw
          if elementsLoadedPerInst < validElementsLoadedPerInst:
            break # Went too far, not enough load elements at this VW
          if state["LSC%s"%tc] % grvw == 0:
            if dbFract:
              print("  stepdown success (valid)elementsLoadedPerInst=", validElementsLoadedPerInst, "/", elementsLoadedPerInst, "grvw=", grvw, "lsc=", state["LSC%s"%tc])
            bestVw = grvw
          grvw //= 2
        break

      # TODO - could have this generate dwordx3 loads in addition, step down by 1 instead of div2
      # Would need to change asm code gen to generate x3
      grvw //= 2
      # end-- while loop

    if bestVw == -1:
      if dbFract:
        print ("reject fractional - no acceptable tile dim? GlobalReadVectorWidth", \
         state["GlobalReadVectorWidth"])
      return False  # could not find a solution, perhaps only possible for half ?

    state["GlobalLoadVectorWidth%s"%tc] = bestVw
    if bestVw != state["GlobalReadVectorWidth"]:
      if dbFract:
        print("  reducing GlobalLoadVectorWidth%s from %u to %u" \
            % (tc, state["GlobalReadVectorWidth"], bestVw))

    # How many loads per threads in each dimension.
    # threads which are outside the global read tile bounds will be clipped
    # in the assembly code generator.
    # Multiply the LSC*GRVW
    state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim, state["LSC%s"%tc])
    state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

    nlc = state["NumLoadsCoalesced%s"%tc]
    nlp = state["NumLoadsPerpendicular%s"%tc]

    # LoadTile must at least cover the MacroTile:
    assert(nlc*state["LSC%s"%tc] >= parDim)
    assert(nlp*state["LSP%s"%tc] >= perpDim)

    perpOverhang = perpDim % state["LSP%s"%tc]
    state["fractionalPerpOverhang%s"%tc] = perpOverhang
    if dbFract:
      # how many threads compute Global Read Offsets (GRO) that are not used
      print("  PerLoadTile=%ux%u elements Loads/WI=%ux%u LoadTile/WI=%ux%u (MT=%ux%u), %u/%u = %.1f%% WI GRO used %s" \
          % (state["LSC%s"%tc], state["LSP%s"%tc], \
             nlc, nlp, \
             nlc*state["LSC%s"%tc], nlp*state["LSP%s"%tc], \
             parDim, perpDim, \
             parDim*perpDim, \
             nlc*nlp*state["NumThreads"]*state["GlobalLoadVectorWidth%s"%tc], \
             float(parDim*perpDim), \
             float(nlc*nlp*state["NumThreads"]*state["GlobalLoadVectorWidth%s"%tc]) * 100.0) \
             )

      for p in range(0,nlp):
        elementWidth = 4
        if p != nlp-1:
          perp = state["LSP%s"%tc]
        else:
          perp = perpOverhang if perpOverhang else state["LSP%s"%tc]

        validElements = state["LSC%s"%tc] * perp
        print("  buffer_load_element_x%u %ux%ux%u bytes,  %u/%u valid GRO" %\
              (state["GlobalLoadVectorWidth%s"%tc], \
              state["LSC%s"%tc], perp, \
              elementWidth, \
              validElements//state["GlobalLoadVectorWidth%s"%tc],
              state["NumThreads"]))

    return True


  @staticmethod
  def parameterWrapper(state):
    state["LdsBlockSizePerPadA"] = state["LdsBlockSizePerPad"]
    state["LdsBlockSizePerPadB"] = state["LdsBlockSizePerPad"]

    state["UnrollMajorLDSA"]     = state["TransposeLDS"]
    state["UnrollMajorLDSB"]     = state["TransposeLDS"]

    if state["MatrixInstruction"] != [] and len(state["MatrixInstruction"]) == 4:
      # set EnableMatrixInstruction
      state["EnableMatrixInstruction"] = True

      # set MIBlock
      miwg0      = state["MatrixInstruction"][0] if (state["WorkGroup"][0] < state["MatrixInstruction"][0]) else state["WorkGroup"][0]
      MIBlock_BM = miwg0 // state["MatrixInstruction"][0]
      MIBlock_BM = min(MIBlock_BM, state["MatrixInstruction"][3])
      MIBlock_BN = state["MatrixInstruction"][3] // MIBlock_BM

      state["MIBlock"]    = [32, 32, 2, 1, 1, 1]
      state["MIBlock"][0] = state["MatrixInstruction"][0]
      state["MIBlock"][1] = state["MatrixInstruction"][1]
      state["MIBlock"][2] = state["MatrixInstruction"][2]
      state["MIBlock"][3] = state["MatrixInstruction"][3]
      state["MIBlock"][4] = MIBlock_BM
      state["MIBlock"][5] = MIBlock_BN

      # set MIWaveGroup
      numOfWave                = (state["WorkGroup"][0] * state["WorkGroup"][1]) // globalParameters["WavefrontWidth"]
      state['MIWaveGroup']     = [1, 1]
      state['MIWaveGroup'][0]  = min((miwg0 // state["MatrixInstruction"][0]) // MIBlock_BM, numOfWave)
      state['MIWaveGroup'][1]  = numOfWave // state['MIWaveGroup'][0]

      # set MIWaveTIle
      state['MIWaveTile']      = [1, 1]
      state['MIWaveTile'][0]   = state["ThreadTile"][0]
      state['MIWaveTile'][1]   = state["ThreadTile"][1] // state["MatrixInstruction"][1]

    else:
      state["EnableMatrixInstruction"] = False


  ########################################
  # assign all derived parameters
  @staticmethod
  def assignDerivedParameters(state):

    Solution.parameterWrapper(state)

    Solution.assignProblemIndependentDerivedParameters(state)

    for s in Solution.InternalKeys:
        state['_'+s] = state[s]
        #del state[s]

    if state["VectorStore"] == -1:
        state["_VectorStore"] = 1 # default, may be changed if needed to generate a valid kernel

    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    ProblemType.assignDerivedParameters(state["ProblemType"])
    if not state["Valid"]:
      print1("in assignDerivedParameters, state['Valid'] = False")
      return

    # Init LoopIters parameter in case of early exit
    # For backwards compatibility with older yaml files
    state["LoopIters"] = 0
    if "LoopUnroll" in state:
      state["LoopIters"] = state["LoopUnroll"]


    if state["EnableMatrixInstruction"]:
      if not (state["ProblemType"]["DataType"].isSingle() \
              or state["ProblemType"]["DataType"].isBFloat16() \
              or state["ProblemType"]["DataType"].isHalf()):
        reject(state, "didn't support Matrix Instruction with type %s" % str(state["ProblemType"]["DataType"]))
      if not state["MIBlock"] or len(state["MIBlock"]) != 6:
        reject(state, "invalid MIBlock")
      if not state["MIWaveGroup"] or len(state["MIWaveGroup"]) != 2:
        reject(state, "invalid MIWaveGroup")
      if not state["MIWaveTile"] or len(state["MIWaveTile"]) != 2:
        reject(state, "invalid MIWaveTile")
      if not state["ProblemType"]["HighPrecisionAccumulate"] \
         and not state["ProblemType"]["DataType"].isSingle() :
        reject(state, "Matrix instructions for half types are natively accumulated" + \
         " in fp32 precision. Please add the following config:" + \
         "\n - HighPrecisionAccumulate: True")
      if state["LdsBlockSizePerPadA"] != 0 and state["UnrollMajorLDSA"] == False:
        reject(state, "didn't support LdsBlockSizePerPadA on tile major LDS yet")
        if state["LdsBlockSizePerPadA"] < state["DepthU"]:
          reject(state, "reject: DepthU %u > LdsBlockSizePerPadA %u" % (state["DepthU"], state["LdsBlockSizePerPad"]))

      if state["LdsBlockSizePerPadB"] != 0 and state["UnrollMajorLDSB"] == False:
        reject(state, "didn't support LdsBlockSizePerPadB on tile major LDS yet")
        if state["LdsBlockSizePerPadB"] < state["DepthU"]:
          reject(state, "reject: DepthU %u > LdsBlockSizePerPadB %u" % (state["DepthU"], state["LdsBlockSizePerPad"]))
    else:
      if state["UnrollMajorLDSA"] or state["UnrollMajorLDSB"]:
        reject(state, "didn't support UnrollMajorLDS in VALU mode yet")

      if state["LdsBlockSizePerPadA"] != 0 or state["LdsBlockSizePerPadB"] != 0:
        reject(state, "didn't support LdsBlockSizePerPad in VALU mode yet")

      if state["ThreadTile0"] > 16 or state["ThreadTile1"] > 16:
        reject(state, "Invalid value for ThreadTile")

    if state["ProblemType"]["Tensor0"]==0:
      state["ThreadTileA"] = state["ThreadTile0"]
      state["ThreadTileB"] = state["ThreadTile1"]
      state["SubGroupA"] = state["SubGroup0"]
      state["SubGroupB"] = state["SubGroup1"]
      state["MacroTileA"] = state["MacroTile0"]
      state["MacroTileB"] = state["MacroTile1"]
    else:
      state["ThreadTileB"] = state["ThreadTile0"]
      state["ThreadTileA"] = state["ThreadTile1"]
      state["SubGroupB"] = state["SubGroup0"]
      state["SubGroupA"] = state["SubGroup1"]
      state["MacroTileB"] = state["MacroTile0"]
      state["MacroTileA"] = state["MacroTile1"]

    # Init vars early since there are early-exit return statements below
    state["DirectToLdsA"] = False
    state["DirectToLdsB"] = False
    state["LocalWriteUseSgprA"] = False
    state["LocalWriteUseSgprB"] = False

    state["WorkGroupMapping" ] = abs(state["WorkGroupMapping"])

    problemType = state["ProblemType"]
    if not problemType["UseInitialStridesAB"]:
      for (tc) in ('A','B'):
        state["AssertStride%sEqual"%tc][0]=1

    if not problemType["UseInitialStridesCD"]:
      for (tc) in ('C','D'):
        state["AssertStride%sEqual"%tc][0]=1

    # Add AssertStride*Equal for PackBatchDims, if needed
    for (mask, tc) in ((0x1,'B'), (0x2,'A')):
      if state["PackBatchDims"] & mask:
        for bi in problemType["IndicesBatch"]:
          state["AssertStride%sEqual"%tc][problemType["IndexAssignments%s"%tc].index(bi)] = 0

    for (tc,batchMask) in (('A', 0x1), ('B', 0x2)):
      freeDims = [i for i in problemType["IndexAssignments%s"%tc] if i in problemType["IndicesFree"]]
      if not freeDims and (not problemType["AllowNoFreeDims"] or not (state["PackBatchDims"] & batchMask)):
        reject(state, "tensor%s contains no free indices.  Set AllowNoFreeDims and PackBatchDims&%s" % (tc, batchMask))
        return False

    # Determine which indices will be packed together as this impacts several different parms (sizes, magic numbers, etc)
    # The order in PackedC*Indices also determines the order that dimensions are packed - the first elements in
    # the list are the fastest-moving elements.
    # The store code optimizes for C0 being the coalesced dimension and C1 the perp dimension.
    # C0/C1 indices can come from IndexAssignmentsA or IndexAssignmentsB
    # grid size [0,1]
    state["PackedC0IdxChars"] = []
    state["PackedC0IndicesX"] = []
    indexChars = globalParameters["IndexChars"]
    # Pack all the dimensions (batch and free) of A into grid[0]

    if problemType["Index0"] in problemType["IndexAssignmentsA"]:
      tc0 = 'A'
      tc1 = 'B'
    else:
      tc0 = 'B'
      tc1 = 'A'
    assert(isPackedIndex(state, problemType["Index01A"], 0x1))
    assert(isPackedIndex(state, problemType["Index01B"], 0x2))

    # Pack all the dimensions (batch and free) of A into grid[0]
    for idx in problemType["IndexAssignments%s"%tc0]:
      if isPackedIndex(state, idx, 0x1):
        assert (idx < problemType["NumIndicesC"])
        state["PackedC0IdxChars"].append("%s" % indexChars[idx])
        state["PackedC0IndicesX"].append(idx)

    state["PackedC1IdxChars"] = []
    state["PackedC1IndicesX"] = []
    for idx in problemType["IndexAssignments%s"%tc1]:
      if isPackedIndex(state, idx, 0x2):
        assert (idx < problemType["NumIndicesC"])
        state["PackedC1IdxChars"].append("%s" % indexChars[idx])
        state["PackedC1IndicesX"].append(idx)

    # If dims are packed, then need to ensure a global vector load isn't split by a tensor dim
    # (since this could result in non-contiguous addresses)
    # Current implementation ensures that the vector load is not partial across the Free* boundary:
    # GlobalLoadVectorWidth=1 will always meet this requirement.
    # (TODO - could make this more sophisticated if dims use default strides and are thus contiguous)
    packedC0 = len(state["PackedC0IdxChars"])>1
    packedC1 = len(state["PackedC1IdxChars"])>1

    bufferLoad = state["BufferLoad"] and state["KernelLanguage"] == "Assembly"
    if not bufferLoad:
      state["DirectToLds"] = False
      state["_UseSgprForGRO"] = False
      state["FractionalLoad"] = False

    #These modes only work under certain conditions, apply them here:
    #  - The "NoLoad" loop is only generated if PrefetchGlobalRead>0
    #  - And Suppress does not work if GSU>1 for some reason
    state["SuppressNoLoadLoop"] &= (bufferLoad and state["PrefetchGlobalRead"] and (state["GlobalSplitU"]==1))
    # Pointer swap only used if PGR=1 - so set ExpandPointerSwap=0 here
    state["ExpandPointerSwap"]  &= (bufferLoad and state["PrefetchGlobalRead"])

    #print("PackedC0IdxChars", state["PackedC0IdxChars"])
    #print("PackedC1IdxChars", state["PackedC1IdxChars"])

    # Set up stagger shift:
    bpeAB = int(4*state["ProblemType"]["DataType"].numRegisters())
    # (1<<staggerStrideShift) is number of loop iterations to traverse the stride
    try:
        staggerStrideShift = (int)(math.ceil(math.log(state["StaggerUStride"] / \
                (state["DepthU"] * bpeAB), 2)))
    except ValueError:
        staggerStrideShift = 0
    if staggerStrideShift < 0:
      reject(state, "StaggerUStride=%u is less than size of DepthU=%u * BytesPerElement=%u" \
        % (state["StaggerUStride"], state["DepthU"], bpeAB))
      return
    #print "staggerStrideShift=", staggerStrideShift, "depthu=", state["DepthU"]
    state["_staggerStrideShift"] = staggerStrideShift
    if state["StaggerU"] == 0:
      state["StaggerUMapping"] = 0

    # VectorWidth default handling
    if state["VectorWidth"] < 1:
      if state["EnableMatrixInstruction"]:
        state["VectorWidth"] = 2 if (state["ProblemType"]["DataType"].numRegisters() == 0.5) else 1
      else:
        state["VectorWidth"] = int(4 / state["ProblemType"]["DataType"].numRegisters())
        while state["ThreadTile0"] % state["VectorWidth"] != 0 \
            or state["ThreadTile1"] % state["VectorWidth"] != 0:
          state["VectorWidth"] //= 2

    # TT0,1 both must be multiples of VW, b/c of rC, rA, rB
    if not state["EnableMatrixInstruction"]:
      if state["ThreadTile0"] % state["VectorWidth"] != 0 \
          or state["ThreadTile1"] % state["VectorWidth"] != 0:
        reject(state, "ThreadTile0 %u or ThreadTile1 %u not a multiple of VectorWidth %u" \
            % (state["ThreadTile0"], state["ThreadTile1"], \
            state["VectorWidth"]))
        return

    if len(problemType["IndicesSummation"]) > 1:
      # not supported with multiple summations, bug is maybe something with
      # how stagger iteration is wraped when unroll loop exits
      state["StaggerU"] = 0

    # Some restrictions for half:
    if state["KernelLanguage"] == "Assembly" \
      and state["ProblemType"]["DataType"].isHalf():

      # Vector-width must be at least 2 for Half (since unroll loop uses packed operations?)
      if state["VectorWidth"] < 2:
        reject(state, "VectorWidth must be >= 2 for half")
      if globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"]:
        if not state["ProblemType"]["HighPrecisionAccumulate"] and state["AssertFree0ElementMultiple"] % 2 != 0:
          # beta-on-edge has AF0EM requirement except for HPA kernels
          reject(state, "Archs with HasEccHalf require AF0EM%2==0 except for HPA kernels")

    #if state["KernelLanguage"] == "Assembly" and state["PackSummationDims"]:
    #    reject(state, "PackSummationDims does not yet support assembly")

    # Default GlobalReadVectorWidth
    if state["GlobalReadVectorWidth"] == -1:
      state["GlobalReadVectorWidth"] = state["VectorWidth"]

    # Default GlobalStoreVectorWidth
    if state["StoreVectorWidth"] == -1:
      #TODO : re-enable later after running testlists
      #state["StoreVectorWidth"] = state["VectorWidth"]
      # use wider store for best store optimization
      state["StoreVectorWidth"] = 4


    if state["VectorWidth"]*state["ProblemType"]["DataType"].numBytes() > 16:
      # reject - VW too big
      reject(state, "VW * DataType.numBytes() > 16")

    if state["GlobalReadVectorWidth"]*state["ProblemType"]["DataType"].numBytes() > 16:
      # reject - GRVW too big
      reject(state, "GRVW * DataType.numBytes() > 16")

    # LocalSplitU too large?
    numElementsPerWorkGroup = state["MacroTile0"]*state["MacroTile1"]
    if numElementsPerWorkGroup < state["NumThreads"]:
      reject(state, "NumElementsPerWorkGroup %u < NumThreads %u; reduce LocalSplitU" \
          % (numElementsPerWorkGroup, state["NumThreads"]))
      return
    state["NumElementsPerThread"] = numElementsPerWorkGroup // \
        state["NumThreads"]
    state["GlobalWriteVectorWidth"] = min(state["VectorWidth"], state["NumElementsPerThread"] )
    if state["NumElementsPerThread"] % state["GlobalWriteVectorWidth"] != 0:
      reject(state, "LSU NumElementsPerThread %u not divisible into GWVW %u" \
          % (state["NumElementsPerThread"], state["GlobalWriteVectorWidth"]))
      return
    state["NumGlobalWriteVectorsPerThread"] = state["NumElementsPerThread"] \
        // state["GlobalWriteVectorWidth"]


    # LocalSplitU but can't NumThreads%MacroTile doesn't support sideways store
    if state["LocalSplitU"] > 1:
      if state["NumThreads"] % state["MacroTile0"] != 0:
        reject(state, "LocalSplitU but NumThreads=%u not divisible by MT0=%u for sideways store" \
            % (state["NumThreads"], state["MacroTile0"]))
        return
      if state["MacroTile0"]*state["MacroTile1"] % state["NumThreads"] != 0:
        reject(state, "LocalSplitU but MT0*MT1=%u elements doesn't divide into NumThreads=%u" \
            % (state["MacroTile0"]*state["MacroTile1"], state["NumThreads"]))
        return

    # GlobalSplitU doesn't work with some other things:
    if state["GlobalSplitU"] > 1:
      if not state["GlobalSplitUSummationAssignmentRoundRobin"] \
          and state["LoopTail"]:
        reject(state, "GlobalSplitU and LoopTail require SummationAssignmentRoundRobin=True since strongly breaks Tensile kernel architecture")
        return
      supported = \
        state["ProblemType"]["DataType"].isSingle() or \
        state["ProblemType"]["DestDataType"].isInt32() or \
        (state["KernelLanguage"] == "Assembly" and \
         (state["ProblemType"]["DataType"].isHalf() and \
          not state["ProblemType"]["HighPrecisionAccumulate"]))
      if not supported:
        reject(state, "GlobalSplitU only compatible with single or asm and (half or mixed) precision")
        return

    if state["VectorAtomicWidth"] == -1:
      if state["ProblemType"]["DataType"].isHalf():
        state["VectorAtomicWidth"] = 2
        #state["VectorAtomicWidth"] = 8 / state["ProblemType"]["DataType"].numBytes()
      else:
        state["VectorAtomicWidth"] = 1 # TODO - remove this and next line when VAW works for other types

    if state["VectorAtomicWidth"] >= 2 \
       and not state["ProblemType"]["DataType"].isHalf():
         reject (state, "VectorAtomicWidth>=2 only supported for half")

    if state["ProblemType"]["DataType"].isHalf() and \
      state["KernelLanguage"] == "Assembly":

      if state["VectorWidth"] < 2:
        reject(state, "Assembly half requires VectorWidth >= 2")

      if state["GlobalSplitU"] > 1:
        if state["VectorAtomicWidth"] <2:
          reject(state, "Assembly GSU half requires VectorWidth >= 2 (for 32-bit CAS)")

        if state["AssertFree0ElementMultiple"] < 2:
          reject(state, "Assembly GSU half requires AF0EM>=2 (for atomics on edge tiles)")

    ########################################
    # Initial DepthU
    ########################################
    userDepthU = state["DepthU"]
    # DepthU == -1 means glvw=1
    if state["DepthU"] == -1:
      if state["MacroTile0"] != state["MacroTile1"]:
        reject(state, "DepthU=0 requires square MacroTile")
        return

    if userDepthU < 0:
      depthU = 2
      maxDepthU = globalParameters["MaxDepthU"]
    else:
      depthU = userDepthU
      maxDepthU = userDepthU

    ########################################
    # Search DepthU
    ########################################
    while True: # exit criteria at end
      validDepthU = True

      if depthU % (state["PrefetchLocalRead"]+1) != 0:
        validDepthU = False

      # how many elements to load
      if state["ProblemType"]["TLUA"]:
        totalElementsCoalescedA = state["MacroTile0"]
        totalElementsPerpA = depthU
      else:
        totalElementsCoalescedA = depthU
        totalElementsPerpA = state["MacroTile0"]

      if state["ProblemType"]["TLUB"]:
        totalElementsCoalescedB = state["MacroTile1"]
        totalElementsPerpB = depthU
      else:
        totalElementsCoalescedB = depthU
        totalElementsPerpB = state["MacroTile1"]

      totalElementsA = totalElementsCoalescedA * totalElementsPerpA
      totalElementsB = totalElementsCoalescedB * totalElementsPerpB


      if state["FractionalLoad"]:
        if not Solution.setGlobalLoadTileDimFractional(state, "A", depthU):
          validDepthU = False
        if not Solution.setGlobalLoadTileDimFractional(state, "B", depthU):
          validDepthU = False
      else:
        tva = totalElementsA // state["GlobalReadVectorWidth"]
        tvb = totalElementsB // state["GlobalReadVectorWidth"]
        if not Solution.setGlobalLoadVectorWidth(state, "A", tva):
          validDepthU = False
        if not Solution.setGlobalLoadVectorWidth(state, "B", tvb):
          validDepthU = False

      if validDepthU and state["KernelLanguage"] == "Assembly" \
         and (state["ProblemType"]["DataType"].isHalf() \
              or state["ProblemType"]["DataType"].isBFloat16()):
        if globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"]:
          if state["GlobalLoadVectorWidthA"] == 1 or state["GlobalLoadVectorWidthB"] == 1:
            reject(state, "HalfEcc requires GLVWA > 1")


      # Now convert elements to vectors based on GlobalReadVectorWidth
      totalVectorsCoalescedA = totalElementsCoalescedA // state["GlobalReadVectorWidth"]
      totalVectorsCoalescedB = totalElementsCoalescedB // state["GlobalReadVectorWidth"]
      totalVectorsA = totalElementsA // state["GlobalReadVectorWidth"]
      totalVectorsB = totalElementsB // state["GlobalReadVectorWidth"]

      if 0:
        print("info:", pvar(state, "NumThreads"), pvar(state, "DepthU"), \
                       pvar(state, "ThreadTile0"), pvar(state, "ThreadTile1"), \
                       "WG=%ux%u" % (state["WorkGroup"][0], state["WorkGroup"][1]), \
                       pvar(state, "MacroTileA"), pvar(state, "MacroTileB"))
        print("info: totalElementsCoalescedA=", totalElementsCoalescedA, \
              " totalVectorsCoalescedA=", totalVectorsCoalescedA, " totalVectorsA=", totalVectorsA)
        print("info: totalElementsCoalescedB=", totalElementsCoalescedB, \
              " totalVectorsCoalescedB=", totalVectorsCoalescedB, " totalVectorsB=", totalVectorsB)
        print ("info", pvar(state, "VectorWidth"))
                #, pvar(state, "GlobalLoadVectorWidthA"), pvar(state, "GlobalLoadVectorWidthB"))

      #if state["ProblemType"]["DataType"].isHalf() \
      #    and (state["GlobalLoadVectorWidthA"] == 1 \
      #    or state["GlobalLoadVectorWidthB"] == 1):
      #  validDepthU = False

      if not state["FractionalLoad"]:
        if userDepthU == -1: # no vectors
          if state["GlobalLoadVectorWidthA"] != 1 \
              or state["GlobalLoadVectorWidthB"] != 1:
            validDepthU = False
        elif userDepthU == -2:
          if max( state["GlobalLoadVectorWidthA"], \
              state["GlobalLoadVectorWidthB"]) \
              < state["GlobalReadVectorWidth"]:
            validDepthU = False
        elif userDepthU <= -3:
          if min( state["GlobalLoadVectorWidthA"], \
              state["GlobalLoadVectorWidthB"]) \
              < state["GlobalReadVectorWidth"]:
            validDepthU = False

      if validDepthU:
        if not state["ProblemType"]["TLUA"]:
          if depthU < state["GlobalLoadVectorWidthA"]:
            validDepthU = False

        if not state["ProblemType"]["TLUB"]:
          if depthU < state["GlobalLoadVectorWidthB"]:
            validDepthU = False

      # this depthU is valid, done unless user wants to double (for TN)
      if validDepthU:
        if userDepthU < -3: # for every int below -3, use next doubled value
          userDepthU += 1
          depthU *= 2
          continue
        else: # use this found value
          state["DepthU"] = depthU
          break

      # this depthU not valid
      else:
        # keep looking
        if depthU < maxDepthU:
          depthU += 2
          continue
        # give up
        else:
          reject(state, "No valid DepthU found")
          return
    ########################################
    # end DepthU loop
    ########################################

    assert(state["DepthU"]> 0)

    if state["UnrollIncIsDepthU"] or state["PackSummationDims"] == 1 \
       or bool(problemType["ZeroPadA"]) or bool(problemType["ZeroPadB"]):
        # unrollIncIsDepthU does not support tail loop, so add asem requirement to reject
        # problems that require tail loop.
        if state["DepthU"] % state["AssertSummationElementMultiple"] != 0:
          reject(state, "PackSummationDims=1 requires DepthU is integer multiple of ASEM")
        else:
          state["AssertSummationElementMultiple"] = state["DepthU"]
        # not supported with PSD, has some interaction with iter
        state["StaggerU"] = 0

    if not state["FractionalLoad"]:
      if not Solution.setGlobalLoadTileDimClassic(state, "A", state["NumLoadsA"], \
          totalVectorsCoalescedA, totalElementsPerpA):
        return
      if not Solution.setGlobalLoadTileDimClassic(state, "B", state["NumLoadsB"], \
          totalVectorsCoalescedB, totalElementsPerpB):
        return

    # TODO
    if (0 and state["LSCA"] % state["GlobalLoadVectorWidthA"] != 0):
      reject(state, "lsca % grvw != 0")
      return
    if (0 and state["LSPA"] % state["GlobalLoadVectorWidthA"] != 0):
      reject(state, "lspa % grvw != 0")
      return
    if (0 and state["LSCB"] % state["GlobalLoadVectorWidthB"] != 0):
      reject(state, "lscb % grvw != 0")
      return
    if (0 and state["LSPB"] % state["GlobalLoadVectorWidthB"] != 0):
      reject(state, "lspb % grvw != 0")
      return

    state["LVCA"] = roundupRatio(state["LSCA"] , state["GlobalLoadVectorWidthA"])
    state["LVPA"] = roundupRatio(state["LSPA"] , state["GlobalLoadVectorWidthA"])
    state["LVCB"] = roundupRatio(state["LSCB"] , state["GlobalLoadVectorWidthB"])
    state["LVPB"] = roundupRatio(state["LSPB"] , state["GlobalLoadVectorWidthB"])

    for tc in ('A','B'):
      if problemType["TLU%s"%tc]:
        pos = problemType["IndexAssignments%s"%tc].index(problemType["Index01%s"%tc])
      else:
        pos = problemType["IndexAssignments%s"%tc].index(problemType["IndexUnroll"])

      unitStride = False
      stride = -1
      stride = state["AssertStride%sEqual"%tc].get(pos,-1)
      if stride==1:
        unitStride = True
      if not unitStride and state["GlobalLoadVectorWidth%s"%tc] != 1:
        reject(state,
            "Non-unit stride(%s) for coalesced dimension (index=%d) requires GlobalLoadVectorWidth%s==1" \
                % ("TBD" if stride==-1 else str(stride), \
                   problemType["IndexAssignments%s"%tc][pos], tc))

      for p in state["AssertStride%sEqual"%tc].keys():
        if p>len(problemType["IndexAssignments%s"%tc]):
          raise RuntimeError ("AssertStride%sEqual index position %d is > len(IndexAssignments%s)" % \
                                tc, p, tc)

    maxIndex = max(problemType["IndexAssignmentsA"] + problemType["IndexAssignmentsB"])
    for p in state["AssertSizeEqual"]:
      if p>maxIndex:
        raise RuntimeError ("AssertSize index position=%d is > maxIndex=%d" % (p, maxIndex))


    # Some of these might become 0?
    if 0:
      print("info: ", pvar(state, "LVCA"), pvar(state, "LVPA"), \
            pvar(state, "LVCB"), pvar(state, "LVPB"))

    # lds buffer size for A, B
    if state["KernelLanguage"] == "Source" and \
       state["LdsPadA"] != state["LdsPadB"]:
      reject(state, "Source KernelLanguage only supports LdsPadA == LdsPadB")
      return

    if state["LdsPadA"] == -1:
      state["LdsPadA"] = 0 if state["ProblemType"]["TLUA"] else state["VectorWidth"]
      assert(state["LdsPadA"] >= 0)
    if state["LdsPadB"] == -1:
      state["LdsPadB"] = 0 if state["ProblemType"]["TLUB"] else state["VectorWidth"]
      assert(state["LdsPadB"] >= 0)

    if (state["UnrollMajorLDSA"] or state["UnrollMajorLDSB"]) and (not state["EnableMatrixInstruction"]):
        reject(state, "UnrollMajorLDS Supports only in EnableMatrixInstruction=1")

    if state["LdsBlockSizePerPadA"] == -1:
      if state["EnableMatrixInstruction"] and state["UnrollMajorLDSA"]:
        state["LdsBlockSizePerPadA"] = 256
      else:
        state["LdsBlockSizePerPadA"] = 0

    if state["LdsBlockSizePerPadB"] == -1:
      if state["EnableMatrixInstruction"] and state["UnrollMajorLDSB"]:
        state["LdsBlockSizePerPadA"] = 256
      else:
        state["LdsBlockSizePerPadB"] = 0

    if state["LocalReadVectorWidth"] != -1:
      if (state["UnrollMajorLDSA"] == False or state["UnrollMajorLDSB"] == False):
        reject(state, "LocalReadVectorWidth requires UnrollMajorLDS=1")

    ldsAlign = int(64 / state["ProblemType"]["DataType"].numRegisters())

    if state["UnrollMajorLDSA"]:
      ldsNumElementsA = (state["DepthU"] + state["LdsPadA"]) * state["MacroTile0"]
      padInterval = state["LdsBlockSizePerPadA"] // bpeAB
      if padInterval != 0:
        ldsNumElementsA = int((state["DepthU"] * state["MacroTile0"]) / padInterval * (padInterval + state["LdsPadA"]))
      ldsNumElementsAlignedA = roundUpToNearestMultiple(ldsNumElementsA, ldsAlign)
    else:
      ldsNumElementsA = state["DepthU"] * (state["MacroTile0"] + state["LdsPadA"])
      ldsNumElementsAlignedA = roundUpToNearestMultiple(ldsNumElementsA, ldsAlign)

    if state["UnrollMajorLDSB"]:
      ldsNumElementsB = (state["DepthU"] + state["LdsPadB"]) * state["MacroTile1"]
      padInterval = state["LdsBlockSizePerPadB"] // bpeAB
      if padInterval != 0:
        ldsNumElementsB = int((state["DepthU"] * state["MacroTile1"]) / padInterval * (padInterval + state["LdsPadB"]))
      ldsNumElementsAlignedB = roundUpToNearestMultiple(ldsNumElementsB, ldsAlign)
    else:
      ldsNumElementsB = state["DepthU"] * (state["MacroTile1"] + state["LdsPadB"])
      ldsNumElementsAlignedB = roundUpToNearestMultiple(ldsNumElementsB, ldsAlign)

    # todo, can the alignment be a power of 2?
    state["LdsOffsetA"] = 0
    if state["PrefetchGlobalRead"]:
      state["LdsNumElementsAlignedA"] = ldsNumElementsAlignedA
      state["LdsNumElementsAlignedB"] = ldsNumElementsAlignedB
      state["LdsOffsetB"] = state["LdsOffsetA"] + state["LdsNumElementsAlignedA"]

      offsetBlk = state["LdsOffsetB"] + ldsNumElementsAlignedB
      offsetBlk = int(2**(math.ceil(math.log(offsetBlk, 2))))

      state["LdsOffsetA_Blk"] = offsetBlk
      state["LdsOffsetB_Blk"] = state["LdsOffsetA_Blk"] + state["LdsNumElementsAlignedA"]
      ldsNumElementsAB = state["LdsOffsetB_Blk"]+ ldsNumElementsB
    else:
      state["LdsOffsetB"] = ldsNumElementsAlignedA
      ldsNumElementsAB = ldsNumElementsAlignedA + ldsNumElementsB

    # lds buffer size for reduction
    ldsNumElementsReduction = state["LocalSplitU"]*state["MacroTile0"]*state["MacroTile1"] if state["LocalSplitU"] > 1 else 0

    # lds max occupancy
    ldsSizeOccupancy = globalParameters["DeviceLDS"] // state["MaxOccupancy"]
    ldsNumElementsOccupancy = ldsSizeOccupancy // state["ProblemType"]["DataType"].numBytes()

    #print("ldsNumElementsA", ldsNumElementsA)
    #print("ldsNumElementsB", ldsNumElementsB)
    #print("ldsNumElementsAlignedA", ldsNumElementsAlignedA)
    #print("ldsNumElementsAlignedB", ldsNumElementsAlignedB)
    #print("ldsNumElementsAB", ldsNumElementsAB)

    # lds size is the greater of the two
    ldsNumElements = max(ldsNumElementsAB, ldsNumElementsReduction, ldsNumElementsOccupancy)
    state["LdsNumElements"] = ldsNumElements
    ldsSize = ldsNumElements * state["ProblemType"]["DataType"].numBytes()
    if ldsSize > globalParameters["MaxLDS"]:
      reject(state, "Kernel Uses %u > %u bytes of LDS" % ( ldsSize, globalParameters["MaxLDS"]))
      return

    # LoopUnroll  = DepthU / LocalSplitU
    if "LocalSplitU" in state and "DepthU" in state:
      state["LoopUnroll"] = state["DepthU"] // state["LocalSplitU"]
    if state["LoopUnroll"] * state["LocalSplitU"] != state["DepthU"]:
      state["Valid"] = False
    if state["KernelLanguage"] != "Assembly" and state["InnerUnroll"] != 1:
      reject(state, "InnerUnroll only supported on assembly")
    state["LoopUnroll"] //= state["InnerUnroll"]
    ldl = state["LocalDotLayout"]
    if ldl > 1:
      # Disable DirectToLds for LDL > 1. Necessary because we need to swizzle the input data
      state["DirectToLds"] = False
      if (state["AssertSummationElementMultiple"] % ldl != 0):
        reject(state, "LocalDotLayout > 1 only supports ASEM a multiple of LDL")
        return
      if (state["ProblemType"]["HighPrecisionAccumulate"] != True or state["InnerUnroll"] != ldl):
        reject(state, "LocalDotLayout > 1 only supports HighPrecisionAccumulate set to true and InnerUnroll equal to LocalDotLayout")
        return

    if 0:
      print("info: ", pvar(state, "LoopUnroll"), " LDS Stats:", pvar(state, "LdsOffsetA"), pvar(state, "LdsOffsetB"))
      print("info: ", pvar(state["ProblemType"], "TLUA"), \
          pvar(state, "NumLoadsCoalescedA"), pvar(state, "NumLoadsPerpendicularA"), \
          pvar(state, "LSCA"), pvar(state, "LSPA"))
      print("info:", pvar(state["ProblemType"], "TLUB"), \
          pvar(state, "NumLoadsCoalescedB"), pvar(state, "NumLoadsPerpendicularB"), \
          pvar(state, "LSCB"), pvar(state, "LSPB"))

    state["LoopIters"] = state["LoopUnroll"]
    if "MatrixInstK" in state:
      state["LoopIters"] //= state["MatrixInstK"]

    # LoopIters should greater than PrefetchLocalRead
    if (state["LoopIters"] - state["PrefetchLocalRead"]) < 1:
      reject(state, "LoopIters %u should greater than PrefetchLocalRead %u" \
        % (state["LoopIters"],state["PrefetchLocalRead"]))

    # reject conditions with lower performance
    if state["ScheduleIterAlg"] == 2 and \
    (state["ExpandPointerSwap"] != 1 or state["LoopIters"] != 1 or state["ScheduleGlobalRead"] != 1):
      reject(state, "ScheduleIterAlg 2 only work with EPS1_SGW1, LoopIter=1")

    # Determine if we can load directly-to-LDS.
    # Transpose requires a trip through registers to perform the transpose so can't use DirectToLdsA
    # LDS loads always write 4 bytes apart so can use only 4-byte operations
    #   TODO - for doubles we need to add something special here?
    # The matrix must not require transposing since that is done by reading to VGPR and writing in different order
    # The LSC (load size coalesced) must load some multiple of 256 bytes since that is what each DirectToLds load provides
    # Note for these matrices LSC is same as MacroTile dim
    # MatrixInstruction rules:
    # DirectToLDS is supported for TLU=0  (make sure transposeLDS=1)
    # LDS (load size coalesced) * LSPA must load some multiple of 256 bytes. each DirecToLds instruction provides 256 bytes
    if state["DirectToLds"]:
      # The tail loop requires half summation elements be a multiple of two to use DirectToLds feature
      elementMultipleOk = not state["ProblemType"]["DataType"].isHalf() \
                          or state["AssertSummationElementMultiple"] % 2 == 0

      wavefronts = state["NumThreads"] // globalParameters["WavefrontWidth"]
      numBytes = state["ProblemType"]["DataType"].numBytes()

      # DirectToLds loads return 256 bytes/wave
      # If fractional, ensure we are using all of the bytes that will be delivered

      if elementMultipleOk and state["NumThreads"] % globalParameters["WavefrontWidth"] == 0:

        if state["EnableMatrixInstruction"]:
          # use with transposeLDS
          if (state["GlobalLoadVectorWidthA"] * numBytes == 4) \
            and (( not state["ProblemType"]["TransposeA"]  \
                   and state["LSCA"] * numBytes == 256 * wavefronts \
                   and state["LSCA"] * numBytes == state["NumThreads"] * 4 ) or \
                 ( state["ProblemType"]["TransposeA"] and state["UnrollMajorLDSA"]  \
                   and state["LSCA"] * state["LSPA"] * numBytes == 256 * wavefronts \
                   and state["LSCA"] * state["LSPA"] * numBytes == state["NumThreads"] * 4)) :
            state["DirectToLdsA"] = True
            state["LocalWriteUseSgprA"] = True

          if (state["GlobalLoadVectorWidthB"] * state["ProblemType"]["DataType"].numBytes() == 4) \
            and (( state["ProblemType"]["TransposeB"]  \
                   and state["LSCB"] * numBytes == 256 * wavefronts \
                   and state["LSCB"] * numBytes == state["NumThreads"] * 4 ) or \
                 ( not state["ProblemType"]["TransposeB"] and state["UnrollMajorLDSB"]  \
                   and state["LSCB"] * state["LSPB"] * numBytes == 256 * wavefronts \
                   and state["LSCB"] * state["LSPB"] * numBytes == state["NumThreads"] * 4)) :
            state["DirectToLdsB"] = True
            state["LocalWriteUseSgprB"] = True
        else:
          if (state["GlobalLoadVectorWidthA"] * numBytes == 4) \
            and not state["ProblemType"]["TransposeA"] \
            and state["LSCA"] * numBytes == 256 * wavefronts \
            and state["LSCA"] * numBytes == state["NumThreads"] * 4 :
            state["DirectToLdsA"] = True
            state["LocalWriteUseSgprA"] = True

          if (state["GlobalLoadVectorWidthB"] * state["ProblemType"]["DataType"].numBytes() == 4) \
            and state["ProblemType"]["TransposeB"] \
            and elementMultipleOk \
            and state["LSCB"] * numBytes == 256 * wavefronts \
            and state["LSCB"] * numBytes == state["NumThreads"] * 4 :
            state["DirectToLdsB"] = True
            state["LocalWriteUseSgprB"] = True

      if 0:
        print("DirectToLds Conditions (elementMultipleOk=", elementMultipleOk, \
              "wavefronts=", wavefronts, ")")
        print("  (LSCA)",state["LSCA"],"*", "(numBytes)", numBytes, "=?", "256 * (wavefronts)", wavefronts, \
              "=>", (state["LSCA"] * numBytes == 256 * wavefronts))
        print("  (LSCA)",state["LSCA"],"*", "(numBytes)", numBytes, "=?", state["NumThreads"], "* 4", \
              "=>", (state["LSCA"] * numBytes == state["NumThreads"]*4))
        print("  (LSCB)",state["LSCB"],"*", "(numBytes)", numBytes, "=?", "256 * (wavefronts)", wavefronts, \
              "=>", (state["LSCB"] * numBytes == 256 * wavefronts))
        print("  (LSCB)",state["LSCB"],"*", "(numBytes)", numBytes, "=?", state["NumThreads"], "* 4", \
              "=>", (state["LSCB"] * numBytes == state["NumThreads"]*4))

        print("A: TLU=", state["ProblemType"]["TLUA"], " MT=", state["MacroTile0"], \
               " LSCA=", state["LSCA"], "LSPA=", state["LSPA"], "GLVB_A=", state["GlobalLoadVectorWidthA"], \
               " dataTypeNumBytes=", state["ProblemType"]["DataType"].numBytes(), \
               "  ->DirectToLdsA=", state["DirectToLdsA"], \
               " NumLoadsCoalescedA=", state["NumLoadsCoalescedA"], \
               " NumLoadsPerpendicularA=", state["NumLoadsPerpendicularA"])
        print("B: TLU=", state["ProblemType"]["TLUB"], " MT=", state["MacroTile1"], \
               " LSCB=", state["LSCB"],"LSPB=", state["LSPB"],  "GLVB_B=", state["GlobalLoadVectorWidthB"], \
               " dataTypeNumBytes=", state["ProblemType"]["DataType"].numBytes(), \
               "  ->DirectToLdsB=", state["DirectToLdsB"], \
               " NumLoadsCoalescedB=", state["NumLoadsCoalescedB"], \
               " NumLoadsPerpendicularB=", state["NumLoadsPerpendicularB"])

      # Update parent variable so kernel display is accurate
      state["DirectToLds"] = state["DirectToLdsA"] or state["DirectToLdsB"]

    # Precise bounds check uses the "num_records" field in the buffer to
    # precisely detect when we are inbounds or not.  Only a one-dimensional
    # check is used since this is faster and also for computation we only
    # need to ensure that none of the loads fault.  threads which are
    # computing bogus sections of the C tile will later be ignored.
    # precise checking only works when all elements of the load are in-bounds
    # since if the vload crosses boundary we ignore all components not just the
    # ones that are OOB. See comments for groOffsetInMacroTile in KernelWriterAssembly.py
    #
    # So check for the cases where the unroll loop can
    # generate partial loads here and reject PBC solutions:
    # For non-TLU the free dim is in perp dim - should always be TRUE?  TODO
    if state["ProblemType"]["TLUA"]:
      state["GuaranteeNoPartialA"] = state["AssertFree0ElementMultiple"]%state["GlobalLoadVectorWidthA"]==0
    else:
      state["GuaranteeNoPartialA"] = True

    if state["ProblemType"]["TLUB"]:
      state["GuaranteeNoPartialB"] = state["AssertFree1ElementMultiple"]%state["GlobalLoadVectorWidthB"]==0
    else:
      state["GuaranteeNoPartialB"] = True

    #--
    # ShiftPtr can't use UseSgprForGRO since it needs to modify the VGPR pointers
    if bufferLoad and state["_UseSgprForGRO"] and state["EdgeType"]=="ShiftPtr":
      if not state["GuaranteeNoPartialA"] or not state["GuaranteeNoPartialB"]:
        state["_UseSgprForGRO"] = False
        #reject(state, "PBC with wide load has insufficient overlap guarantees- try GRVW=1 or adding appropriate Assert*ElementMultiple")

    if state["EnableMatrixInstruction"]:
      cont1 = not state["GuaranteeNoPartialA"]
      cont2 = ((state["MIOutputVectorWidth"] % state["GlobalLoadVectorWidthA"]) != 0)
      if cont1 and cont2:
        reject(state, "GlobalLoadVectorWidthA %u %% MIOutputVectorWidth %u must be 0" % \
          (state["GlobalLoadVectorWidthA"], state["MIOutputVectorWidth"]))

      cont1 = not state["GuaranteeNoPartialB"]
      cont2 = ((state["MIOutputVectorWidth"] % state["GlobalLoadVectorWidthB"]) != 0)
      if cont1 and cont2:
        reject(state, "GlobalLoadVectorWidthB %u %% MIOutputVectorWidth %u must be 0" % \
          (state["GlobalLoadVectorWidthB"], state["MIOutputVectorWidth"]))
    else:
      if not bufferLoad or not state["GuaranteeNoPartialA"]:
        # Restrict GRVW/VW combos so shift-ptr logic will work
        if state["GlobalLoadVectorWidthA"] > 1 \
            and state["GlobalLoadVectorWidthA"] != state["VectorWidth"]:
            reject(state, "GlobalLoadVectorWidthA %u must be == VectorWidth %u or == 1" % \
                    (state["GlobalLoadVectorWidthA"], state["VectorWidth"]))

      if not bufferLoad or not state["GuaranteeNoPartialB"]:
        # Restrict GRVW/VW combos so shift-ptr logic will work
        if state["GlobalLoadVectorWidthB"] > 1 \
            and state["GlobalLoadVectorWidthB"] != state["VectorWidth"]:
            reject(state, "GlobalLoadVectorWidthB %u must be == VectorWidth %u or == 1" % \
                    (state["GlobalLoadVectorWidthB"], state["VectorWidth"]))

    # these work everywhere, no special restrictions
    state["AssertMinApproxSize"] = 0

    if state["KernelLanguage"] == "Assembly":
      if state["VectorWidth"] > 1:
        # VW>1 kernels require dims>1
        state["AssertMinApproxSize"] = 3
    elif state["VectorWidth"] > 1:
      # VW>1 kernels require dims>1
      state["AssertMinApproxSize"] = 2

    # Use SGPR to store an offset from GlobalReadOffsetA+0.
    # (as opposed to using dedicated VGPR for each GRO
    # Requires preciseBounds check since we rely on the buffer bounds check, not
    # individual vector registers doing bounds compares.
    if not bufferLoad and state["FractionalLoad"]:
        reject(state, "Fractional requires BufferLoad")

    if state["_UseSgprForGRO"] == -1:
      # Don't use SGPR if it looks like we might not have enough - better to leave PBC enabled even if we have to use VGPR
      # 40 is based on current SGPR usage, this may need to be tuned in the future:
      numLoadsA = state["NumLoadsCoalescedA"]*state["NumLoadsPerpendicularA"]
      numLoadsB = state["NumLoadsCoalescedB"]*state["NumLoadsPerpendicularB"]
      if numLoadsA + numLoadsB > 35:
        #print "info: Disabling UseSgprForGRO since predicting too many SGPR will be used"
        state["_UseSgprForGRO"] = 0
      else:
        state["_UseSgprForGRO"] = 1


    if packedC0 and not state["GuaranteeNoPartialA"]:
      reject(state, "packedC0 requires GuaranteeNoPartialA")
    if packedC1 and not state["GuaranteeNoPartialB"]:
      reject(state, "packedC1 requires GuaranteeNoPartialB")

    if packedC0 or packedC1:
      state["_UseSgprForGRO"] = 0

      if state["EdgeType"] != "ShiftPtr":
        reject(state, "Packed dims requires EdgeType==ShiftPtr")
      if state["KernelLanguage"] == "Assembly":
        if not bufferLoad:
          reject(state, "Packed dims for Assembly requires BufferLoad")
        if not state["LdcEqualsLdd"]:
          # this would require an extra VGPR for addressing (since shared VGPRS are per-row)
          # and also would require that the dimension extraction and scale code be implemented
          # for LDD as well. see emitExtractAndScalePackedDims
          reject(state, "Packed dims for Assembly requires LdcEqualsLdd==True")

    if packedC0 and state["PackGranularity"]==2:
      if state["KernelLanguage"] == "Source":
        if state["AssertFree0ElementMultiple"]<state["VectorWidth"]:
          reject(state, "packedC0 Source requires AF0EM>=VectorWidth (for loads and stores)")
      else:
        if state["AssertFree0ElementMultiple"]<state["VectorWidth"]\
          or state["AssertFree0ElementMultiple"] == 1:
            if state["VectorStore"] <= 0:
              state["_VectorStore"] = 0
            else:
              reject(state, "packedC0 Assembly requires AF0EM>=VectorWidth or not VectorStore (for stores)")

    if state["AssertStrideCEqual"].get(0,-1) != 1 or state["AssertStrideDEqual"].get(0,-1) != 1:
      # Disable vector stores if not allowed:
      if state["VectorStore"] <= 0:
        state["_VectorStore"] = 0
      else:
        reject(state, "UseInitialStridesCD requires not VectorStore since store locations not adjacent")

    # Not currently suppored.  Support would require some changes in the
    # zeroPadRegs management:
    #   - don't allocate VGPRs for multiple perp/pad cases
    #   - guardZeroPad needs to add soffset to scalar calc
    if problemType["ZeroPadA"] or problemType["ZeroPadB"]:
      state["_UseSgprForGRO"] = 0

    # current requirement to avoid buffer loads that span multiple entries
    # if the summation dim participating in the ZeroPad is not fast-moving then
    # likely have more performant options.
    for tc in ('A', 'B'):
      if problemType["ZeroPad%s"%tc] and state["KernelLanguage"] == "Assembly":
        if state["GlobalLoadVectorWidth%s"%tc] != 1:
          reject(state, "asm ZeroPad requires GlobalLoadVectorWidth==1")
        if not bufferLoad:
          reject(state, "asm ZeroPad requires BufferLoad")

    # avoid bug somehow related to GlobalSplitU + Persistent
    # avoid bug related to WGM<0
    # avoid bug somehow related to HPA + Persistent
    if state["PersistentKernel"] and (\
            (state["KernelLanguage"] == "Assembly" and state["GlobalSplitU"] != 1) or \
            (state["KernelLanguage"] == "Assembly" and state["WorkGroupMapping"] < 0) or \
            (state["KernelLanguage"] == "Assembly" and problemType["HighPrecisionAccumulate"]) ):
      state["PersistentKernel"] = 0

    if state["MagicDivAlg"] == 2 and globalParameters["NewClient"] != 2:
      warn("Legacy client does not support MagicDivAlg==2, forcing MagicDivAlg=1")
      state["MagicDivAlg"] = 1

    if state["PackSummationDims"] == 2 and globalParameters["NewClient"] != 2:
      raise RuntimeError ("Legacy client does not support PackSummationDims (ASEM issues), aborting")

    if state["UnrollIncIsDepthU"] and globalParameters["NewClient"] != 2:
      raise RuntimeError ("Legacy client does not support UnrollIncIsDepthU=1 (ASEM issues), aborting")

    problemType["AssignedDerivedParameters"] = True


  ########################################
  # create a dictionary with booleans on whether to include parameter in name
  @staticmethod
  def getMinNaming(objs):
    # early return
    if len(objs) == 0:
      return {}
    # determine keys
    requiredParameters = {}
    if isinstance(objs[0], Solution):
      keys = list(objs[0]._state.keys())
    else:
      keys = list(objs[0].keys())
    # only 1, rather than name being nothing, it'll be everything
    if len(objs) == 1:
      for key in keys:
        if key in list(validParameters.keys()):
          requiredParameters[key] = False
    else:
      for key in keys:
        required = False
        if key in list(validParameters.keys()):
          for i in range(1, len(objs)):
            if objs[0][key] != objs[i][key]:
              required = True
              break
        if required:
          requiredParameters[key] = True
        else:
          requiredParameters[key] = False

    requiredParameters["ProblemType"]       = False # always prepended
    requiredParameters["MacroTile0"]        = False # always prepended
    requiredParameters["MacroTile1"]        = False # always prepended
    requiredParameters["DepthU"]            = False # always prepended
    requiredParameters["LdcEqualsLdd"]      = False # always prepended
    requiredParameters["MatrixInstruction"] = False # always prepended
    requiredParameters["MatrixInstM"]       = False # always prepended
    requiredParameters["MatrixInstN"]       = False # always prepended
    requiredParameters["MatrixInstK"]       = False # always prepended
    requiredParameters["MatrixInstB"]       = False # always prepended
    requiredParameters["MatrixInstBM"]      = False # always prepended
    requiredParameters["MatrixInstBN"]      = False # always prepended

    requiredParameters["Kernel"]       = True  # distinguish kernels from solutions
                                               # for single-source compilation
    return requiredParameters

  ########################################
  @ staticmethod
  def getNameFull(state):
    requiredParameters = {}
    for key in state:
      if key in list(validParameters.keys()):
        requiredParameters[key] = True
    return Solution.getNameMin(state, requiredParameters)

  ########################################
  # Get Name Min
  @ staticmethod
  def getNameMin(state, requiredParameters):
    name = ""
    first = True
    # put problem first
    if "ProblemType" in state:
      name += str(state["ProblemType"]) + "_"
    if "MacroTile0" in state \
        and "MacroTile1" in state \
        and "DepthU" in state:
      name += "%s%ux%ux%u_" \
          % ( Solution.getParameterNameAbbreviation("MacroTile"), \
          state["MacroTile0"], state["MacroTile1"], state["DepthU"] )
    if "MatrixInstM" in state:
      name += "%s%ux%ux%ux%u" \
          % ( Solution.getParameterNameAbbreviation("MatrixInstruction"), \
          state["MatrixInstM"], state["MatrixInstN"],  state["MatrixInstK"], state["MatrixInstB"])
    if "LdcEqualsLdd" in state:
      if state["LdcEqualsLdd"]:
        name += "SE_"
      else:
        name += "SN_"
    for key in sorted(state.keys()):
      if key in requiredParameters and key[0] != '_':
        if requiredParameters[key]:
          if not first:
            name += "_"
          else:
            first = False
          name += "%s%s" % ( Solution.getParameterNameAbbreviation(key), \
              Solution.getParameterValueAbbreviation(key, state[key]) )
    return name

  ########################################
  # create a dictionary of lists of parameter values
  @staticmethod
  def getSerialNaming(objs):
    data = {}
    for objIdx in range(0, len(objs)):
      obj = objs[objIdx]
      for paramName in sorted(obj.keys()):
        if paramName in list(validParameters.keys()):
          paramValue = obj[paramName]
          if paramName in data:
            if paramValue not in data[paramName]:
              data[paramName].append(paramValue)
          else:
            data[paramName] = [ paramValue ]
    maxObjs = 1
    for paramName in data:
      if not isinstance(data[paramName][0],dict):
        data[paramName] = sorted(data[paramName])
      maxObjs *= len(data[paramName])
    numDigits = len(str(maxObjs))
    return [ data, numDigits ]

  ########################################
  # Get Name Serial
  @ staticmethod
  def getNameSerial(state, serialNaming):
    data = serialNaming[0]
    numDigits = serialNaming[1]

    serial = 0
    multiplier = 1
    for paramName in sorted(state.keys()):
      if paramName in list(validParameters.keys()):
        paramValue = state[paramName]
        paramData = data[paramName]
        paramNameMultiplier = len(paramData)
        if paramValue in paramData:
          paramValueIdx = paramData.index(paramValue)
        serial += paramValueIdx * multiplier
        multiplier *= paramNameMultiplier
    name = "%s%0*u" % ("S" if isinstance(state, Solution) else "K", \
        numDigits, serial)
    return name


  ########################################
  @ staticmethod
  def getParametersIndented(state, indent):
    s = ""
    s += "%sProblemType: %s\n" % (indent, str(state["ProblemType"]))
    for key in sorted(state):
      s += "%s%s: %s\n" % (indent, str(key), str(state[key]))
    return s

  ########################################
  @ staticmethod
  def getParameterNameAbbreviation( name ):
    return ''.join([c for c in name if not c.islower()])

  ########################################
  @ staticmethod
  def getParameterValueAbbreviation( key, value ):
    if isinstance(value, str):
      return ''.join([c for c in value if c.isupper()])
    elif isinstance(value, bool):
      return "1" if value else "0"
    elif isinstance(value, int):
      if value >= 0:
        return "%u" % value
      else: # -1 -> n1
        return "n%01u" % abs(value)
    elif isinstance(value, ProblemType):
      return str(value)
    elif isinstance(value, tuple) or key == 'ISA':
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += str(value[i])
      return abbrev
    elif isinstance(value, list):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += Solution.getParameterValueAbbreviation(key, value[i])
        if i < len(value)-1:
          abbrev += "_"
      return abbrev
    elif isinstance(value, dict):
      s =  "_".join(["%d%d"%(pos,k) for pos,k in value.items()])
      return s
    else:
      printExit("Parameter \"%s\" is new object type" % str(value) )
      return str(value)

  # make class look like dict
  def keys(self):
    return list(self._state.keys())
  def __len__(self):
    return len(self._state)
  def __iter__(self):
    return iter(self._state)

  def __getitem__(self, key):
    return self._state[key]
  def __setitem__(self, key, value):
    self._name = None
    self._state[key] = value
  def __str__(self):
    if self._name is None:
      self._name = Solution.getNameFull(self._state)
    return self._name
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return deepcopy(self._state)
  def __hash__(self):
    return hash(str(self))
    #return hash(self.getAttributes())
  def __eq__(self, other):
    #return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()
    return isinstance(other, Solution) and str(self) == str(other)
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

