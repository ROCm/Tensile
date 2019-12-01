################################################################################
# Copyright (C) 2016-2019 Advanced Micro Devices, Inc. All rights reserved.
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

import sys,traceback
from functools import reduce
from .Common import globalParameters, defaultProblemType, assignParameterWithDefault, printExit, assignParameterRequired, defaultSolution, validParameters, print1
from .Common import validTensorAFormats, validTensorBFormats, validTensorDFormats, validConvolutionConfig
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
    traceback.print_stack(None, 2)
  if state != None:
    state["Valid"] = False

# print a labled variable
def pvar(state, field):
  return field + "=" + str(state[field])

def roundupRatio(dividend, divisor):
  return int(math.ceil(float(dividend) / float(divisor)))

class DimAB(Enum):
  OnlyA = 0  # Only allowed in A tensor
  OnlyB = 1  # Only allowed in B tensor
  BothAB = 2 # Must be in both A and B

class Fbs(Enum):
  Free=0     # Expect to be free dimension
  Batch=1    # Expect to be batch dimension
  Sum=2      # Expect to be summation dimension

################################################################################
class Convolution:
  class Dimension:
    """
    A description of the dimension - short char, usage, and const strides
    Dimensions are later assigned tensile indices and assigned to A/B
    based on the desired formats.
    """
    # stride=-1 indicates TBD stride; >=0 indicates a compile-time constant
    def __init__(self, shortChar, description, dimAB, strideA=-1, strideB=-1):
      self.shortChar = shortChar
      self.description = description
      self.dimAB = dimAB
      self.strideA=strideA
      self.strideB=strideB

    def __str__(self):
      s = "%5s : %s" % ("'%s'"%self.shortChar, self.description)
      if 1:
        if self.strideA != -1:
          s+=" [strideA:%d]"%self.strideA
        if self.strideB != -1:
          s+=" [strideB:%d]"%self.strideB
      #s += " (strideA=%d strideB=%d)" % (self.strideA, self.strideB)
      return s
    def __repr__(self):
      return self.shortChar

  SummaryProperties=[\
        'OperationType','DestDataType','DataType','HighPrecisionAccumulate',\
        'TensorAFormat','TensorBFormat','TensorDFormat',\
        'Filter', 'Stride','Dilation','PadStart','PadEnd','GroupCount',\
        'NumIndicesC', 'IndexAssignmentsA','IndexAssignmentsB',\
        'IndicesFree', 'IndicesBatch', 'IndicesSummation',\
        'SetConstStrideA', 'SetConstStrideB',\
        'UseBeta', 'UseInitialStridesAB', 'UseInitialStridesCD', \
        ]

  def __init__(self, problemTypeOut, convolutionType, config):

    self.convolutionDims={};
    self.convolutionType = convolutionType
    self.config = config

    for k in config:
      if k not in validConvolutionConfig:
        raise RuntimeError ("unknown convolution config field '%s'"%k)

    self.tensorAFormat  = config.get("TensorAFormat",  "NCHW")
    assert self.tensorAFormat in validTensorAFormats
    self.formatNumSpatialDims = len(self.tensorAFormat)-2
    assert (self.formatNumSpatialDims>=2 and self.formatNumSpatialDims<=3)

    self.tensorBFormat = config.get("TensorBFormat", "KCYX" if self.formatNumSpatialDims==2 else 'KCZYX')
    assert self.tensorBFormat in validTensorBFormats
    self.tensorDFormat = config.get("TensorDFormat",
          'KCYX' if convolutionType=='ConvolutionBackwardWeights' else self.tensorAFormat)

    if self.tensorDFormat == 0:
      self.tensorDFormat = self.tensorAFormat
    assert self.tensorDFormat in validTensorDFormats
    assert len(self.tensorAFormat) == len(self.tensorBFormat) == len(self.tensorDFormat)

    # index 0,1,2 = W,H,D = X,Y,Z
    if config.get("Spatial",None):
      self.spatial  = self.dimxParm(config, "Spatial",-1)
    else:
      self.spatial = None
    self.filter   = self.dimxParm(config, "Filter",1)
    self.stride   = self.dimxParm(config, "Stride",1)
    self.dilation = self.dimxParm(config, "Dilation",1)
    self.padStart = self.dimxParm(config, "PadStart",0)
    self.padEnd   = self.dimxParm(config, "PadEnd",0)
    self.packSpatialDims = config.get("PackedSpatialDims", 1)

    if not all(i==1 for i in self.stride[1:]):
      self.packSpatialDims = 0

    assert all(i==0 for i in self.padStart)  # padding not supported yet
    assert all(i==0 for i in self.padEnd)    # padding not supported yet
    assert (len(self.filter)==len(self.stride)==len(self.dilation) \
            ==len(self.padStart)==len(self.padEnd))

    self.groupCount = config.get("GroupCount", 1)
    self.indexAssignments = []

    # Index assignment have fastest-moving first
    ndim = Convolution.Dimension('N',   'Minibatch dimension. size#T=N.  strideB#T=0.', DimAB.BothAB)
    kdim = Convolution.Dimension('K',   'Cout. size#T=Cout.', DimAB.OnlyB)
    if self.packSpatialDims:
      constStrideA = -1 # default no const
      if self.stride[0] != -1:
        constStrideA=self.stride[0]
      sdims = [Convolution.Dimension('HW', \
          'Spatially packed HW. size#T=H_o*W_o. strideA#T=strideW(#S0).', \
          DimAB.OnlyA, strideA=constStrideA)]
    else:
      sdims = []
      schars = [1,'W','H','D']
      # sdims[0] is W
      for si in range(self.formatNumSpatialDims):
        sc=schars[si+1]
        constStrideA = -1 # default no const
        if si==0:
            if self.stride[si] != -1:
                constStrideA=self.stride[si]
            strideMsg = "stride%s(#S0)"%sc
        else:
            strideMsg = "%s_in*stride%s(#S%d)"%(schars[si],sc,si)
        sdims.append(Convolution.Dimension(sc,  \
            'Spatial %s. size#T=%s_o strideA#T=%s.'%(sc,sc,strideMsg), \
            DimAB.OnlyA, strideA=constStrideA))
    cdim = Convolution.Dimension('C', 'Cin.  size#T=Cin.  stride#T=1', DimAB.BothAB)

    if convolutionType in ("ConvolutionForward", "ConvolutionBackwardData"):
      # Make index assignments following standard Tensile Index assignment rules (see Common.py)
      # - Indices < NumCindices are batch or free indices and are present in TensorD
      # - Indices >= NumCindices are summation indices.  cidx is cin / summation so must be after nidx
      # - Memory order for TensorD is NumCindices...0, with 0 the fastest-moving dim.
      # Specific assignments to A and B (and associated impact on memory order of those tensors) is
      # specified by order of parms to registerA and registerB below.
      if self.tensorDFormat in ('NCHW','NCDHW', 'NHWC','NDHWC'):
        kidx = len(sdims)
        nidx = kidx+1
        cidx = nidx+1
      elif self.tensorDFormat in ("CNHW", "CNDHW"):
        # need to re-order batch dim to control memory order in output space
        nidx = len(sdims)
        kidx = nidx+1
        cidx = kidx+1

      sumIdx = cidx # cidx is first summation index, filter summations follow as needed
      # Create summation dimensions for non-unit filters and assign summation indices

      assert(len(self.filter)) == self.formatNumSpatialDims
      filterDims = []
      for (rfi,filterValue) in enumerate(self.filter[::-1]):
        if filterValue != 1:
          fi = self.formatNumSpatialDims - rfi -1 # forward filter index, 0...
          sumIdx = sumIdx+1
          filterChar = chr(ord('X')+fi)
          filterValueStr = "TBD" if filterValue==-1 else str(filterValue)
          prevChar = ['1', 'W', 'W*H']
          # TODO - stride setconst maybe applies only for NCHW/CNHW format not NHWC
          # can modify message here based on format or position of indices?
          filterMsg = "Filter%s. size#T=Filter%s(%s). strideA#T=Dilation%s(#D%d)*%s." \
              % (filterChar, filterChar, filterValueStr, filterChar, fi, \
                 prevChar[fi])
          filterDim = Convolution.Dimension(filterChar, filterMsg, DimAB.BothAB,
                       strideA=self.dilation[fi] if  fi==0 else -1)
          filterDims.append( (sumIdx, Fbs.Sum, filterDim) )

      spatialDims = []
      # reverse dims  so can pass spatialDims to register functions in 'convolution' order
      for si,sdim in enumerate(sdims):
        spatialDims.insert(0, (si, Fbs.Free, sdim))
      self.numSpatialDims = len(spatialDims) # dims actually used in the tensor

      if self.tensorAFormat in ("NCHW", "NCDHW"):
        self.registerA( [(nidx,Fbs.Batch,ndim), (cidx,Fbs.Sum,cdim)] + spatialDims + filterDims )
      elif self.tensorAFormat in ("NHWC", "NDHWC"):
        self.registerA( [(nidx,Fbs.Batch,ndim)] + spatialDims + filterDims + [(cidx,Fbs.Sum,cdim)] )
      elif self.tensorAFormat in ("CNHW", "CNDHW"):
        self.registerA( [(cidx,Fbs.Sum,cdim), (nidx,Fbs.Batch,ndim)] + spatialDims + filterDims )
      else:
        raise RuntimeError ("unknown activation format '%s'"%self.tensorAFormat)

      problemTypeOut["NumIndicesC"] = 2+len(spatialDims)

      transposeCK =  convolutionType=="ConvolutionBackwardData"
      ndim.strideB = 0
      if self.tensorBFormat in ("KCYX",'KCZYX') and not transposeCK or\
         self.tensorBFormat in ("CKYX",'CKZYX') and transposeCK:
        self.registerB( [(nidx,Fbs.Batch,ndim), (kidx,Fbs.Free,kdim), (cidx,Fbs.Sum,cdim)] + filterDims )
      elif self.tensorBFormat in ("CKYX",'CKZYX') and not transposeCK or\
           self.tensorBFormat in ("KCYX",'KCZYX') and transposeCK:
        self.registerB( [(nidx,Fbs.Batch,ndim), (cidx,Fbs.Sum, cdim), (kidx,Fbs.Free,kdim)] + filterDims )
      elif self.tensorBFormat in ("CYXK",'CZYXK') and not transposeCK or\
           self.tensorBFormat in ("KYXC",'KZYXC') and transposeCK:
        self.registerB( [(nidx,Fbs.Batch,ndim), (cidx,Fbs.Sum, cdim)] + filterDims + [(kidx,Fbs.Free,kdim)] )
      elif self.tensorBFormat in ("KYXC",'KZYXC') and not transposeCK or\
           self.tensorBFormat in ("CYXK",'CZYXK') and transposeCK:
        self.registerB( [(nidx,Fbs.Batch,ndim), (kidx,Fbs.Free, kdim)] + filterDims + [(cidx,Fbs.Sum,cdim)] )
      else:
        raise RuntimeError ("unknown weight format '%s'"%self.tensorBFormat)

    if convolutionType=="ConvolutionBackwardWeights":
      # index assignments - create filter dims
      filterDims = []
      #import pdb; pdb.set_trace()
      for (fi,filterValue) in enumerate(self.filter):
        if filterValue != 1:
          prevChar = ['1', 'W', 'W*H']
          filterChar = chr(ord('X')+fi)
          filterValueStr = "TBD" if filterValue==-1 else str(filterValue)
          filterMsg = "Filter%s. size#T=Filter%s(%s). strideA#T=Dilation%s(#D%d)*%s." \
              % (filterChar, filterChar, filterValueStr, filterChar, fi,\
                 prevChar[self.formatNumSpatialDims-fi-1])
          filterDim = Convolution.Dimension(filterChar, filterMsg, DimAB.OnlyA, \
                        strideA=self.dilation[fi] if  fi==0 else -1)
          # assign the tensile indices here - 0..number_of_nonunit_filters-1
          # insert like a stack to feed 'conv' order expected by register* functions
          filterDims.insert(0,( (len(filterDims), Fbs.Free, filterDim)))

      if self.tensorDFormat in ("KCYX", "KCZYX"):
        cidx=len(filterDims)  # free
        kidx=cidx+1      # free
        nidx=kidx+1      # summation
        sidxStart=nidx+1      # spatial summations (if needed)
      else:
        raise RuntimeError ("unknown tensorD format '%s'"%self.tensorDFormat)

      if self.packSpatialDims:
        sdims = [Convolution.Dimension('HW', \
            'Spatially packed HW. size#T=H_i*W_i. strideA#T=strideW(#S0).', \
            DimAB.BothAB, strideA=1)] # strides ignored since stride already applied
      else:
        raise RuntimeError ("backward weights only supports packSpatialDims")

      spatialDims = []
      # reverse dims  so can pass spatialDims to register functions in 'convolution' order
      for si,sdim in enumerate(sdims):
        spatialDims.insert(0, (sidxStart+si, Fbs.Sum, sdim))
      self.numSpatialDims = len(spatialDims) # dims actually used in the tensor


      if self.tensorAFormat in ("NCHW", "NCDHW"):
        self.registerA( [(nidx,Fbs.Sum,ndim), (cidx,Fbs.Free,cdim)] + filterDims + spatialDims)
        self.registerB( [(nidx,Fbs.Sum,ndim), (kidx,Fbs.Free,kdim)] + spatialDims)
      else:
        raise RuntimeError ("unknown tensorA format '%s'"%self.tensorAFormat)

      problemTypeOut["NumIndicesC"] = 2+len(filterDims)

    # convert from convolution order to tensor order:
    self.indexA.reverse()
    self.indexB.reverse()

    problemTypeOut["IndexAssignmentsA"] = [x[0] for x in self.indexA]
    problemTypeOut["IndexAssignmentsB"] = [x[0] for x in self.indexB]
    problemTypeOut["UseBeta"] = False # MI kernels don't use beta

    problemTypeOut["SetConstStrideA"]=[]
    for (idx,fbs,dim) in self.indexA:
      if dim.strideA != -1:
        problemTypeOut["SetConstStrideA"].append([idx,dim.strideA])
    problemTypeOut["SetConstStrideA"].sort()

    problemTypeOut["SetConstStrideB"]=[]
    for (idx,fbs,dim) in self.indexB:
      if dim.strideB != -1:
        problemTypeOut["SetConstStrideB"].append([idx,dim.strideB])
    problemTypeOut["SetConstStrideB"].sort()

    if [x for x in problemTypeOut["SetConstStrideA"] if x==[0,1]] or \
       [x for x in problemTypeOut["SetConstStrideB"] if x==[0,1]]:
      problemTypeOut["UseInitialStridesAB"] = False
    else:
      problemTypeOut["UseInitialStridesAB"] = True

    #self.printUsage(problemTypeOut)

  def dimIdx(self, convolutionChar):
    return self.convolutionDims[convolutionChar][0]

  def convolutionChar(self, dimIdx):
    return self.indexAssignments[dimIdx][1].shortChar

  @property
  def filterTbd(self):
    return -1 in self.filter

  @property
  def strideTbd(self):
    return -1 in self.stride

  @property
  def dilationTbd(self):
    return -1 in self.dilation

  @property
  def padTbd(self):
    return -1 in self.padStart or -1 in self.padEnd

  def makeProblem(self, keepTbd, n, c, k, spatialIn=None):
    """
    Generate valid problem dims for specified convolution
    Return [ [sizes],[stridesA] ]

    If keepTbd is true, then makeProblem will compute known values but return -1 for unknowns.
    For example, a constant filter parm can be used to compute some tensor sizes.  One the other hand,
    if spatial dims are not specified, then this function cannot compute the associated dimension size
    or stride and will leave them as -1.
      - -1 sizes are invalid and should be ignored by caller; however the other sizes can be used as a reference
      - -1 strides are valid.  (The client will compute a sensible default for -1 strides)

    TBD values are assumed to be 1 (filter/dilation/stride) or 0(pad) via abs(..) function
    """
    numDims = 1 + max(max([x[0] for x in self.indexA]), max([x[0] for x in self.indexB]))
    sizes = [-1]*numDims
    astrides = [-1]*numDims

    sizes[self.convolutionDims['N'][0]]=n
    sizes[self.convolutionDims['C'][0]]=c
    sizes[self.convolutionDims['K'][0]]=k

    if spatialIn==None:
      if self.spatial != None:
        spatialIn = self.spatial
      else:
        if keepTbd:
          spatialIn = [-1]*self.formatNumSpatialDims
        else:
          raise RuntimeError ("problemSize must specify spatial parms or set ConvolutionConfig.spatial")

    if len(spatialIn) != self.formatNumSpatialDims:
      raise RuntimeError ("len(spatialIn=", spatialIn, ") must match formatNumSpatialDims(%d)"%self.formatNumSpatialDims)

    spatialTbd = -1 in spatialIn

    # convert any TBD<0 to default 0
    padStart = [0 if p<0 else p for p in self.padStart]
    padEnd   = [0 if p<0 else p for p in self.padEnd]

    # convert to Output dimensions:
    spatialOut=[0]*len(spatialIn)
    for i in range(self.formatNumSpatialDims):
      if keepTbd and (spatialTbd or self.filterTbd or self.strideTbd or self.padTbd):
        spatialTbd = 1
        spatialOut[i] = -1
      else:
        spatialOut[i] = int((spatialIn[i] - abs(self.filter[i]) + 1 - padStart[i] - padEnd[i]) / abs(self.stride[i]))

    #import pdb; pdb.set_trace()
    for fi,filterValue in enumerate(self.filter):
      if filterValue != 1 and filterValue != -1:
        pos = self.convolutionDims[chr(ord('X')+fi)][0]
        if keepTbd and self.filterTbd:
          sizes[pos] = -1
        else:
          sizes[pos] = filterValue

        if keepTbd and (self.dilationTbd or self.strideTbd):
          astrides[pos] = -1
        else:
          astrides[pos] = abs(self.dilation[0]) if fi==0 else spatialIn[fi-1]*abs(self.dilation[fi])

    if self.numSpatialDims==1:
      spatialName="DHW"[3-self.formatNumSpatialDims:]
      pos=self.convolutionDims[spatialName][0]
      if keepTbd and spatialTbd:
        sizes[pos] = -1
      else:
        sizes[pos] = reduce((lambda x, y: x * y), spatialOut) # product of all spatial dimes
      if keepTbd and self.strideTbd:
        astrides[pos] = -1
      else:
        astrides[pos] = abs(self.stride[0])
    else:
      for si,sout in enumerate(spatialOut):
        spatialChars=['W','H','D']
        pos = self.convolutionDims[spatialChars[si]][0]
        if keepTbd and spatialTbd:
          sizes[pos] = -1
        else:
          sizes[pos] = sout

        if keepTbd and (spatialTbd or self.strideTbd):
          astrides[pos]=-1
        else:
          astrides[pos]=abs(self.stride[0]) if si==0 else spatialIn[si-1]*abs(self.stride[si])

    if not keepTbd:
      assert all(i!=-1 for i in sizes)

    # translate to strides for A tensor in IndexAssignmentsA order:
    orderedStrides=[]
    for (idx,fbs,dim) in self.indexA:
      orderedStrides.append(astrides[idx])

    return (sizes, orderedStrides)

  def registerA(self, dimList):
    """
    Provide a list of indices in convolution order - these will be reversed when assigned to IndexAssignmentsAB
    The order of items in the list determines the IndexAssignment order.
    Each tuple in the list is (idx,fbs,dim).
     - idx is the tensor index
     - fbs indicates if the tensor is expected to be Free,Sum,or Batch.  This is used for later check.
     - dim is Convolution.Dimension class that describes the dimension (for Usage info)
    """
    for (idx,fbs,dim) in dimList:
      if dim.dimAB not in (DimAB.OnlyA, DimAB.BothAB):
        raise RuntimeError ("dimension '%s' can't be registered to TensorA" % dim.shortChar)
      #print("registerA idx=", idx, "dim=", dim)
      try:
        self.indexAssignments[idx] = None
      except IndexError:
        self.indexAssignments.extend([None]*(1+idx-len(self.indexAssignments)))
      self.indexAssignments[idx] = (fbs,dim)
      self.convolutionDims[dim.shortChar] = (idx,dim)
    self.indexA = dimList

  def registerB(self, dimList):
    """ See registerA """
    for (idx,fbs,dim) in dimList:
      if dim.dimAB not in (DimAB.OnlyB, DimAB.BothAB):
        raise RuntimeError ("dimension '%s' can't be registered to TensorB" % dim.shortChar)
      #print("B", idx, dim)
      try:
        self.indexAssignments[idx] = None
      except IndexError:
        self.indexAssignments.extend([None]*(1+idx-len(self.indexAssignments)))
      self.indexAssignments[idx] = (fbs,dim)
      self.convolutionDims[dim.shortChar] = (idx,dim)

    self.indexB = dimList

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

  def printUsage(self, problemType):
    print("Tensile Index Assignments and Usage:")
    print("   Tensile    : ConvChar: Explanation/Usage")
    for (idx,dim2) in enumerate(self.indexAssignments):
        (fbs,dim)=dim2
        tensileChar = globalParameters['IndexChars'][idx]
        usage = str(dim)
        usage = usage.replace('#T', tensileChar)
        for i in range(len(self.stride)):
            usage = usage.replace('#S%d'%i, str(self.stride[i]) if self.stride[i]>=0 else 'TBD')
        for i in range(len(self.dilation)):
            usage = usage.replace('#D%d'%i, str(self.dilation[i]) if self.dilation[i]>=0 else 'TBD')
        print("  %d('%c') %-5s:   %s" % (idx, tensileChar, str(fbs).split('.')[1], usage))

    print ()
    print ("- Spatial sizes D_i, H_i, W_i refer to size of INPUT dimension.")
    print ("- Spatial sizes D_o, H_o, W_o refer to size of OUTPUT dimension.")
    print ("     For example W_o =  (W_i - X - padStart - padEnd + 1)/stride")
    print ("- (TBD)' indicates the parm is flexible and must be specified at runtime.")
    print ("- (i)' where i is an integer constant, indicates the parm is hard-coded at compile time.")
    print ("  The runtime value must match the compile-time value.")
    print ("- Unspecified strides use default stride value:")
    print ("    stride[i] = (stride[i-1]*size[i]) for i>0 ; 1 for i==0")

    print ()
    print ("ProblemType Definition:")
    for k in Convolution.SummaryProperties:
      try:
        print ("  ", k, ":", problemType[k])
      except KeyError:
        pass

  def checkDims(self, freeIndices, batchIndices, sumIndices):
    for dimList in (self.indexA, self.indexB):
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
    id += "_indices:" + '.'.join([x[1].shortChar for x in self.indexAssignments])
    if self.spatial:
      id += "_spatial:" + "x".join([str(x) for x in self.spatial[::-1]])
    id += "_filter:" + "x".join([str(x) for x in self.filter[::-1]])
    id += "_stride:" + "x".join([str(x) for x in self.stride[::-1]])
    id += "_dilation:" + "x".join([str(x) for x in self.dilation[::-1]])
    id += "_padStart:" + "x".join([str(x) for x in self.padStart[::-1]])
    id += "_padEnd:" + "x".join([str(x) for x in self.padEnd[::-1]])
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
      if globalParameters["PrintConvolutionUsage"]:
        print()
        self.convolution.printUsage(self)
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
          printExit("ZeroPad%s=%s freeDim=%u occurs in more than one tuple"%(tc, zp, freeDim))
        freeDims[freeDim] = 1
        if sumDim in sumDims:
          printExit("ZeroPad%s=%s sumDim=%u occurs in more than one tuple"%(tc, zp, sumDim))
        sumDims[sumDim] = 1

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
        #state["NumIndicesBatch"] = (i+1)-state["NumIndicesFree"]
        state["IndicesBatch"].append(i)

      elif inA or inB:
        #state["NumIndicesFree"] = (i+1)
        state["IndicesFree"].append(i)
      else:
        printExit("invalid index %u (inC but not (inA or inB))" % i)

    # determine num summation
    for i in range(state["NumIndicesC"], state["TotalIndices"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        #state["NumIndicesSummation"] = (i+1)-state["NumIndicesC"]
        state["IndicesSummation"].append(i)
      else:
        printExit("invalid index %u (expected summation but not (inA and inB))" % i)
    # print index assignments
    if 0:
      print1("IndicesFree:  %s" % state["IndicesFree"])
      print1("IndicesBatch: %s" % state["IndicesBatch"])
      print1("IndicesSum:   %s" % state["IndicesSummation"])
      print1("IndexAssignmentsA:   %s" % state["IndexAssignmentsA"])
      print1("IndexAssignmentsB:   %s" % state["IndexAssignmentsB"])


    for k in ('IndexAssignmentsA','IndexAssignmentsB'):
      if len(state[k]) != len(set(state[k])):
        printExit("duplicate index in %s=%s"% (k,state[k]))

    state["NumIndicesFree"] = len(state["IndicesFree"])
    state["NumIndicesBatch"] = len(state["IndicesBatch"])
    state["NumIndicesSummation"] = len(state["IndicesSummation"])
    if state["NumIndicesFree"] < 2 :
      printExit("Tensile requires >= 2 free indices; FreeIndices=%s."%state["IndicesFree"])

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
    state["Index01A"] = -1
    state["Index01B"] = -1
    for i in state["IndexAssignmentsA"]:
      if i in state["IndicesFree"]:
        state["Index01A"] = i
        break
    for i in state["IndexAssignmentsB"]:
      if i in state["IndicesFree"]:
        state["Index01B"] = i
        break
    #print2("Index01A: %u" % state["Index01A"])
    #print2("Index01B: %u" % state["Index01B"])
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
          if sizeTypeKey == "Range":
            psr = ProblemSizeRange(problemType, dictionary[sizeTypeKey])
            self.ranges.append( psr )
          elif sizeTypeKey == "Exact":
            e = dictionary[sizeTypeKey]
            if len(e) == problemType["TotalIndices"]:
              if problemType["OperationType"] == "GEMM":
                e += [-1, -1, -1, -1]
              self.exacts.append(tuple(e))
            elif len(e) == (problemType["TotalIndices"] + problemType["NumIndicesLD"]):
              self.exacts.append(tuple(e))
            else:
              printExit("ExactSize %s doesn't match indices of ProblemType %s" \
                  % (e, problemType) )

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
          [self.convertLeadingDims(problemSize) for problemSize in self.ranges[i].problemSizes]
      self.exacts[:] = [self.convertLeadingDims(problemSize) for problemSize in self.exacts]

    self.sizes = set()
    for sizeRange in self.ranges:
      self.sizes.update(sizeRange.problemSizes)
    self.sizes.update(self.exacts)
    self.sizes = sorted( list( self.sizes ) )
    self.totalProblemSizes = len(self.sizes)

    # max sizes
    self.maxD = 0
    self.maxC = 0
    self.maxA = 0
    self.maxB = 0
    for problemSize in self.sizes:
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


  def convertLeadingDims(self, problemSize):
    return problemSize[:self.problemType["NumIndicesC"]+1] + \
           (max(problemSize[0], problemSize[self.problemType["IndexAssignmentsLD"][0]]),) + \
           (max(problemSize[0], problemSize[self.problemType["IndexAssignmentsLD"][1]]),) + \
           (max(problemSize[self.problemType["IndexAssignmentsLD"][2]],
                problemSize[self.problemType["IndexAssignmentsA"][0]]),) + \
           (max(problemSize[self.problemType["IndexAssignmentsLD"][3]],
                problemSize[self.problemType["IndexAssignmentsB"][0]]),)

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
      else:
        self._state['ISA'] = list(globalParameters["CurrentISA"])

    # assign parameters without defaults
    for key in config:
      if key != "ProblemType" and key not in self._state:
        self._state[key] = config[key]
    self["Valid"] = True
    self["AssignedProblemIndependentDerivedParameters"] = False
    self["AssignedDerivedParameters"] = False
    Solution.assignDerivedParameters(self._state)
    self._name = None

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
      kernel["ProblemType"] = {}
      kernel["ProblemType"]["UseBeta"] = beta
      kernel["ProblemType"]["DataType"] = problemType["DataType"]
      kernel["ProblemType"]["DestDataType"] = problemType["DestDataType"]
      kernel["ProblemType"]["ComputeDataType"] = problemType["ComputeDataType"]
      kernel["ProblemType"]["Index0"] = problemType["Index0"]
      kernel["ProblemType"]["Index1"] = problemType["Index1"]
      kernel["ProblemType"]["UseInitialStridesAB"] = \
          problemType["UseInitialStridesAB"]
      kernel["ProblemType"]["UseInitialStridesCD"] = \
          problemType["UseInitialStridesCD"]
      kernel["ProblemType"]["SetConstStrideA"] = \
          problemType["SetConstStrideA"]
      kernel["ProblemType"]["SetConstStrideB"] = \
          problemType["SetConstStrideB"]
      kernel["ProblemType"]["ZeroPadA"] = \
          problemType["ZeroPadA"]
      kernel["ProblemType"]["ZeroPadB"] = \
          problemType["ZeroPadB"]
      kernel["ProblemType"]["ConvolutionConfig"] = \
          problemType["ConvolutionConfig"]
      kernel["ProblemType"]["NumIndicesC"] = problemType["NumIndicesC"]
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

    state["SubGroup0"] = state["WorkGroup"][0]
    state["SubGroup1"] = state["WorkGroup"][1]
    state["LocalSplitU"] = state["WorkGroup"][2]
    state["NumThreads"] = state["SubGroup0"] * state["SubGroup1"] * state["LocalSplitU"]

    state["ThreadTile0"] = state["ThreadTile"][0]
    state["ThreadTile1"] = state["ThreadTile"][1]

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
    minGrvw = 2 if globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"] else 1
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
      #print "reject fractional - no acceptable tile dim? GlobalReadVectorWidth", \
      # state["GlobalReadVectorWidth"]
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
      print("  PerLoadTile=%ux%u elements Loads/WI=%ux%u LoadTile/WI=%ux%u (MT=%ux%u), %u/%u = %.1f%% WI GRO used" \
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


  ########################################
  # assign all derived parameters
  @staticmethod
  def assignDerivedParameters(state):
    Solution.assignProblemIndependentDerivedParameters(state)

    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    ProblemType.assignDerivedParameters(state["ProblemType"])
    if not state["Valid"]:
      print1("in assignDerivedParameters, state['Valid'] = False")
      return

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

    # Determine which indices will be packed together as this impacts several different parms (sizes, magic numbers, etc)
    # The order in PackedC*Indices also determines the order that dimensions are packed - the first elements in
    # the list are the fastest-moving elements.
    # grid size [0,1]
    problemType = state["ProblemType"]
    state["PackedC0IdxChars"] = []
    state["PackedC0IndicesX"] = []
    indexChars = globalParameters["IndexChars"]
    # Pack all the dimensions (batch and free) of A into grid[0]
    assert(isPackedIndex(state, problemType["Index0"], 0x1))
    assert(isPackedIndex(state, problemType["Index1"], 0x2))

    if state["PackBatchDims"]==1:
        for bi in problemType["IndicesBatch"]:
            found = False
            for sc in problemType["SetConstStrideB"]:
                if sc[0]==bi and sc[1]==0:
                    found = True
            if not found:
                print ("Warning: batch index [%s,0] should be in SetConstStrideB"%bi)
                #problemType["SetConstStrideB"].append([bi,0])
    if state["PackBatchDims"]==2:
        for bi in problemType["IndicesBatch"]:
            found = False
            for sc in problemType["SetConstStrideA"]:
                if sc[0]==bi and sc[1]==0:
                    found = True
            if not found:
                print ("Warning: batch index [%s,0] should be in SetConstStrideA"%bi)
                #problemType["SetConstStrideA"].append([bi,0])

    for idx in problemType["IndexAssignmentsA"]:
      if isPackedIndex(state, idx, 0x1):
        assert (idx < problemType["NumIndicesC"])
        state["PackedC0IdxChars"].append("%s" % indexChars[idx])
        state["PackedC0IndicesX"].append(idx)

    state["PackedC1IdxChars"] = []
    state["PackedC1IndicesX"] = []
    # Pack all the dimensions (batch and free) of A into grid[0]
    for idx in problemType["IndexAssignmentsB"]:
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
    #print "staggerStrideShift=", staggerStrideShift, "depthu=", state["DepthU"]
    state["_staggerStrideShift"] = staggerStrideShift
    if state["StaggerU"] == 0:
      state["StaggerUMapping"] = 0

    # VectorWidth default handling
    if state["VectorWidth"] < 1:
      state["VectorWidth"] = int(4 / state["ProblemType"]["DataType"].numRegisters())
      while state["ThreadTile0"] % state["VectorWidth"] != 0 \
          or state["ThreadTile1"] % state["VectorWidth"] != 0:
        state["VectorWidth"] //= 2
    # TT0,1 both must be multiples of VW, b/c of rC, rA, rB
    if state["ThreadTile0"] % state["VectorWidth"] != 0 \
        or state["ThreadTile1"] % state["VectorWidth"] != 0:
      reject(state, "ThreadTile0 %u or ThreadTile1 %u not a multiple of VectorWidth %u" \
          % (state["ThreadTile0"], state["ThreadTile1"], \
          state["VectorWidth"]))
      return

    if state["PackSummationDims"] == 1:
        if state["DepthU"] % state["AssertSummationElementMultiple"] != 0:
          reject(state, "PackSummationDims=1 requires DepthU is integer multiple of ASEM")
        else:
          state["AssertSummationElementMultiple"] = state["DepthU"]

    # Some restrictions for half:
    if state["KernelLanguage"] == "Assembly" \
       and state["ProblemType"]["DataType"].isHalf():

       # Vector-width must be at least 2 for Half (since unroll loop uses packed operations?)
       if state["VectorWidth"] < 2:
         reject(state, "VectorWidth must be >= 2 for half")
       if globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"]:
         if (state["AssertSummationElementMultiple"] % 2 != 0 or \
             state["AssertFree0ElementMultiple"] % 2 != 0):
           # tail loop has ASEM requirement and beta-on-edge has AF0EM requirement
            reject(state, "Archs with HasEccHalf require ASEM%2==0 and AF0EM%2==0")

    if state["KernelLanguage"] == "Assembly" and state["PackSummationDims"]:
        reject(state, "PackSummationDims does not yet support assembly")

    # Default GlobalReadVectorWidth
    if state["GlobalReadVectorWidth"] == -1:
      state["GlobalReadVectorWidth"] = state["VectorWidth"]


    if not state["BufferLoad"] or state["KernelLanguage"] != "Assembly":
      state["BufferLoad"] = False
      state["DirectToLds"] = False
      state["UseSgprForGRO"] = False
      state["FractionalLoad"] = False

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

      if validDepthU and state["KernelLanguage"] == "Assembly" and state["ProblemType"]["DataType"].isHalf():
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

    ldsAlign = int(64 / state["ProblemType"]["DataType"].numRegisters())
    ldsNumElementsA = state["DepthU"]*(state["MacroTile0"]+state["LdsPadA"])
    ldsNumElementsAlignedA = roundUpToNearestMultiple(ldsNumElementsA,ldsAlign)
    ldsNumElementsB = state["DepthU"]*(state["MacroTile1"]+state["LdsPadB"])
    ldsNumElementsAlignedB = roundUpToNearestMultiple(ldsNumElementsB,ldsAlign)
    # import pdb
    # pdb.set_trace()
    # todo, can the alignment be a power of 2?
    state["LdsOffsetA"] = 0
    if state["PrefetchGlobalRead"]:
      state["LdsNumElementsAlignedA"] = ldsNumElementsAlignedA
      state["LdsNumElementsAlignedB"] = ldsNumElementsAlignedB
      state["LdsOffsetB"] = state["LdsOffsetA"] \
        + state["LdsNumElementsAlignedA"]

      offsetBlk = state["LdsOffsetB"] + ldsNumElementsAlignedB
      offsetBlk = int(2**(math.ceil(math.log(offsetBlk, 2))))

      state["LdsOffsetA_Blk"] = offsetBlk
      state["LdsOffsetB_Blk"] = state["LdsOffsetA_Blk"] \
        + state["LdsNumElementsAlignedA"]
      ldsNumElementsAB = state["LdsOffsetB_Blk"]+ ldsNumElementsB
    else:
      state["LdsOffsetB"] = ldsNumElementsAlignedA
      ldsNumElementsAB = ldsNumElementsAlignedA + ldsNumElementsB

    # lds buffer size for reduction
    ldsNumElementsReduction = state["LocalSplitU"]*state["MacroTile0"]*state["MacroTile1"] if state["LocalSplitU"] > 1 else 0

    # lds max occupancy
    ldsSizeOccupancy = globalParameters["DeviceLDS"] // state["MaxOccupancy"]
    ldsNumElementsOccupancy = ldsSizeOccupancy // state["ProblemType"]["DataType"].numBytes()

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

    # LoopUnroll too small
    if state["LoopUnroll"] < 2:
      reject(state, "LoopUnroll %u is less than 2" \
          % (state["LoopUnroll"]))


    # Determine if we can load directly-to-LDS.
    # Transpose requires a trip through registers to perform the transpose so can't use DirectToLdsA
    # LDS loads always write 4 bytes apart so can use only 4-byte operations
    #   TODO - for doubles we need to add something special here?
    # The matrix must not require transposing since that is done by reading to VGPR and writing in different order
    # The LSC (load size coalesced) must load some multiple of 256 bytes since that is what each DirectToLds load provides
    # Note for these matrices LSC is same as MacroTile dim
    if state["DirectToLds"]:
      # The tail loop requires half summation elements be a multiple of two to use DirectToLds feature
      elementMultipleOk = not state["ProblemType"]["DataType"].isHalf() \
                          or state["AssertSummationElementMultiple"] % 2 == 0

      wavefronts = state["NumThreads"] // globalParameters["WavefrontWidth"]
      numBytes = state["ProblemType"]["DataType"].numBytes()

      # DirectToLds loads return 256 bytes/wave
      # If fractional, ensure we are using all of the bytes that will be delivered

      if elementMultipleOk \
        and state["NumThreads"] % globalParameters["WavefrontWidth"] == 0:

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
    if state["BufferLoad"] and state["UseSgprForGRO"] and state["EdgeType"]=="ShiftPtr":
      if not state["GuaranteeNoPartialA"] or not state["GuaranteeNoPartialB"]:
        state["UseSgprForGRO"] = False
        #reject(state, "PBC with wide load has insufficient overlap guarantees- try GRVW=1 or adding appropriate Assert*ElementMultiple")

    if not state["BufferLoad"] or not state["GuaranteeNoPartialA"]:
      # Restrict GRVW/VW combos so shift-ptr logic will work
      if state["GlobalLoadVectorWidthA"] > 1 \
          and state["GlobalLoadVectorWidthA"] != state["VectorWidth"]:
          reject(state, "GlobalLoadVectorWidthA %u must be == VectorWidth %u or == 1" % \
                  (state["GlobalLoadVectorWidthA"], state["VectorWidth"]))

    if not state["BufferLoad"] or not state["GuaranteeNoPartialB"]:
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
    if not state["BufferLoad"]:
      state["UseSgprForGRO"] = 0
      if state["FractionalLoad"]:
        reject(state, "Fractional requires BufferLoad")

    if state["UseSgprForGRO"] == -1:
      # Don't use SGPR if it looks like we might not have enough - better to leave PBC enabled even if we have to use VGPR
      # 40 is based on current SGPR usage, this may need to be tuned in the future:
      numLoadsA = state["NumLoadsCoalescedA"]*state["NumLoadsPerpendicularA"]
      numLoadsB = state["NumLoadsCoalescedB"]*state["NumLoadsPerpendicularB"]
      if numLoadsA + numLoadsB > 35:
        #print "info: Disabling UseSgprForGRO since predicting too many SGPR will be used"
        state["UseSgprForGRO"] = 0
      else:
        state["UseSgprForGRO"] = 1


    if packedC0 and not state["GuaranteeNoPartialA"]:
      reject(state, "packedC0 requires GuaranteeNoPartialA")
    if packedC1 and not state["GuaranteeNoPartialB"]:
      reject(state, "packedC1 requires GuaranteeNoPartialB")

    if packedC0 or packedC1:

      state["UseSgprForGRO"] = 0

      if state["EdgeType"] != "ShiftPtr":
        reject(state, "Packed dims requires EdgeType==ShiftPtr")
      if state["KernelLanguage"] == "Assembly":
        if not state["BufferLoad"]:
          reject(state, "Packed dims for Assembly requires BufferLoad")
        if not state["LdcEqualsLdd"]:
          # this would require an extra VGPR for addressing (since shared VGPRS are per-row)
          # and also would require that the dimension extraction and scale code be implemented
          # for LDD as well. see emitExtractAndScalePackedDims
          reject(state, "Packed dims for Assembly requires LdcEqualsLdd==True")

    if packedC0 and state["VectorStore"] and state["PackGranularity"]==2 \
        and state["AssertFree0ElementMultiple"]<state["VectorWidth"]:
          reject(state, "packedC0 requires AF0EM>VectorWidth (for stores)")
    if packedC1 and state["VectorStore"] and state["PackGranularity"]==2 \
        and state["AssertFree1ElementMultiple"]<state["VectorWidth"]:
          # Not sure if this is actually required??
          reject(state, "packedC1 requires AF1EM>VectorWidth (for stores)")

    # current requirement to avoid buffer loads that span multiple entries
    # if the summation dim participating in the ZeroPad is not fast-moving then
    # likely have more performant options.
    for tc in ('A', 'B'):
      if problemType["ZeroPad%s"%tc] and state["KernelLanguage"] == "Assembly":
        if state["GlobalLoadVectorWidth%s"%tc] != 1:
          reject(state, "asm ZeroPad requires GlobalLoadVectorWidth==1")
        if not state["BufferLoad"]:
          reject(state, "asm ZeroPad requires BufferLoad")

    # avoid bug somehow related to GlobalSplitU + Persistent
    # avoid bug related to WGM<0
    # avoid bug somehow related to HPA + Persistent
    if state["PersistentKernel"] and (\
            (state["KernelLanguage"] == "Assembly" and state["GlobalSplitU"] != 1) or \
            (state["KernelLanguage"] == "Assembly" and state["WorkGroupMapping"] < 0) or \
            (state["KernelLanguage"] == "Assembly" and problemType["HighPrecisionAccumulate"]) ):
      state["PersistentKernel"] = 0

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
    requiredParameters["ProblemType"] = False # always prepended
    requiredParameters["MacroTile0"] = False # always prepended
    requiredParameters["MacroTile1"] = False # always prepended
    requiredParameters["DepthU"] = False # always prepended
    requiredParameters["LdcEqualsLdd"] = False # always prepended
    requiredParameters["Kernel"] = True # distinguish kernels from solutions
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
    if "LdcEqualsLdd" in state:
      if state["LdcEqualsLdd"]:
        name += "SE_"
      else:
        name += "SN_"
    for key in sorted(state.keys()):
      if key in requiredParameters:
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

