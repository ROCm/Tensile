################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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


from SolutionStructs import DataType
from Common import globalParameters, printExit
from KernelWriter import KernelWriter

################################################################################
# Make OpenCL Kernel String
################################################################################
class KernelWriterSource(KernelWriter):

  ##############################################################################
  # Make OpenCL Kernel String
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    super(KernelWriterSource, self).__init__( \
        kernelMinNaming, kernelSerialNaming)

    if self.language == "OCL":
      # everything escaped extra b/c string
      self.endLine = "\\n\"\n\""
      self.endLinePP = "\\\\" + self.endLine
    else:
      self.endLine = "\n"
      self.endLinePP =  "\\" + self.endLine

    if self.language == "OCL":
      self.getGroupIdStr = "get_group_id"
      self.getNumGroupsStr = "get_num_groups"
      self.getLocalIdStr = "get_local_id"
      self.getGlobalIdStr = "get_global_id"
      self.sharedDeclStr = "__local "
      self.sharedPtrStr = "__local "
      self.globalPtrStr = "__global "
      self.syncStr = "barrier(CLK_LOCAL_MEM_FENCE);"
      self.fenceStr = "mem_fence(CLK_LOCAL_MEM_FENCE);"
      self.macFStr = "mad"
      self.macDStr = "mad"
      self.int64Str = "long"
      self.uint64Str = "unsigned long"
      self.vectorComponents = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
      self.atomicCasStr = "atomic_cmpxchg"
      self.volatileStr = "volatile "
      self.deviceFunctionStr = ""
    else:
      self.getGroupIdStr = "hc_get_group_id"
      self.getNumGroupsStr = "hc_get_num_groups"
      self.getLocalIdStr = "hc_get_workitem_id"
      self.getGlobalIdStr = "hc_get_workitem_absolute_id"
      self.sharedDeclStr = "__shared__ "
      self.sharedPtrStr = ""
      self.globalPtrStr = ""
      self.syncStr = "__syncthreads();"
      self.fenceStr = self.syncStr
      self.macFStr = "fmaf"
      self.macDStr = "fma"
      self.int64Str = "int64_t"
      self.uint64Str = "uint64_t"
      self.vectorComponents = ["x", "y", "z", "w"]
      self.atomicCasStr = "atomicCAS"
      self.volatileStr = ""
      self.deviceFunctionStr = "__device__ "

    self.commentPrefix = "/*"
    self.commentSuffix = "*/"
    self.commentHR = "*"*40
    self.indent = "  "


  ##############################################################################
  #
  #   Functions to Write Kernel Segments
  #
  ##############################################################################

  ##############################################################################
  # Open String
  ##############################################################################
  def openString(self, kernel):
    kStr = ""
    if self.language == "OCL":
      kernelName = self.getKernelName(kernel)
      kStr += "\n"
      kStr += "std::string %s_src_%u = \"" % (kernelName, self.stringIdx)
    return kStr

  ##############################################################################
  # Close String
  ##############################################################################
  def closeString(self, kernel):
    kStr = ""
    if self.language == "OCL":
      kStr += "\";\n"
      self.stringIdx += 1
    return kStr

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def initKernel(self, kernel):
    super(KernelWriterSource, self).initKernel( kernel )
    pass

  ##############################################################################
  # Function Prefix
  ##############################################################################
  def functionPrefix(self, kernel):
    kStr = ""
    if kernel["ProblemType"]["DataType"].isHalf():
      if self.language == "OCL":
        self.vectorComponents = ["p[0]", "p[1]"]
      else:
        self.vectorComponents = ["p[0]", "p[1]"]

    ####################################
    # kernel preprocessor definitions
    kStr += self.endLine
    kStr += "/* tile parameters */" + self.endLine
    kStr += "#define NUM_THREADS %3d%s" \
        % (kernel["NumThreads"], self.endLine )
    kStr += "#define SG%s %d%s" \
        % (self.tileChar0, kernel["SubGroup0"], self.endLine )
    kStr += "#define SG%s %d%s" \
        % (self.tileChar1, kernel["SubGroup1"], self.endLine )
    kStr += "#define TT%s %d%s" \
        % (self.tileChar0, kernel["ThreadTile0"], self.endLine )
    kStr += "#define TT%s %d%s" \
        % (self.tileChar1, kernel["ThreadTile1"], self.endLine )
    kStr += "#define MT%s (SG%s*TT%s)%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine )
    kStr += "#define MT%s (SG%s*TT%s)%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine )
    kStr += self.endLine
    kStr += "/* DepthU parameters*/%s" % self.endLine
    kStr += "#define CPSV (NUM_THREADS / MT%s * VECTOR_WIDTH)%s" \
        % (self.tileChar0, self.endLine)
    kStr += "#define LOCAL_SPLITU %d%s" \
        % (kernel["LocalSplitU"], self.endLine )
    kStr += "#define UNROLL %d%s" \
        % (kernel["LoopUnroll"], self.endLine )
    kStr += "#define LOCAL_DEPTHU (LOCAL_SPLITU*UNROLL)%s" % (self.endLine )
    if kernel["GlobalSplitU"] > 1:
      kStr += "#define GLOBAL_SPLITU %u%s" \
          % (kernel["GlobalSplitU"], self.endLine )
    kStr += self.endLine
    kStr += "/* other */%s" % self.endLine
    kStr += "#define PAD %u%s" % (kernel["LdsPad"], self.endLine)
    kStr += "#define WORK_GROUP_MAPPING %u%s" \
        % (abs(kernel["WorkGroupMapping"]), self.endLine)
    kStr += "#define VECTOR_WIDTH %u%s" % (kernel["VectorWidth"], self.endLine)
    kStr += self.endLine

    ####################################
    # num loads
    kStr += "/* num loads parallel and perpendicular to coalesced */%s" \
        % self.endLine
    kStr += "#define NLCA %d%s" % (kernel["NumLoadsCoalescedA"], self.endLine )
    kStr += "#define NLCB %d%s" % (kernel["NumLoadsCoalescedB"], \
        self.endLine )

    kStr += "#define NLPA %d%s" % (kernel["NumLoadsPerpendicularA"], \
        self.endLine )
    kStr += "#define NLPB %d%s" % (kernel["NumLoadsPerpendicularB"], \
        self.endLine )
    kStr += self.endLine

    ####################################
    # load sizes
    kStr += "/* load sizes parallel and perpendicular to coalesced */%s" % self.endLine
    if kernel["ProblemType"]["TLUA"]:
      kStr += "#define LSCA (MT%s/NLCA)%s" \
          % (self.tileCharA, self.endLine)
      kStr += "#define LSPA (LOCAL_DEPTHU/NLPA)" + self.endLine
    else:
      kStr += "#define LSCA (LOCAL_DEPTHU/NLCA)%s" \
          % (self.endLine)
      kStr += "#define LSPA (MT%s/NLPA)%s" \
          % ( self.tileCharA, self.endLine)
    if kernel["ProblemType"]["TLUB"]:
      kStr += "#define LSCB (MT%s/NLCB)%s" \
          % (self.tileCharB, self.endLine)
      kStr += "#define LSPB (LOCAL_DEPTHU/NLPB)" + self.endLine
    else:
      kStr += "#define LSCB (LOCAL_DEPTHU/NLCB)%s" \
          % (self.endLine)
      kStr += "#define LSPB (MT%s/NLPB)%s" % (self.tileCharB, self.endLine)
    kStr += "#define LVCA (LSCA/VECTOR_WIDTH)%s" % (self.endLine)
    kStr += "#define LVCB (LSCB/VECTOR_WIDTH)%s" % (self.endLine)
    kStr += "#define LVPA (LSPA/VECTOR_WIDTH)%s" % (self.endLine)
    kStr += "#define LVPB (LSPB/VECTOR_WIDTH)%s" % (self.endLine)


    # local buffer size
    kStr += "#define LDS_OFFSET_B %u%s" % (kernel["LdsOffsetB"], self.endLine)
    kStr += "#define LDS_NUM_ELEMENTS %u%s" % (kernel["LdsNumElements"], \
        self.endLine)

    # prefetch local buffer offsets
    # layout is redA, redB, blkA, blkB
    if kernel["PrefetchGlobalRead"]:
      kStr += "#define LDS_OFFSET_BLK %u%s" \
         % (kernel["LdsOffsetA_Blk"], self.endLine)

    ########################################
    # z-ordering
    if True:
      kStr += self.endLine
      kStr += "#ifndef Z_ORDER_FUNCTIONS%s" % self.endLine
      kStr += "#define Z_ORDER_FUNCTIONS%s" % self.endLine
      kStr += "%svoid z_order(%s" % (self.deviceFunctionStr, self.endLine)
      kStr += "    unsigned int *z0, // 16-bits output%s" % self.endLine
      kStr += "    unsigned int *z1, // 16-bits output%s" % self.endLine
      kStr += "    unsigned int serial ) { // 32-bits input%s" % self.endLine
      kStr += "  *z0 = serial;%s" % (self.endLine)
      kStr += "  *z1 = (serial >> 1);%s" % (self.endLine)
      kStr += "  *z0 &= 0x55555555;%s"  % (self.endLine)
      kStr += "  *z1 &= 0x55555555;%s"  % (self.endLine)
      kStr += "  *z0 |= ( (*z0) >> 1 );%s" % (self.endLine)
      kStr += "  *z1 |= ( (*z1) >> 1 );%s" % (self.endLine)
      kStr += "  *z0 &= 0x33333333;%s"  % (self.endLine)
      kStr += "  *z1 &= 0x33333333;%s"  % (self.endLine)
      kStr += "  *z0 |= ( (*z0) >> 2 );%s" % (self.endLine)
      kStr += "  *z1 |= ( (*z1) >> 2 );%s" % (self.endLine)
      kStr += "  *z0 &= 0x0f0f0f0f; %s" % (self.endLine)
      kStr += "  *z1 &= 0x0f0f0f0f;%s"  % (self.endLine)
      kStr += "  *z0 |= ( (*z0) >> 4 );%s" % (self.endLine)
      kStr += "  *z1 |= ( (*z1) >> 4 );%s" % (self.endLine)
      kStr += "  *z0 &= 0x00ff00ff;%s"  % (self.endLine)
      kStr += "  *z1 &= 0x00ff00ff;%s"  % (self.endLine)
      kStr += "  *z0 |= ( (*z0) >> 8 );%s" % (self.endLine)
      kStr += "  *z1 |= ( (*z1) >> 8 );%s" % (self.endLine)
      kStr += "  *z0 &= 0x0000ffff;%s"  % (self.endLine)
      kStr += "  *z1 &= 0x0000ffff;%s"  % (self.endLine)
      kStr += "}%s" % self.endLine
      kStr += self.endLine
      kStr += "%sunsigned int round_down_power_of_2( unsigned int d0, unsigned int d1) {%s" % (self.deviceFunctionStr, self.endLine)
      kStr += "  unsigned int pow2 = min(d0, d1);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 1);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 2);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 4);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 8);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 16);%s" % self.endLine
      kStr += "  pow2 = pow2 - (pow2 >> 1);%s" % self.endLine
      kStr += "  return pow2;%s" % self.endLine
      kStr += "}%s" % self.endLine
      kStr += self.endLine
      kStr += "%svoid generalized_z_order(%s" % (self.deviceFunctionStr, self.endLine)
      kStr += "    unsigned int *z0,%s" % self.endLine
      kStr += "    unsigned int *z1,%s" % self.endLine
      kStr += "    unsigned int d0,%s" % self.endLine
      kStr += "    unsigned int d1,%s" % self.endLine
      kStr += "    unsigned int maxPow2,%s" % self.endLine
      kStr += "    unsigned int max0,%s" % self.endLine
      kStr += "    unsigned int max1 ) {%s" % self.endLine
      kStr += "  if (! maxPow2) maxPow2 = round_down_power_of_2( max0, max1 );%s" % self.endLine
      kStr += "  // determine which tile wg is in and relative coord in tile%s" % self.endLine
      kStr += "  unsigned int offset0 = 0; // coord of tile%s" % self.endLine
      kStr += "  unsigned int offset1 = 0; // coord of tile%s" % self.endLine
      kStr += "  unsigned int start0 = 0;%s" % self.endLine
      kStr += "  unsigned int start1 = 0;%s" % self.endLine
      kStr += "  unsigned int tile = maxPow2;%s" % self.endLine
      kStr += "  unsigned int tilem1 = tile - 1;%s" % self.endLine
      kStr += "  for ( unsigned int i = 0; i < 16; i++ ) {%s" % self.endLine
      kStr += "    start0 = d0 & ~tilem1; // (d0 / tile) * tile;%s" % self.endLine
      kStr += "    start1 = d1 & ~tilem1; // (d1 / tile) * tile;%s" % self.endLine
      kStr += "    offset0 |= start0; // +=%s" % self.endLine
      kStr += "    offset1 |= start1;%s" % self.endLine
      kStr += "    d0 &= ~start0; // -=%s" % self.endLine
      kStr += "    d1 &= ~start1;%s" % self.endLine
      kStr += "    unsigned int end0 = start0 + tile; // cant be | b/c evals to 0+4->4 or 4+4->8%s" % self.endLine
      kStr += "    unsigned int end1 = start1 + tile;%s" % self.endLine
      kStr += "    if ( end0 <= max0 && end1 <= max1 ) break; // both end and max can be non-pow2%s" % self.endLine
      kStr += "    max0 -= start0; // cant be &~ b/c max0 doesnt necessarily have multiple of start0 to turn off%s" % self.endLine
      kStr += "    max1 -= start1;%s" % self.endLine
      kStr += "    tile >>= 1;%s" % self.endLine
      kStr += "    tilem1 >>= 1;%s" % self.endLine
      kStr += "  }%s" % self.endLine
      kStr += "  // d0, d1 is relative coord within tile%s" % self.endLine
      kStr += self.endLine
      kStr += "  // z-order relative coord%s" % self.endLine
      kStr += "  unsigned int serial = d0 + d1 * tile;%s" % self.endLine
      kStr += "  z_order( z0, z1, serial );%s" % self.endLine
      kStr += "  // add tile offset onto z-ordered index%s" % self.endLine
      kStr += "  *z0 |= offset0;%s" % self.endLine
      kStr += "  *z1 |= offset1;%s" % self.endLine
      #kStr += "  if (get_local_id(0)==0) printf(\\\"%%u, %%u -> %%u, %%u\\\\n\\\", d0, d1, (*z0), (*z1));%s" % self.endLine
      kStr += "}%s" % self.endLine
      kStr += "#endif%s" % self.endLine

    ####################################
    # global memory indices
    kStr += self.endLine
    kStr += "/* global memory indices */" + self.endLine
    # C
    kStr += "#define GLOBAL_C(IDX%s" % self.indexChars[0]
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine
    # A non-vector
    kStr += "#define GLOBAL_OFFSET_A(IDX%s" \
        % self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      kStr += ", IDX%s" \
          % self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
    indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    kStr += ") (( (IDX%s)*strideA%s" % (indexChar, indexChar)
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
      kStr += " + (IDX%s)*strideA%s" % (indexChar, indexChar)
    kStr += " ))%s" % self.endLine
    # B non-vector
    kStr += "#define GLOBAL_OFFSET_B(IDX%s" \
        % self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      kStr += ", IDX%s" \
          % self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
    indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    kStr += ") (( (IDX%s)*strideB%s" % (indexChar, indexChar)
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
      kStr += " + (IDX%s)*strideB%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine
    kStr += self.endLine

    ####################################
    # data types
    kStr += "/* data types */" + self.endLine
    kStr += "#define DATA_TYPE %s%s" \
        % (kernel["ProblemType"]["DataType"].toDevice(self.language), \
        self.endLine)
    vecStr = kernel["ProblemType"]["DataType"].toDevice(self.language)
    if kernel["VectorWidth"] > 1:
      vecStr += str(kernel["VectorWidth"])
    kStr += "#define VECTOR_TYPE %s%s" % (vecStr, self.endLine)

    if self.language == "OCL":
      kStr += "#define MAC(A,B,DST) mad(A,B,DST)"
    else:
      # TODO convert to always using hfma2 since we always have even number of fma's, half2*=rC
      if kernel["ProblemType"]["DataType"].isHalf():
        kStr += "#define MAC(A,B,DST) DST = __hfma(A,B,DST)"
      else:
        kStr += "#define MAC(A,B,DST) DST += A*B"
    kStr += self.endLine

    if self.language == "HIP" and kernel["ProblemType"]["DataType"].isComplex():
      kStr += "#define s0 x" + self.endLine
      kStr += "#define s1 y" + self.endLine

    ####################################
    # Atomic Global MAC
    if kernel["GlobalSplitU"] > 1:
      kStr += self.comment("atomic add float")
      kStr += "#ifndef ATOMIC_FLOAT_FUNCTION%s" % (self.endLine)
      kStr += "#define ATOMIC_FLOAT_FUNCTION%s" % (self.endLine)
      if self.language == "OCL":
        """
        kStr += self.endLine
        kStr += "void atomicAddType(%s%sfloat *fPtr, float operand) {%s" \
            % (self.volatileStr, self.globalPtrStr, self.endLine)
        kStr += "  volatile atomic_float *aPtr = (atomic_float*)(fPtr);%s" % (self.endLine)
        kStr += "  float oldValue, newValue;%s" % (self.endLine)
        kStr += "  oldValue = atomic_load_explicit(aPtr, memory_order_relaxed, memory_scope_device);%s" % (self.endLine)
        #kStr += "  oldValue = atomic_load(aPtr);%s" % (self.endLine)
        kStr += "  do {%s" % (self.endLine)
        kStr += "    newValue = oldValue + operand;%s" % (self.endLine)
        #kStr += "    prevReturn = %s(uPtr, prevVal.ui, newVal.ui);%s" \
        #    % (self.atomicCasStr, self.endLine)
        kStr += "  } while ( !atomic_compare_exchange_weak_explicit(aPtr, &oldValue, newValue, memory_order_relaxed, memory_order_relaxed) );%s" % (self.endLine)
        #kStr += "  } while ( !atomic_compare_exchange_weak(aPtr, &oldValue, newValue) );%s" % (self.endLine)
        kStr += "}%s" % (self.endLine)
        """
        kStr += "typedef union {%s" % (self.endLine)
        kStr += "  unsigned int ui;%s" % (self.endLine)
        kStr += "  float f;%s" % (self.endLine)
        kStr += "} AtomicFloat;%s" % (self.endLine)
        kStr += self.endLine
        kStr += "%svoid atomicAddType(%s%sfloat *fPtr, float operand) {%s" \
            % ("__device__ " if self.language == "HIP" else "", \
            self.volatileStr, self.globalPtrStr, self.endLine)
        kStr += "  AtomicFloat newVal;%s" % (self.endLine)
        kStr += "  AtomicFloat prevVal;%s" % (self.endLine)
        kStr += "  %s%sunsigned int *uPtr = (%s%sunsigned int *)fPtr;%s" \
            % (self.volatileStr, self.globalPtrStr, self.volatileStr, \
            self.globalPtrStr, self.endLine)
        kStr += "  unsigned int prevReturn = *uPtr;%s" % (self.endLine)
        kStr += "  do {%s" % (self.endLine)
        kStr += "    prevVal.ui = prevReturn;%s" % (self.endLine)
        kStr += "    newVal.f = prevVal.f + operand;%s" % (self.endLine)
        kStr += "    prevReturn = %s(uPtr, prevVal.ui, newVal.ui);%s" \
            % (self.atomicCasStr, self.endLine)
        kStr += "  } while (prevVal.ui != prevReturn);%s" % (self.endLine)
        kStr += "}%s" % (self.endLine)
      else:
        """
        kStr += "%svoid atomicAddType(%s%sfloat *fPtr, float operand) {%s" \
            % ("__device__ " if self.language == "HIP" else "", \
            self.volatileStr, self.globalPtrStr, self.endLine)
        kStr += "  %s%sunsigned int *uPtr = (%s%sunsigned int *)fPtr;%s" \
            % (self.volatileStr, self.globalPtrStr, self.volatileStr, \
            self.globalPtrStr, self.endLine)
        #kStr += "  unsigned int old = *uPtr;%s" % (self.endLine)
        kStr += "  unsigned int old = atomicAdd(uPtr, 0); // atomic read%s" % (self.endLine)
        kStr += "  unsigned int assumed, newValue;%s" % (self.endLine)
        kStr += "  do {%s" % (self.endLine)
        kStr += "    assumed = old;%s" % (self.endLine)
        kStr += "    newValue = __float_as_uint(operand + __uint_as_float(assumed));%s" % (self.endLine)
        kStr += "    old = %s(uPtr, assumed, newValue);%s" \
            % (self.atomicCasStr, self.endLine)
        kStr += "  } while (assumed != old);%s" % (self.endLine)
        kStr += "}%s" % (self.endLine)
        """
        kStr += self.endLine
        kStr += "__device__ inline void atomicAddType(%s%sfloat *fPtr, float operand) {%s" \
            % (self.volatileStr, self.globalPtrStr, self.endLine)
        kStr += "  std::atomic<float> *aPtr = reinterpret_cast<std::atomic<float>*>(fPtr);%s" % (self.endLine)
        kStr += "  float oldValue, newValue;%s" % (self.endLine)
        kStr += "  oldValue = aPtr->load(std::memory_order_relaxed);%s" % (self.endLine)
        kStr += "  do {%s" % (self.endLine)
        kStr += "    newValue = oldValue + operand;%s" % (self.endLine)
        #kStr += "    prevReturn = %s(uPtr, prevVal.ui, newVal.ui);%s" \
        #    % (self.atomicCasStr, self.endLine)
        #kStr += "  } while ( !std::atomic_compare_exchange_weak_explicit(aPtr, &oldValue, newValue, std::memory_order_acq_rel, std::memory_order_release) );%s" % (self.endLine)
        kStr += "  } while ( !std::atomic_compare_exchange_weak_explicit(aPtr, &oldValue, newValue, std::memory_order_relaxed, std::memory_order_release) );%s" % (self.endLine)
        kStr += "}%s" % (self.endLine)

      kStr += "#endif%s" % self.endLine


    ####################################
    # MACs
    kStr += self.endLine
    kStr += "/* MAC's */" + self.endLine
    if kernel["ProblemType"]["DataType"].isReal():
      # real data
      kStr += "#define TYPE_MAC(MULA,MULB,DST) " \
          + "DST = MAC(MULA,MULB,DST);" + self.endLine
      # GSU
      if kernel["GlobalSplitU"] > 1: # 1st kernel will have taken care of B
        if kernel["ProblemType"]["UseBeta"]:
          kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG,BETA) atomicAddType(&(DST), (ALPHA)*(REG));"
        else:
          kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) atomicAddType(&(DST), (ALPHA)*(REG));"

      else:
        if kernel["ProblemType"]["UseBeta"]:
          # dst = alpha*reg + dst*beta
          kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG,BETA) " \
              + "DST = (ALPHA)*(REG) + (BETA)*(DST);" + self.endLine
        else:
          # dst = alpha*reg
          kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) " \
              + "DST = (ALPHA)*(REG);" + self.endLine
    else:
      # complex data
      if not kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # neither conjugate
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAC( -MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # A conjugate (negate imaginary A.s1)
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAC( -MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif not kernel["ProblemType"]["ComplexConjugateA"] and kernel["ProblemType"]["ComplexConjugateB"]:
        # B conjugate (negate imaginary B.s1)
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAC( -MULA.s1, -MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      else:
        # A & B conjugate (negate imaginary .s1)
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s1, -MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAC( -MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      if kernel["ProblemType"]["UseBeta"]:
        # dst = alpha*reg + beta*dst
        kStr += (
          "#define TYPE_MAC_WRITE( DST, ALPHA, REG, BETA ) "+self.endLinePP +
          "  /* (1) */ " + self.endLinePP +
          "  type_mac_tmp = REG.s0; " + self.endLinePP +
          "  REG.s0 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s0 = MAC( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLinePP +
          "  REG.s1 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s1 = MAC(  ALPHA.s1, type_mac_tmp, REG.s1 ); "+self.endLinePP+
          "  /* (2) */ " + self.endLinePP +
          "  REG.s0 = MAC(  BETA.s0, DST.s0, REG.s0 ); " + self.endLinePP +
          "  REG.s0 = MAC( -BETA.s1, DST.s1, REG.s0 ); " + self.endLinePP +
          "  REG.s1 = MAC(  BETA.s1, DST.s0, REG.s1 ); " + self.endLinePP +
          "  REG.s1 = MAC(  BETA.s0, DST.s1, REG.s1 ); " + self.endLinePP +
          "  /* (3) */ " + self.endLinePP +
          "  DST = REG;" + self.endLine )
      else:
        # dst = alpha*reg
        kStr += (
          "#define TYPE_MAC_WRITE( DST, ALPHA, REG ) "+self.endLinePP+
          "  /* (1) */ " + self.endLinePP +
          "  type_mac_tmp = REG.s0; " + self.endLinePP +
          "  REG.s0 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s0 = MAC( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLinePP +
          "  REG.s1 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s1 = MAC(  ALPHA.s1, type_mac_tmp, REG.s1 ); "+self.endLinePP+
          "  /* (3) */ " + self.endLinePP +
          "  DST = REG;" + self.endLine )

    ####################################
    # sumation unroll
    kStr += self.endLine
    kStr += "/* %dx%d micro-tile */%s" \
      % (kernel["ThreadTile0"], kernel["ThreadTile1"], self.endLine)
    numMacs = 2 if kernel["PrefetchLocalRead"] else 1

    for m in range(0, numMacs):
      kStr += "#define MAC_%ux%u" \
          % (kernel["ThreadTile0"], kernel["ThreadTile1"])
      if kernel["PrefetchLocalRead"]:
        kStr += ("" if m==0 else "_BLK")
      kStr += self.endLinePP

      """
    if False:
      if kernel["VectorWidth"] == 1:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0], rA[1], rA[2], rA[3], rA[4], rA[5], rA[6], rA[7], rB[0], rB[1], rB[2], rB[3], rB[4], rB[5], rB[6], rB[7]); %s" % (self.endLinePP)
      if kernel["VectorWidth"] == 2:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0].%s, rA[0].%s, rA[1].%s, rA[1].%s, rA[2].%s, rA[2].%s, rA[3].%s, rA[3].%s, rB[0].%s, rB[0].%s, rB[1].%s, rB[1].%s, rB[2].%s, rB[2].%s, rB[3].%s, rB[3].%s); %s" % ( \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.endLinePP)
      if kernel["VectorWidth"] == 4:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0].%s, rA[0].%s, rA[0].%s, rA[0].%s, rA[1].%s, rA[1].%s, rA[1].%s, rA[1].%s, rB[0].%s, rB[0].%s, rB[0].%s, rB[0].%s, rB[1].%s, rB[1].%s, rB[1].%s, rB[1].%s); %s" % ( \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.endLinePP)
      """

      for b in range(0, kernel["ThreadTileB"]):
        for a in range(0, kernel["ThreadTileA"]):
          # a
          vecA = a / kernel["VectorWidth"]
          elemA = a % kernel["VectorWidth"]
          strA = "rA[%d%s]" % (vecA, ("+TT%s/VECTOR_WIDTH"%self.tileCharA) \
              if m>0 else "")
          if kernel["VectorWidth"] > 1:
            strA += ".%s" % self.vectorComponents[elemA]
          # b
          vecB = b / kernel["VectorWidth"]
          elemB = b % kernel["VectorWidth"]
          strB = "rB[%d%s]" % (vecB, ("+TT%s/VECTOR_WIDTH"%self.tileCharB) \
              if m>0 else "")
          if kernel["VectorWidth"] > 1:
            strB += ".%s" % self.vectorComponents[elemB]
          # c
          strC = "rC[%d+%d*TT%s/VECTOR_WIDTH]" % (vecA, b, self.tileChar0 )
          elemC = elemA
          if kernel["VectorWidth"] > 1:
            strC += ".%s" % self.vectorComponents[elemC]
          """
          kStr += "  printf(\\\"T[%%u,%u,%u]: %s:%%.0f += %s:%%.0f * %s:%%.0f\\\\n\\\", serial, %s, %s, %s); %s" % (a, b, strC, strA, strB, strC, strA, strB, self.endLinePP)
          """
          kStr += "  TYPE_MAC(%s,%s,%s); %s" % (strA, strB, strC, \
              self.endLinePP)
      if kernel["UnrollMemFence"]:
        kStr += "  " + self.fenceStr
      kStr += self.endLine

    ####################################
    # preprocessor definitions of kernel arguments
    firstStride = 0
    if kernel["ProblemType"]["UseInitialStrides"]:
      # no strides #defined
      lastStrideC = 0
      lastStrideA = 0
      lastStrideB = 0
    else:
      # #define initial stride
      kStr += "/* hard-coded initial strides */%s" \
          % self.endLine
      lastStrideC = 1
      lastStrideA = 1
      lastStrideB = 1

    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideA):
      kStr += "#define strideA" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]] \
          + " 1" + self.endLine
    for i in range(firstStride, lastStrideB):
      kStr += "#define strideB" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]] \
          + " 1" + self.endLine
    return kStr


  ##############################################################################
  # Function Signature Prefix
  ##############################################################################
  def functionSignaturePrefix(self, kernel):
    s = ""
    if self.language == "HIP":
      s += "#pragma clang diagnostic push" + self.endLine
      s += "#pragma clang diagnostic ignored \"-Wunused-parameter\"" + self.endLine
    return s


  ##############################################################################
  # Function Signature
  ##############################################################################
  def functionSignature(self, kernel ):
    kernelName = self.getKernelName(kernel)

    # determine chars for fast access
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.indexChars[kernel["ProblemType"]["Index0"]]
    self.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.indexChars[kernel["ProblemType"]["Index1"]]
    self.tileChar0 = self.indexChars[kernel["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[kernel["ProblemType"]["Index1"]]

    s = ""
    # kernel name
    if self.language == "OCL":
      s += "__attribute__((reqd_work_group_size(NUM_THREADS,1,1)))"
      s += self.endLine
      s += "__kernel "
    else:
      s += "extern \"C\"\n"
      s += "__global__ "
    s += "void %s" % ( kernelName )
    s += "(" + self.endLine
    # pointers
    globalStr = "__global "
    if self.language == "HIP":
      s += "  hipLaunchParm lp," + self.endLine
      globalStr = ""
    restrictStr = "restrict"
    if self.language == "HIP":
      restrictStr = "__restrict__"
    ptrStr = kernel["ProblemType"]["DataType"].toDevice(self.language)
    s += "  " + globalStr + ptrStr \
        + " *C,"
    s += self.endLine
    s += "  " + globalStr + ptrStr \
        + " const * " + restrictStr + " A,"
    s += self.endLine
    s += "  " + globalStr + ptrStr \
        + " const * " + restrictStr + " B"

    # alpha & beta
    s += "," + self.endLine + "  " \
        + kernel["ProblemType"]["DataType"].toDevice(self.language) + " const alpha"
    if kernel["ProblemType"]["UseBeta"]:
      s += "," + self.endLine + "  " \
          + kernel["ProblemType"]["DataType"].toDevice(self.language) + " const beta"

    # offsets
    s += ( "," + self.endLine + "  unsigned int const offsetC,"
        + self.endLine +
        "  unsigned int const offsetA," + self.endLine +
        "  unsigned int const offsetB" )

    # strides
    firstStride = 1
    if kernel["ProblemType"]["UseInitialStrides"]:
      firstStride = 0
    lastStrideC = kernel["ProblemType"]["NumIndicesC"]
    lastStrideA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    lastStrideB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    for i in range(firstStride, lastStrideC):
      s += "," + self.endLine + "  unsigned int const strideC" + self.indexChars[i]
    for i in range(firstStride, lastStrideA):
      s += "," + self.endLine + "  unsigned int const strideA" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
    for i in range(firstStride, lastStrideB):
      s += "," + self.endLine + "  unsigned int const strideB" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]

    # sizes
    for i in range(0, kernel["ProblemType"]["TotalIndices"]):
      s += "," + self.endLine + "  unsigned int const size" + self.indexChars[i]
    s += " )"
    return s

  ##############################################################################
  # Function Signature Suffix
  ##############################################################################
  def functionSignatureSuffix(self, kernel):
    s = ""
    if self.language == "HIP":
      s += self.endLine
      s += "#pragma clang diagnostic pop" + self.endLine
    return s

  ##############################################################################
  # Function Begin
  ##############################################################################
  def functionBegin(self, kernel):
    s = ""
    s += " {" + self.endLine
    return s

  ##############################################################################
  # Allocate Resources
  ##############################################################################
  def allocateResources(self, kernel):
    kStr = ""
    kStr += self.endLine
    kStr += "  /* registers for MAC's */" + self.endLine
    kStr += "  VECTOR_TYPE rC[TT%s*TT%s/VECTOR_WIDTH];%s" \
        % (self.tileChar0, self.tileChar1, self.endLine )
    for i in range(0, kernel["ThreadTile0"]*kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for s in range(0, kernel["VectorWidth"]):
        kStr += "  rC[%u]%s = 0;%s" % (i, ((".%s"%self.vectorComponents[s]) \
            if kernel["VectorWidth"]>1 else ""), self.endLine)

    kStr += "  VECTOR_TYPE rA[TT%s/VECTOR_WIDTH%s];%s" \
        % (self.tileChar0, ("*2" if kernel["PrefetchLocalRead"] else ""), \
        self.endLine)
    kStr += "  VECTOR_TYPE rB[TT%s/VECTOR_WIDTH%s];%s" \
        % (self.tileChar1, ("*2" if kernel["PrefetchLocalRead"] else ""), \
        self.endLine)

    ####################################
    # registers for global -> local load
    kStr += self.endLine
    kStr += "  /* registers for global->local */" + self.endLine
    kStr += "  VECTOR_TYPE "
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        kStr += "a_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedA"]-1 \
            and perp == kernel["NumLoadsPerpendicularA"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "
    kStr += "  VECTOR_TYPE "
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        kStr += "b_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedB"]-1 \
            and perp == kernel["NumLoadsPerpendicularB"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "

    ####################################
    # allocate local memory
    kStr += self.endLine
    kStr += "  /* allocate local memory */" + self.endLine
    kStr += "  %sDATA_TYPE localMemory[LDS_NUM_ELEMENTS];%s" \
        % (self.sharedDeclStr, self.endLine )

    ####################################
    # zero
    if kernel["ProblemType"]["DataType"].isHalf() \
        and kernel["VectorWidth"] > 1 \
        and (kernel["LoopTail"] or kernel["EdgeType"] == "Branch"):
      #zeroString = kernel["ProblemType"][\
      #    "DataType"].zeroString(self.language, kernel["VectorWidth"])
      #zeroString = "0, 0"
      kStr += "  VECTOR_TYPE VECTOR_ZERO;%s" % ( self.endLine )
      kStr += "  VECTOR_ZERO.p[0] = 0;%s" % self.endLine
      kStr += "  VECTOR_ZERO.p[1] = 0;%s" % self.endLine
      kStr += "#define SCALAR_ZERO 0%s" % self.endLine
    elif kernel["ProblemType"]["DataType"].isComplex():
      kStr += "  DATA_TYPE SCALAR_ZERO;%s" % ( self.endLine )
      kStr += "  SCALAR_ZERO.s0 = 0;%s" % self.endLine
      kStr += "  SCALAR_ZERO.s1 = 0;%s" % self.endLine
      kStr += "#define VECTOR_ZERO SCALAR_ZERO%s" % self.endLine
    else:
      kStr += "#define VECTOR_ZERO %s%s" % ( kernel["ProblemType"][\
         "DataType"].zeroString(self.language, kernel["VectorWidth"]), \
         self.endLine )
      kStr += "#define SCALAR_ZERO %s%s" % ( kernel["ProblemType"][\
         "DataType"].zeroString(self.language, 1), \
         self.endLine )


    return kStr

  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  def graWorkGroup(self, kernel):
    kStr = ""
    wg0 = "wg%s" % self.tileChar0
    wg1 = "wg%s" % self.tileChar1
    kernel["TransposeWorkGroupGrid"] = kernel["WorkGroupMapping"] < 0
    nwgg = not kernel["TransposeWorkGroupGrid"] # normal work-group grid

    ########################################
    # Dimension WorkGroups
    if kernel["WorkGroupMappingType"] == "D":
      if kernel["WorkGroupMapping"] == 1:
        kStr += "  unsigned int %s = %s(%u);%s" \
            % ( wg0, self.getGroupIdStr, 0 if nwgg else 1, self.endLine)
        kStr += "  unsigned int %s = %s(%u);%s" \
            % ( wg1, self.getGroupIdStr, 1 if nwgg else 0, self.endLine)

      elif kernel["WorkGroupMapping"] == -1:
        kStr += "  %s wgSerial = %s(%u) + %s(%u) * %s(%u);%s" \
            % (self.uint64Str, self.getGroupIdStr, 1 if nwgg else 0, self.getGroupIdStr, 0 if nwgg else 0, \
            self.getNumGroupsStr, 1 if nwgg else 0, self.endLine)
        kStr += "  unsigned int %s = wgSerial %% %s(%u);%s" \
            % (wg0, self.getNumGroupsStr, 0 if nwgg else 1, self.endLine)
        kStr += "  unsigned int %s = wgSerial / %s(%u);%s" \
            % (wg1, self.getNumGroupsStr, 0 if nwgg else 1, self.endLine)

      ########################################
      # Blocked rows or columns
      elif kernel["WorkGroupMappingType"] == "D":
        if (kernel["WorkGroupMapping"] > 0) == (not kernel["TransposeWorkGroupGrid"]):
          coal0 = kernel["WorkGroupMapping"] > 0
          dimCoal = (0 if coal0 else 1) # meaning the block is a column going down d0
          dimPerp = (1 if coal0 else 0)
          """
          not trans
          nwgg=true
          wgm>0
          coal0=true # misnomer!
          dimCoal if nwgg else dimPerp -> dimCoal = 0
          dimPerp if nwgg else dimCoal -> dimPerp = 1

          yes trans
          nwgg=false
          wgm<0
          coal0=false
          dimCoal if nwgg else dimPerp -> dimPerp = 0
          dimPerp if nwgg else dimCoal -> dimCoal = 1

          WGB>0 and NoTrans
          block  = wg1/WGB
          blockR = wg1%WGB # wg1 & log2(WGB)
          sIB = wg0 + blockR*ng0 # serial in block
          if (before last block) {
            wg0 = sIB/WGB
            wg1 = sIB%WGB + block*WGB
          } else {
            lBW = ng1%WGB # last block width
            wg0 = sIB/lBW
            wg1 = sIB%lBW + block*WGB
          }
          """

          # work-group free indices
          kStr += self.endLine
          kStr += "  unsigned int %s, %s;%s" \
              % (wg0, wg1, self.endLine)
          kStr += "  %s wgSerial = %s(%u) + (%s(%u)%%WORK_GROUP_MAPPING) * %s(%u);%s" \
              % (self.uint64Str, self.getGroupIdStr, 0 if True else 1, self.getGroupIdStr, 1 if True else 0, \
              self.getNumGroupsStr, 0 if True else 1, self.endLine)
          kStr += "  unsigned int block = %s(%u)/WORK_GROUP_MAPPING;%s" \
              % (self.getGroupIdStr, 1 if True else 0, self.endLine );
          kStr += "  unsigned int blockRemainder = %s(1) %% WORK_GROUP_MAPPING;%s" % \
              ( self.getNumGroupsStr, self.endLine )
          for blockRemainder in range(0, abs(kernel["WorkGroupMapping"])):
            blockWidth = abs(kernel["WorkGroupMapping"]) if blockRemainder==0 else blockRemainder
            if blockRemainder > 0:
              kStr += " else "
            else:
              kStr += "  "
            if blockRemainder < abs(kernel["WorkGroupMapping"])-1:
              kStr += "if ( blockRemainder == %u) " % (blockRemainder)
            kStr += "{%s" % self.endLine
            kStr += "    %s = wgSerial / %u;%s" \
                % ((wg0 if nwgg else wg1), blockWidth, self.endLine)
            kStr += "    %s = wgSerial %% %u + block*WORK_GROUP_MAPPING;%s" \
                % ((wg1 if nwgg else wg0), blockWidth, self.endLine)
            kStr += "  }"
          kStr += "%s" % self.endLine
          #kStr += "if (get_local_id(0)==0) printf(\\\"%%3u: %%2u, %%2u\\\\n\\\", (get_group_id(0)+get_group_id(1)*get_num_groups(0)), %s, %s); return;%s" % (wg0, wg1, self.endLine)


          """
          # if not in last super group
          kStr += "  if ( wgSerial < numWorkGroupsBeforeLastBlock) {%s" \
                  % (self.endLine)
          kStr += "    %s = (wgSerial/WORK_GROUP_MAPPING) %% %s(0);%s" \
              % ((wg0 if nwgg else wg1), \
              self.getNumGroupsStr, self.endLine)
          kStr += "    %s = block*WORK_GROUP_MAPPING + wgSerial %% WORK_GROUP_MAPPING;%s" \
              % ((wg1 if nwgg else wg0), \
              self.endLine)

         # if in last super group
          kStr += "  } else {%s" % self.endLine
          kStr += "    %s = (wgSerial-numWorkGroupsBeforeLastBlock)/lastBlockWidth;%s" \
              % ((wg0 if nwgg else wg1), \
              self.endLine)
          kStr += "    %s = block*WORK_GROUP_MAPPING + wgSerial %% lastBlockWidth;%s" \
              % ((wg1 if nwgg else wg0), \
              self.endLine)

          # if in last super group
          kStr += "  }%s" % self.endLine
          """
        else: # slow case
          coal0 = kernel["WorkGroupMapping"] > 0
          dimCoal = (0 if coal0 else 1)
          dimPerp = (1 if coal0 else 0)

          # work-group free indices
          kStr += self.endLine
          kStr += "  unsigned int %s, %s;%s" \
              % (wg0, wg1, self.endLine)
          kStr += "  %s wgSerial = %s(%u) + %s(%u) * %s(%u);%s" \
              % (self.uint64Str, self.getGroupIdStr, 0 if nwgg else 1, self.getGroupIdStr, 1 if nwgg else 0, \
              self.getNumGroupsStr, 0 if nwgg else 1, self.endLine)
          kStr += "  %s block = wgSerial / (%s(%u)*WORK_GROUP_MAPPING);%s" \
              % (self.uint64Str, self.getNumGroupsStr, dimCoal if nwgg else dimPerp, self.endLine );
          kStr += "  unsigned int lastBlockWidth = %s(%u) %% WORK_GROUP_MAPPING;%s" % \
              ( self.getNumGroupsStr, dimPerp if nwgg else dimCoal, self.endLine )
          kStr += "  unsigned int numWorkGroupsBeforeLastBlock = (%s(%u) - lastBlockWidth)*%s(%u);%s" \
              % (self.getNumGroupsStr, dimPerp if nwgg else dimCoal, self.getNumGroupsStr, dimCoal if nwgg else dimPerp, \
              self.endLine)

          # if not in last super group
          kStr += "  if ( wgSerial < numWorkGroupsBeforeLastBlock) {%s" \
              % (self.endLine)
          kStr += "    %s = (wgSerial/WORK_GROUP_MAPPING) %% %s(%s);%s" \
              % ((wg0 if coal0 else wg1), \
              self.getNumGroupsStr, dimCoal if nwgg else dimPerp, self.endLine)
          kStr += "    %s = block*WORK_GROUP_MAPPING + wgSerial %% WORK_GROUP_MAPPING;%s" \
              % ((wg1 if coal0 else wg0), \
              self.endLine)

          # if in last super group
          kStr += "  } else {%s" % self.endLine
          kStr += "    %s = (wgSerial-numWorkGroupsBeforeLastBlock)/lastBlockWidth;%s" \
              % ((wg0 if coal0 else wg1), \
              self.endLine)
          kStr += "    %s = block*WORK_GROUP_MAPPING + wgSerial %% lastBlockWidth;%s" \
              % ((wg1 if coal0 else wg0), \
              self.endLine)

          # if in last super group
          kStr += "  }%s" % self.endLine



    ########################################
    # Z-Order
    elif kernel["WorkGroupMappingType"] == "Z":

      ########################################
      # wg0,1 -> zg0,1
      kStr += "  unsigned int %s = %s(%u);%s" \
          % (wg0, self.getGroupIdStr, 0 if nwgg else 1, self.endLine)
      kStr += "  unsigned int %s = %s(%u);%s" \
          % (wg1, self.getGroupIdStr, 1 if nwgg else 0, self.endLine)
      kStr += "  unsigned int nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
      kStr += "  unsigned int nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)

      if abs(kernel["WorkGroupMapping"]) == 1: # Generalized Z-Order
        kStr += "generalized_z_order(&%s, &%s, %s, %s, 0, nwg0, nwg1);%s" \
            % ( wg0, wg1, wg0, wg1, self.endLine)

      elif abs(kernel["WorkGroupMapping"]) == 2: # Z-Order round up and return early
        kStr += "  unsigned int wgSerial = %s + %s * %s(%u);%s" % (wg0, wg1, self.getNumGroupsStr, 0 if nwgg else 1, self.endLine)
        kStr += "  z_order(&%s, &%s, wgSerial);%s" % (wg0, wg1, self.endLine)
        kStr += "  if (%s >= nwg0 || %s >= nwg1) return; // wg mapped out of bounds after z-ordering%s" \
            % (wg0, wg1, self.endLine)
      else:
        printExit("WorkGroupMappingType=Z and WorkGroupMapping=%u not supported"%kernel["WorkGroupMapping"])

    #kStr += "  if (%s(0)==0) printf(\\\"wg[%%2u,%%2u] -(%%u)-> wg[%%2u,%%2u]\\\\n\\\", %s(0), %s(1), wgSerial, %s, %s); return;%s" % (self.getLocalIdStr, self.getGroupIdStr, self.getGroupIdStr, wg0, wg1, self.endLine)


    return kStr

  ##############################################################################
  # Global Read Addresses: Subgroup
  ##############################################################################
  def graSubgroup(self, kernel):
    #nwgg = (kernel["WorkGroupMapping"] > 0) # not kernel["TransposeWorkGroupGrid"] # normal work-group grid
    nwgg = not kernel["TransposeWorkGroupGrid"] # normal work-group grid
    kStr = ""
    kStr += "  unsigned int serial = %s(0);%s" \
        % (self.getLocalIdStr, self.endLine)
    kStr += "  unsigned int sgId = serial / (SG%s*SG%s);%s" \
        % (self.tileChar0, self.tileChar1, self.endLine)

    if kernel["GlobalSplitU"] > 1:
      if kernel["GlobalSplitUWorkGroupMappingRoundRobin"]:
        kStr += "  unsigned int gsuWgIdx = wg%s %% (%s(%u)/GLOBAL_SPLITU);%s" \
            % (self.tileChar1, self.getNumGroupsStr, 1 if nwgg else 0, ifself.endLine)
        kStr += "  unsigned int gsuSumIdx = wg%s / (%s(%u)/GLOBAL_SPLITU);%s" \
            % (self.tileChar1, self.getNumGroupsStr, 1 if nwgg else 0, self.endLine)
      else:
        kStr += "  unsigned int gsuWgIdx = wg%s / GLOBAL_SPLITU;%s" \
            % (self.tileChar1, self.endLine)
        kStr += "  unsigned int gsuSumIdx = wg%s %% GLOBAL_SPLITU;%s" \
            % (self.tileChar1, self.endLine)

    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Assignment A
  ##############################################################################
  def graTileAssignmentA(self, kernel):
    kStr = ""
    kStr += "  unsigned int globalReadOffsetA%s = (serial%s" \
        % (self.tileCharA, ("%" if self.globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "/") )
    if self.globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorA"] == kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += " + ("
    if kernel["GlobalSplitU"] > 1 and kernel["ProblemType"]["Tensor0"]==1:
      kStr += "gsuWgIdx"
    else:
      kStr += "wg%s" % (self.tileCharA)
    kStr += ")*MT%s;%s" % (self.tileCharA, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Assignment B
  ##############################################################################
  def graTileAssignmentB(self, kernel):
    kStr = ""
    kStr += "  unsigned int globalReadOffsetB%s = (serial%s" \
        % (self.tileCharB, ("%" if self.globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "/") )
    if self.globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] == kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += " + ("
    if kernel["GlobalSplitU"] > 1 and kernel["ProblemType"]["Tensor0"]==0:
      kStr += "gsuWgIdx"
    else:
      kStr += "wg%s" % (self.tileCharB)
    kStr += ")*MT%s;%s" % (self.tileCharB, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A
  ##############################################################################
  def graUnrollAssignmentA(self, kernel):
    kStr = ""
    kStr += "  unsigned int globalReadOffsetA%s = (serial%s" \
        % (self.unrollChar, ("/" if self.globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "%") )
    if self.globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorA"] != kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    if kernel["GlobalSplitU"] > 1:
      if kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        kStr += " + LOCAL_DEPTHU*"
      else:
        kStr += " + (size%s/GLOBAL_SPLITU)*" % self.unrollChar
      kStr += "gsuSumIdx"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Assignment B
  ##############################################################################
  def graUnrollAssignmentB(self, kernel):
    kStr = ""
    kStr += "  unsigned int globalReadOffsetB%s = (serial%s" \
        % (self.unrollChar, ("/" if self.globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "%") )
    if self.globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] != kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    if kernel["GlobalSplitU"] > 1:
      if kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        kStr += " + LOCAL_DEPTHU*"
      else:
        kStr += " + (size%s/GLOBAL_SPLITU)*" % self.unrollChar
      kStr += "gsuSumIdx"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  def graOtherFreeAssignments(self, kernel):
    kStr = ""
    nonTileFreeIndices = range(0, kernel["ProblemType"]["NumIndicesC"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index0"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index1"])
    for i in range(0, len(nonTileFreeIndices)):
      index = nonTileFreeIndices[i]
      kStr += "  unsigned int wg" + self.indexChars[index] \
          + " = ( " + self.getGroupIdStr + "(2)"
      for j in reversed( range( i+1, len(nonTileFreeIndices)) ):
        index2 = nonTileFreeIndices[j]
        kStr += " / size" + self.indexChars[index2]
      kStr += " ) % size" + self.indexChars[index] + ";" + self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  def graOtherSummationAssignments(self, kernel):
    kStr = ""
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      index = i
      kStr += "#define globalReadOffsetA%s 0%s" \
          % (self.indexChars[index], self.endLine)
      kStr += "#define globalReadOffsetB%s 0%s" \
          % (self.indexChars[index], self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets A
  ##############################################################################
  def graTileOffsetsA(self, kernel):
    kStr = ""
    for l in range(0, self.numReadsTileA):
      if self.readTileDimComponentsA:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetA%s_%u_s%u = globalReadOffsetA%s + %u + %d*%s;%s" \
              % (self.tileCharA, l, s, self.tileCharA, s, l, \
              ("LSCA" if kernel["ProblemType"]["TLUA"] else "LSPA"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetA%s_%u = globalReadOffsetA%s + %d*%s;%s" \
            % (self.tileCharA, l, self.tileCharA, l, \
            ("LSCA" if kernel["ProblemType"]["TLUA"] else "LSPA"), \
            self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets B
  ##############################################################################
  def graTileOffsetsB(self, kernel):
    kStr = ""
    for l in range(0, self.numReadsTileB):
      if self.readTileDimComponentsB:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetB%s_%u_s%u = globalReadOffsetB%s + %u + %d*%s;%s" \
              % (self.tileCharB, l, s, self.tileCharB, s, l, \
              ("LSCB" if kernel["ProblemType"]["TLUB"] else "LSPB"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetB%s_%u = globalReadOffsetB%s + %d*%s;%s" \
            % (self.tileCharB, l, self.tileCharB, l, \
            ("LSCB" if kernel["ProblemType"]["TLUB"] else "LSPB"), \
            self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A
  ##############################################################################
  def graUnrollOffsetsA(self, kernel):
    kStr = ""
    for l in range(0, self.numReadsUnrollA):
      if self.readUnrollDimComponentsA:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetA%s_%u_s%u = globalReadOffsetA%s + %u + %d*%s;%s" \
              % (self.unrollChar, l, s, self.unrollChar, s, l, \
              ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSCA"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetA%s_%u = globalReadOffsetA%s + %d*%s;%s" \
            % (self.unrollChar, l, self.unrollChar, l, \
            ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSCA"), \
            self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Offsets B
  ##############################################################################
  def graUnrollOffsetsB(self, kernel):
    kStr = ""
    for l in range(0, self.numReadsUnrollB):
      if self.readUnrollDimComponentsB:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetB%s_%u_s%u = globalReadOffsetB%s + %u + %d*%s;%s" \
              % (self.unrollChar, l, s, self.unrollChar, s, l, \
              ("LSPB" if kernel["ProblemType"]["TLUB"] else "LSCB"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetB%s_%u = globalReadOffsetB%s + %d*%s;%s" \
            % (self.unrollChar, l, self.unrollChar, l, \
            ("LSPB" if kernel["ProblemType"]["TLUB"] else "LSCB"), \
            self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Branch A
  ##############################################################################
  def graBranchA(self, kernel):
    kStr = ""
    for l in range(0, self.numReadsTileA):
      gro = "(globalReadOffsetA%s_%u%s)" % (self.tileCharA, l, \
          ("_s0 + (VECTOR_WIDTH-1)" if self.readTileDimComponentsA else "") )
      limit = "size%s" % (self.tileCharA)
      kStr += "  bool inBoundsA_%u = %s < %s;%s" \
          % (l, gro, \
          limit, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Branch B
  ##############################################################################
  def graBranchB(self, kernel):
    kStr = ""
    for l in range(0, self.numReadsTileB):
        gro = "(globalReadOffsetB%s_%u%s)" % (self.tileCharB, l, \
            ("_s0 + (VECTOR_WIDTH-1)" if self.readTileDimComponentsB else ""))
        limit = "size%s" % self.tileCharB
        kStr += "  bool inBoundsB_%u = %s < %s;%s" \
            % (l, gro, \
            limit, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Shift A
  ##############################################################################
  def graShiftA(self, kernel):
    kStr = ""
    for l in range(0, self.numReadsTileA):
      for s in range(0, self.numReadVectorComponentsA):
        gro = "globalReadOffsetA%s_%u%s" % (self.tileCharA, l, \
            (("_s%u"%s) if self.readTileDimComponentsA else "") )
        limit = "(size%s-%s)" % (self.tileCharA, \
            ("VECTOR_WIDTH" if self.readTileDimVectorA else "1") )
        kStr += "  %s = (%s > %s) ? %s : %s;%s" \
            % (gro, gro, limit, limit, gro, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Shift B
  ##############################################################################
  def graShiftB(self, kernel):
    kStr = ""
    for l in range(0, self.numReadsTileB):
      for s in range(0, self.numReadVectorComponentsB):
        gro = "globalReadOffsetB%s_%u%s" % (self.tileCharB, l, \
            (("_s%u"%s) if self.readTileDimComponentsB else ""))
        limit = "(size%s-%s)" % (self.tileCharB, \
            ("VECTOR_WIDTH" if self.readTileDimVectorB else "1") )
        kStr += "  %s = (%s > %s) ? %s : %s;%s" \
            % (gro, gro, limit, limit, gro, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets A
  ##############################################################################
  def graFinalOffsetsA(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numReadVectorComponentsA):
          kStr += "  %s globalReadOffsetA_%u_%u%s = GLOBAL_OFFSET_A( " \
              % (self.uint64Str, para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsA \
              or self.readUnrollDimComponentsA) else ""))
          for i in range(0, len(kernel["ProblemType"]["IndexAssignmentsA"])):
            index = kernel["ProblemType"]["IndexAssignmentsA"][i]
            if index < kernel["ProblemType"]["NumIndicesC"]:
              if index == kernel["ProblemType"]["TileA"]:
                kStr += "globalReadOffsetA%s_%u%s" \
                    % (self.tileCharA, \
                    (para if kernel["ProblemType"]["TLUA"] else perp), \
                    (("_s%u"%s) if self.readTileDimComponentsA else "") )
              else: # just a group index
                kStr += "wg" + self.indexChars[index]
            else: # summation index
              if index == kernel["ProblemType"]["IndexUnroll"]:
                kStr += "globalReadOffsetA%s_%u%s" \
                    % (self.unrollChar, \
                    (perp if kernel["ProblemType"]["TLUA"] else para), \
                    (("_s%u"%s) if self.readUnrollDimComponentsA else "") )
              else:
                kStr += "globalReadOffsetA%s" % self.indexChars[index]
            if i < len(kernel["ProblemType"]["IndexAssignmentsA"])-1:
              kStr += ", "
          kStr += " );%s" % self.endLine
          """
          kStr += "  printf(\\\"GRA T[%%02u] gROA_%u_%u%s = %%4u\\\\n\\\", serial, globalReadOffsetA_%u_%u%s);%s" \
              % (para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsA \
              or self.readUnrollDimComponentsA) else ""), \
              para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsA \
              or self.readUnrollDimComponentsA) else ""), \
              self.endLine )
          """
    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets B
  ##############################################################################
  def graFinalOffsetsB(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):
          kStr += "  %s globalReadOffsetB_%u_%u%s = GLOBAL_OFFSET_B( " \
              % (self.uint64Str, para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsB \
              or self.readUnrollDimComponentsB) else ""))
          for i in range(0, len(kernel["ProblemType"]["IndexAssignmentsB"])):
            index = kernel["ProblemType"]["IndexAssignmentsB"][i]
            if index < kernel["ProblemType"]["NumIndicesC"]:
              if index == kernel["ProblemType"]["TileB"]:
                kStr += "globalReadOffsetB%s_%u%s" \
                    % (self.tileCharB, \
                    (para if kernel["ProblemType"]["TLUB"] else perp), \
                    (("_s%u"%s) if self.readTileDimComponentsB else "") )
              else: # just a group index
                kStr += "wg" + self.indexChars[index]
            else: # summation index
              if index == kernel["ProblemType"]["IndexUnroll"]:
                kStr += "globalReadOffsetB%s_%u%s" \
                    % (self.unrollChar, \
                    (perp if kernel["ProblemType"]["TLUB"] else para), \
                    (("_s%u"%s) if self.readUnrollDimComponentsB else "") )
              else:
                kStr += "globalReadOffsetB%s" % self.indexChars[index]
            if i < len(kernel["ProblemType"]["IndexAssignmentsB"])-1:
              kStr += ", "
          kStr += " );%s" % self.endLine
          """
          kStr += "  printf(\\\"GRB T[%%02u] gROB_%u_%u%s = %%4u\\\\n\\\", serial, globalReadOffsetB_%u_%u%s);%s" \
              % (para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsB \
              or self.readUnrollDimComponentsB) else ""), \
              para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsB \
              or self.readUnrollDimComponentsB) else ""), \
              self.endLine )
          """
    return kStr

  ##############################################################################
  # Global Read Addresses: Apply User Offsets
  ##############################################################################
  def graApplyUserOffsets(self, kernel):
    kStr = ""
    kStr += "  C += offsetC;%s" % self.endLine
    kStr += "  A += offsetA;%s" % self.endLine
    kStr += "  B += offsetB;%s" % self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses A
  ##############################################################################
  def graAddressesA(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        if self.readTileDimComponentsA or self.readUnrollDimComponentsA:
          for s in range(0, self.numReadVectorComponentsA):
            kStr += "  %sDATA_TYPE const *globalReadA_%u_%u%s = A + globalReadOffsetA_%u_%u%s;%s" \
                % (self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                self.endLine)
        else:
            kStr += "  %sVECTOR_TYPE const *globalReadA_%u_%u = (%sVECTOR_TYPE const *)(A + globalReadOffsetA_%u_%u);%s" \
                % (self.globalPtrStr, para, perp, self.globalPtrStr, \
                para, perp, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses B
  ##############################################################################
  def graAddressesB(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        if self.readTileDimComponentsB or self.readUnrollDimComponentsB:
          for s in range(0, self.numReadVectorComponentsB):
            kStr += "  %sDATA_TYPE const *globalReadB_%u_%u%s = B + globalReadOffsetB_%u_%u%s;%s" \
                % (self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), \
                para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), self.endLine)
        else:
            kStr += "  %sVECTOR_TYPE const *globalReadB_%u_%u = (%sVECTOR_TYPE const *)(B + globalReadOffsetB_%u_%u);%s" \
                % (self.globalPtrStr, para, perp, self.globalPtrStr, \
                para, perp, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments A
  ##############################################################################
  def graIncrementsA(self, kernel, loopIdx):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    kStr += "%s%s globalReadIncA%s = (%s)strideA%s" \
        % (self.indent, self.int64Str, loopChar, \
        self.int64Str, loopChar)
    if loopIdx==kernel["ProblemType"]["NumIndicesSummation"]-1:
      kStr += "*LOCAL_DEPTHU"
      if kernel["GlobalSplitU"] > 1 \
          and kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        kStr += "*GLOBAL_SPLITU"
    else:
      for j in range(loopIdx+1, \
          min(loopIdx+2,kernel["ProblemType"]["NumIndicesSummation"]) ):
        tmpChar = self.indexChars[ \
            kernel["ProblemType"]["IndicesSummation"][j]]
        kStr += " - strideA%s*size%s" % (tmpChar, tmpChar)
    kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments B
  ##############################################################################
  def graIncrementsB(self, kernel, loopIdx):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    kStr += "%s%s globalReadIncB%s = (%s)strideB%s" \
        % (self.indent, self.int64Str, loopChar, \
        self.int64Str, loopChar)
    if loopIdx==kernel["ProblemType"]["NumIndicesSummation"]-1:
      kStr += "*LOCAL_DEPTHU"
      if kernel["GlobalSplitU"] > 1 \
          and kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        kStr += "*GLOBAL_SPLITU"
    else:
      for j in range(loopIdx+1, \
          min(loopIdx+2,kernel["ProblemType"]["NumIndicesSummation"]) ):
        tmpChar = self.indexChars[ \
            kernel["ProblemType"]["IndicesSummation"][j]]
        kStr += " - strideB%s*size%s" % (tmpChar, tmpChar)
    kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A
  ##############################################################################
  def lwaTileAssignmentA(self, kernel):
    kStr = ""
    kStr += "  unsigned int lwA%s = (serial%s" \
        % (self.tileCharA, ("%" if self.globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "/") )
    if self.globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")";
    if kernel["GlobalReadCoalesceVectorA"] == kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment B
  ##############################################################################
  def lwaTileAssignmentB(self, kernel):
    kStr = ""
    kStr += "  unsigned int lwB%s = (serial%s" \
        % (self.tileCharB, ("%" if self.globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "/") )
    if self.globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] == kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A
  ##############################################################################
  def lwaUnrollAssignmentA(self, kernel):
    kStr = ""
    kStr += "  unsigned int lwA%s = (serial%s" \
        % (self.unrollChar, ("/" if self.globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "%") )
    if self.globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")";
    if kernel["GlobalReadCoalesceVectorA"] != kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Unroll Assignment B
  ##############################################################################
  def lwaUnrollAssignmentB(self, kernel):
    kStr = ""
    kStr += "  unsigned int lwB%s = (serial%s" \
        % (self.unrollChar, ("/" if self.globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "%") )
    if self.globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] != kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: First Offset A
  ##############################################################################
  def lwaFirstOffsetA(self, kernel):
    kStr = ""
    kStr += "  unsigned int localWriteFirstOffsetA = lwA%s + lwA%s*(MT%s+PAD);%s" \
        % (self.tileCharA, self.unrollChar, self.tileCharA, self.endLine)
    return kStr

  ##############################################################################
  # Local Write Addresses: First Offset B
  ##############################################################################
  def lwaFirstOffsetB(self, kernel):
    kStr = ""
    kStr += "  unsigned int localWriteFirstOffsetB = lwB%s + lwB%s*(MT%s+PAD) + LDS_OFFSET_B;%s" \
        % (self.tileCharB, self.unrollChar, self.tileCharB, self.endLine)
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A
  ##############################################################################
  def lwaFinalOffsetsA(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "  unsigned int localWriteOffsetA_%u_%u%s = localWriteFirstOffsetA + (%s%d*%s)" \
              % (para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              (("%u + "%s) if self.writeTileDimComponentsA else ""), \
              para, ("LSCA" if not kernel["ProblemType"]["TLUA"] else "LSCA") )
          if not kernel["ProblemType"]["TLUA"]:
            kStr += "*(MT%s+PAD)" % (self.tileCharA)
          kStr += " + (%s%d*%s)" % (
              (("%u + "%s) if self.writeUnrollDimComponentsA else ""), perp, \
              ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSPA") )
          if kernel["ProblemType"]["TLUA"]:
            kStr += "*(MT%s+PAD)" % (self.tileCharA)
          kStr += ";%s" % self.endLine
          """
          kStr += "  printf(\\\"LWA T[%%02u] lWOA_%u_%u%s = %%4u\\\\n\\\", serial, localWriteOffsetA_%u_%u%s);%s" \
              % (para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.endLine )
          """
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets B
  ##############################################################################
  def lwaFinalOffsetsB(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "  unsigned int localWriteOffsetB_%u_%u%s = localWriteFirstOffsetB + (%s%d*%s)" \
              % (para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              (("%u + "%s) if self.writeTileDimComponentsB else ""), para, \
              ("LSCB" if not kernel["ProblemType"]["TLUB"] else "LSCB") )
          if not kernel["ProblemType"]["TLUB"]:
            kStr += "*(MT%s+PAD)" % (self.tileCharB)
          kStr += " + (%s%d*%s)" % ( \
              (("%u + "%s) if self.writeUnrollDimComponentsB else ""), perp, \
              ("LSPB" if not kernel["ProblemType"]["TLUB"] else "LSPB") )
          if kernel["ProblemType"]["TLUB"]:
            kStr += "*(MT%s+PAD)" % (self.tileCharB)
          kStr += ";%s" % self.endLine
          """
          kStr += "  printf(\\\"LWB T[%%02u] lWOB_%u_%u%s = %%4u\\\\n\\\", serial, localWriteOffsetB_%u_%u%s);%s" \
             % (para, perp,
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.endLine )
          """
    return kStr

  ##############################################################################
  # Local Write Addresses: Declare Addresses A
  ##############################################################################
  def lwaDeclareAddressesA(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "  %s%s *localWriteA_%u_%u%s;%s"\
              % (self.sharedPtrStr, \
              ("DATA_TYPE" if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), self.endLine )
    return kStr

  ##############################################################################
  # Local Write Addresses: Declare Addresses B
  ##############################################################################
  def lwaDeclareAddressesB(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "  %s%s *localWriteB_%u_%u%s;%s"\
              % (self.sharedPtrStr, ("DATA_TYPE" \
              if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), self.endLine )
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment A
  ##############################################################################
  def lraTileAssignmentA(self, kernel):
    kStr = ""
    kStr += "  unsigned int lr%s = (serial %% SG%s);%s" \
        % (self.tileChar0, self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment B
  ##############################################################################
  def lraTileAssignmentB(self, kernel):
    kStr = ""
    kStr += "  unsigned int lr%s = (serial / SG%s) %% SG%s;%s" \
        % (self.tileChar1, self.tileChar0, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset A
  ##############################################################################
  def lraFinalOffsetA(self, kernel):
    kStr = ""
    kStr += "  unsigned int localReadOffsetA = lr%s*VECTOR_WIDTH + sgId*(MT%s+PAD);%s" \
        % ( self.tileChar0, self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset B
  ##############################################################################
  def lraFinalOffsetB(self, kernel):
    kStr = ""
    kStr += "  unsigned int localReadOffsetB = lr%s*VECTOR_WIDTH + sgId*(MT%s+PAD) + LDS_OFFSET_B;%s" \
        % (self.tileChar1, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Declare Addresses A
  ##############################################################################
  def lraDeclareAddressesA(self, kernel):
    kStr = ""
    kStr += "  %sVECTOR_TYPE *localReadA;%s" % (self.sharedPtrStr, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Declare Addresses B
  ##############################################################################
  def lraDeclareAddressesB(self, kernel):
    kStr = ""
    kStr += "  %sVECTOR_TYPE *localReadB;%s" % (self.sharedPtrStr, self.endLine)
    return kStr

  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  def declareLoopNumIter(self, kernel):
    kStr = ""
    for loopIdx in kernel["ProblemType"]["IndicesSummation"]:
      loopChar = self.indexChars[loopIdx]
      kStr += "%sunsigned int numIter%s;%s" \
          % (self.indent, loopChar, self.endLine)
    return kStr

  ##############################################################################
  # Calculate Loop Num Iterations
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx):
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx

    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if tailLoop:
      kStr += "%snumIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU);%s" \
          % (self.indent, self.unrollChar, self.unrollChar, self.endLine)
      #kStr += "if (serial==0) printf(\\\"wg0:%u, wg1:%u, gsuWg:%u, gsuSum:%u, numIter:%u, r:%u\\\\n\\\", wg0I, wg1J, gsuWgIdx, gsuSumIdx, numIterK, numIterPerWgRemainder);" + self.endLine
      if kernel["GlobalSplitU"] > 1:
        kStr += "%sif (gsuSumIdx != numIterPerWgRemainder) {%s" \
            % (self.indent, self.endLine)
        kStr += "%s  numIter%s = 0;%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%s}%s" % (self.indent, self.endLine)
        #kStr += "if (serial==0) printf(\\\"WG%u_%u TK:%u\\\\n\\\", get_group_id(0), get_group_id(1), numIterK);" + self.endLine
    else:
      kStr += "%snumIter%s = size%s" \
          % (self.indent, loopChar, loopChar)
      if loopIdx == self.unrollIdx:
        kStr += " / LOCAL_DEPTHU"
      kStr += ";%s" % self.endLine

      if loopIdx == self.unrollIdx and kernel["GlobalSplitU"] > 1:
        kStr += "%sunsigned int numIterMyWg = numIter%s / GLOBAL_SPLITU;%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%sunsigned int numIterPerWgRemainder = numIter%s %% GLOBAL_SPLITU;%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%sif (gsuSumIdx < numIterPerWgRemainder) {%s" \
            % (self.indent, self.endLine)
        kStr += "%s  numIterMyWg++;%s" % (self.indent, self.endLine)
        kStr += "%s}%s" % (self.indent, self.endLine)
        kStr += "%snumIter%s = numIterMyWg;%s" \
            % (self.indent, self.unrollChar, self.endLine)
        #kStr += "if (serial==0) printf(\\\"WG%u_%u UK:%u\\\\n\\\", get_group_id(0), get_group_id(1), numIterK);" + self.endLine
    return kStr



  ##############################################################################
  # Open Loop
  ##############################################################################
  def openLoop(self, kernel, loopIdx):
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx

    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if kernel["LoopDoWhile"]:
      kStr += "%sdo {%s" % (self.indent, self.endLine)
    else:
      kStr += "%swhile (numIter%s-- > %u) {%s" \
          % (self.indent, loopChar, \
          (1 if (kernel["PrefetchGlobalRead"] and loopIdx == self.unrollIdx \
          and not tailLoop) else 0), self.endLine)
    self.indent += "  "
    #if tailLoop:
    #  kStr += "if (serial==0) printf(\\\"WG%u_%u: ti=%u\\\\n\\\", get_group_id(0), get_group_id(1), numIterK);" + self.endLine
    #else:
    #  kStr += "if (serial==0) printf(\\\"WG%u_%u: ui=%u\\\\n\\\", get_group_id(0), get_group_id(1), numIterK);" + self.endLine
    return kStr

  ##############################################################################
  # Close Loop
  ##############################################################################
  def closeLoop(self, kernel, loopIdx):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    self.indent = self.indent[2:]
    if kernel["LoopDoWhile"]:
      kStr += "%s} while (--numIter%s > %u);%s" \
          % (self.indent, loopChar, \
          (1 if kernel["PrefetchGlobalRead"] else 0), self.endLine )
    else:
      kStr += "%s}%s" % (self.indent, self.endLine)
    #kStr += "if (serial==0) printf(\\\"WG%u_%u: rc0=%.0f\\\\n\\\", get_group_id(0), get_group_id(1), rC[0]);" + self.endLine
    return kStr

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self):
    return ""

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  def macIter(self, kernel, black):
    kStr = ""
    kStr += "%sMAC_%ux%u" % (self.indent, \
        kernel["ThreadTile0"],kernel["ThreadTile1"])
    if black:
      kStr += "_BLK"
    kStr += self.endLine
    return kStr

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel):
    kStr = ""
    if kernel["GlobalSplitU"] > 1:
      kStr += "%sif (numIterMyWg >= 1) {%s" \
          % (self.indent, self.endLine)
    else:
      kStr += "%sif (size%s >= LOCAL_DEPTHU) {%s" \
          % (self.indent, self.unrollChar, self.endLine)
    self.indent += "  "
    return kStr
  def closeSumAtLeastUnroll(self, kernel):
    kStr = ""
    self.indent = self.indent[2:]
    kStr += "%s}%s" % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Global Read: Increment A
  ##############################################################################
  def globalReadIncrementA(self, kernel, loopIdx):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numReadVectorComponentsA):
          if self.readTileDimVectorA or self.readUnrollDimVectorA:
            kStr += "%sglobalReadA_%u_%u%s = (%sVECTOR_TYPE const *)( ((%sDATA_TYPE const *)globalReadA_%u_%u%s) + globalReadIncA%s);%s" \
                % (self.indent, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                self.globalPtrStr, self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                loopChar, self.endLine)
          else:
            kStr += "%sglobalReadA_%u_%u%s += globalReadIncA%s%s;%s" \
                % (self.indent, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                loopChar, "" if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else "/VECTOR_WIDTH", \
                self.endLine)
    return kStr

  ##############################################################################
  # Global Read: Increment B
  ##############################################################################
  def globalReadIncrementB(self, kernel, loopIdx):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):
          if self.readTileDimVectorB or self.readUnrollDimVectorB:
            kStr += "%sglobalReadB_%u_%u%s = (%sVECTOR_TYPE const *)( ((%sDATA_TYPE const *)globalReadB_%u_%u%s) + globalReadIncB%s);%s" \
                % (self.indent, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), \
                self.globalPtrStr, self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), \
                loopChar, self.endLine )
          else:
            kStr += "%sglobalReadB_%u_%u%s += globalReadIncB%s%s;%s" \
                % (self.indent, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), \
                loopChar, "" if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else "/VECTOR_WIDTH", \
                self.endLine)
    return kStr

  ##############################################################################
  # Global Read: Do It A
  ##############################################################################
  def globalReadDoA(self, kernel, guardK):
    kStr = ""
    guardUnrolledComponents = self.readUnrollDimVectorA and guardK \
        and kernel["VectorWidth"]>1
    numUnrollVectorComponents = kernel["VectorWidth"] \
        if guardUnrolledComponents else self.numReadVectorComponentsA


    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, numUnrollVectorComponents):
          kStr += "%sa_%u_%u%s = " % (self.indent, para, perp, \
              ((".%s"%self.vectorComponents[s]) if (self.readTileDimComponentsA\
              or self.readUnrollDimComponentsA or guardUnrolledComponents) else "") )
          # guard around K
          if guardK:
            kStr += "( globalReadOffsetA%s_%u%s%s >= (size%s %% LOCAL_DEPTHU%s)%s )" \
                % (self.unrollChar, \
                (perp if kernel["ProblemType"]["TLUA"] else para), \
                (("_s%u"%s) if self.readUnrollDimComponentsA else ""), \
                "+%u"%s if guardUnrolledComponents else "", \
                self.unrollChar, \
                (" + LOCAL_DEPTHU*gsuSumIdx" if kernel["GlobalSplitU"]>1 else ""), \
                (" || !numIter%s"%self.unrollChar) if kernel["GlobalSplitU"] > 1 else "")
          # guard around edge
          if kernel["EdgeType"] == "Branch":
            if guardK:
              kStr += " || "
            kStr += "( !inBoundsA_%u )" % ( \
                (para if kernel["ProblemType"]["TLUA"] else perp) )
          if kernel["EdgeType"] == "Branch" or guardK:
            kStr += " ? %s : " % ("SCALAR_ZERO" if (self.readTileDimComponentsA\
              or self.readUnrollDimComponentsA or guardUnrolledComponents) else "VECTOR_ZERO")
          kStr += "*(%sglobalReadA_%u_%u%s%s);%s" \
              % ("((DATA_TYPE*)" if guardUnrolledComponents else "", \
              para, perp, (("_s%u"%s) if (self.readTileDimComponentsA \
              or self.readUnrollDimComponentsA) else ""), \
              ")+%u"%s if guardUnrolledComponents else "", \
              self.endLine)
    return kStr

  ##############################################################################
  # Global Gead: Do It B
  ##############################################################################
  def globalReadDoB(self, kernel, guardK):
    kStr = ""
    guardUnrolledComponents = self.readUnrollDimVectorB and guardK \
        and kernel["VectorWidth"]>1
    numUnrollVectorComponents = kernel["VectorWidth"] \
        if guardUnrolledComponents else self.numReadVectorComponentsB
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, numUnrollVectorComponents):
          kStr += "%sb_%u_%u%s = " % (self.indent, para, perp, \
              ((".%s"%self.vectorComponents[s]) if (self.readTileDimComponentsB\
              or self.readUnrollDimComponentsB or guardUnrolledComponents) \
              else "") )
          # guard around k
          if guardK:
            kStr += "( globalReadOffsetB%s_%u%s%s >= (size%s %% LOCAL_DEPTHU%s)%s )" \
                % (self.unrollChar, \
                (perp if kernel["ProblemType"]["TLUB"] else para), \
                (("_s%u"%s) if self.readUnrollDimComponentsB else ""), \
                "+%u"%s if guardUnrolledComponents else "", \
                self.unrollChar, \
                (" + LOCAL_DEPTHU*gsuSumIdx" if kernel["GlobalSplitU"]>1 else ""), \
                (" || !numIter%s"%self.unrollChar) if kernel["GlobalSplitU"] > 1 else "")
          # guard around edge
          if kernel["EdgeType"] == "Branch":
            if guardK:
              kStr += " || "
            kStr += "( !inBoundsB_%u )" % ( \
                (para if kernel["ProblemType"]["TLUB"] else perp) )
          if kernel["EdgeType"] == "Branch" or guardK:
            kStr += " ? %s : " % ("SCALAR_ZERO" if (self.readTileDimComponentsB\
              or self.readUnrollDimComponentsB or guardUnrolledComponents) else "VECTOR_ZERO")
          kStr += "*(%sglobalReadB_%u_%u%s%s);%s" \
              % ("((DATA_TYPE*)" if guardUnrolledComponents else "", \
              para, perp, (("_s%u"%s) if (self.readTileDimComponentsB \
              or self.readUnrollDimComponentsB) else ""), \
              ")+%u"%s if guardUnrolledComponents else "", \
              self.endLine)
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets A
  ##############################################################################
  def localWriteSwapOffsetsA(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "%slocalWriteOffsetA_%u_%u%s = (localWriteOffsetA_%u_%u%s + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
              % (self.indent, \
              para, perp, (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              para, perp, (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), self.endLine )
          """
          kStr += "%slocalWriteA_%u_%u%s = (%s%s *)(localMemory + localWriteOffsetA_%u_%u%s);%s"\
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.endLine)
          """
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets B
  ##############################################################################
  def localWriteSwapOffsetsB(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "%slocalWriteOffsetB_%u_%u%s = (localWriteOffsetB_%u_%u%s + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              para, perp, (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), self.endLine )
          """
          kStr += "%slocalWriteB_%u_%u%s = (%s%s *)(localMemory + localWriteOffsetB_%u_%u%s);%s"\
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.endLine)
          """
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets A
  ##############################################################################
  def localWriteResetOffsetsA(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "%slocalWriteOffsetA_%u_%u%s %%= LDS_OFFSET_BLK;%s" \
              % (self.indent, \
              para, perp, (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), self.endLine )
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets B
  ##############################################################################
  def localWriteResetOffsetsB(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "%slocalWriteOffsetB_%u_%u%s %%= LDS_OFFSET_BLK;%s" \
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), self.endLine )
    return kStr



  ##############################################################################
  # Local Write: Init Pointers A
  ##############################################################################
  def localWriteInitPointersA(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "%slocalWriteA_%u_%u%s = (%s%s *)(localMemory + localWriteOffsetA_%u_%u%s);%s"\
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.endLine)
    return kStr

  ##############################################################################
  # Local Write: Init Pointers B
  ##############################################################################
  def localWriteInitPointersB(self, kernel):
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "%slocalWriteB_%u_%u%s = (%s%s *)(localMemory + localWriteOffsetB_%u_%u%s);%s"\
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.endLine)
    return kStr



  ##############################################################################
  # Local Write: Do It A
  ##############################################################################
  def localWriteDoA(self, kernel):
    kStr = ""
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wconditional-uninitialized\"" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "%s*localWriteA_%u_%u%s = a_%u_%u%s;%s" \
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "" ), \
              para, perp, \
              ((".%s"%self.vectorComponents[s]) \
              if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "" ), \
              self.endLine)
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine
    return kStr

  ##############################################################################
  # Local Write: Do It B
  ##############################################################################
  def localWriteDoB(self, kernel):
    kStr = ""
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wconditional-uninitialized\"" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "%s*localWriteB_%u_%u%s = b_%u_%u%s;%s" \
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "" ), \
              para, perp, \
              ((".%s"%self.vectorComponents[s]) \
              if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "" ), \
              self.endLine)
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine
    return kStr

  ##############################################################################
  # Local Read: Swap Offsets A
  ##############################################################################
  def localReadSwapOffsetsA(self, kernel):
    kStr = ""
    kStr += "%slocalReadOffsetA = (localReadOffsetA + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
        % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Wwap Offsets B
  ##############################################################################
  def localReadSwapOffsetsB(self, kernel):
    kStr = ""
    kStr += "%slocalReadOffsetB = (localReadOffsetB + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
        % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A
  ##############################################################################
  def localReadResetOffsetsA(self, kernel):
    kStr = ""
    kStr += "%slocalReadOffsetA %%= LDS_OFFSET_BLK;%s" \
        % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets B
  ##############################################################################
  def localReadResetOffsetsB(self, kernel):
    kStr = ""
    kStr += "%slocalReadOffsetB %%= LDS_OFFSET_BLK;%s" \
        % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Init Pointers A
  ##############################################################################
  def localReadInitPointersA(self, kernel):
    kStr = ""
    kStr += "%slocalReadA = (%sVECTOR_TYPE *)(localMemory + localReadOffsetA);%s" \
        % (self.indent, self.sharedPtrStr, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Init Pointers B
  ##############################################################################
  def localReadInitPointersB(self, kernel):
    kStr = ""
    kStr += "%slocalReadB = (%sVECTOR_TYPE *)(localMemory + localReadOffsetB);%s" \
        % (self.indent, self.sharedPtrStr, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Increment A
  ##############################################################################
  def localReadIncA(self, kernel):
    kStr = ""
    kStr += "%slocalReadA += LOCAL_SPLITU*(MT%s/VECTOR_WIDTH+PAD);%s" \
        % (self.indent, self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Increment B
  ##############################################################################
  def localReadIncB(self, kernel):
    kStr = ""
    kStr += "%slocalReadB += LOCAL_SPLITU*(MT%s/VECTOR_WIDTH+PAD);%s" \
        % (self.indent, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Do It A
  ##############################################################################
  def localReadDoA(self, kernel, black):
    kStr = ""
    for a in range(0, kernel["ThreadTileA"]/kernel["VectorWidth"]):
      kStr += "%srA[%d%s] = localReadA[%d*SG%s]; %s" \
          % (self.indent, a, \
          (("+TT%s/VECTOR_WIDTH"%self.tileCharA) if black else ""), \
          a, self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Do It B
  ##############################################################################
  def localReadDoB(self, kernel, black):
    kStr = ""
    for b in range(0, kernel["ThreadTileB"]/kernel["VectorWidth"]):
      kStr += "%srB[%d%s] = localReadB[%d*SG%s]; %s" \
          % (self.indent, b, \
          (("+TT%s/VECTOR_WIDTH"%self.tileCharB) if black else ""), \
          b, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Shift Vector Components d0
  ##############################################################################
  def shiftVectorComponents0(self, kernel):
    kStr = ""
    kStr += "  unsigned int wgMT%s = size%s - wg%s*MT%s;%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, \
        self.tileChar0, self.endLine)
    kStr += "  if (wgMT%s > MT%s) wgMT%s = MT%s;%s" \
        %(self.tileChar0, self.tileChar0, self.tileChar0, \
        self.tileChar0, self.endLine)
    kStr += "  unsigned int r%s = wgMT%s %% VECTOR_WIDTH;%s" \
        % (self.tileChar0, self.tileChar0, self.endLine)
    kStr += "  if (r%s > 0 && ((wgMT%s/VECTOR_WIDTH)%%SG%s) == serial %% SG%s ) {%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, \
        self.tileChar0, self.endLine)
    kStr += "    unsigned int s%s = (wgMT%s/VECTOR_WIDTH)/SG%s;%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
    for r0 in range(1, kernel["VectorWidth"]):
      kStr += "    if (r%s == %u) {%s" % (self.tileChar0, r0, self.endLine)
      numVectors = kernel["ThreadTile0"]/kernel["VectorWidth"]
      for vIdx in range(0, numVectors):
        if vIdx == 0:
          kStr += "      "
        else:
          kStr += " else "
        if vIdx < numVectors-1:
          kStr += "if (s%s == %u) " % (self.tileChar0, vIdx)
        kStr += "{%s" % self.endLine
        for tt1 in range(0, kernel["ThreadTile1"]):
          for s in range(0, r0):
            kStr += "        rC[%u+%u*(TT%s/VECTOR_WIDTH)].%s = rC[%u+%u*(TT%s/VECTOR_WIDTH)].%s;%s" \
              % (vIdx, tt1, self.tileChar0, self.vectorComponents[s],  \
              vIdx, tt1, self.tileChar0, \
              self.vectorComponents[s+kernel["VectorWidth"]-r0], self.endLine)
        #kStr += "printf(\\\"sv %u %u\\\");%s" % (r0, vIdx, self.endLine)
        kStr += "      }"
        if vIdx == numVectors-1:
          kStr += self.endLine
      kStr += "    }%s" % self.endLine
    kStr += "  }%s" % self.endLine
    return kStr

  ##############################################################################
  # Shift Vectors Components d1
  ##############################################################################
  def shiftVectorComponents1(self, kernel):
    kStr = ""
    kStr += "  unsigned int wgMT%s = size%s - %s*MT%s;%s" \
        % (self.tileChar1, self.tileChar1, "gsuWgIdx" \
        if kernel["GlobalSplitU"]>1 else ("wg%s"%self.tileChar1), \
        self.tileChar1, self.endLine)
    kStr += "  if (wgMT%s > MT%s) wgMT%s = MT%s;%s" \
        %(self.tileChar1, self.tileChar1, self.tileChar1, \
        self.tileChar1, self.endLine)
    kStr += "  unsigned int r%s = wgMT%s %% VECTOR_WIDTH;%s" \
        % (self.tileChar1, self.tileChar1, self.endLine)
    kStr += "  if (r%s > 0 && ((wgMT%s/VECTOR_WIDTH) %% SG%s) == ((serial / SG%s) %% SG%s) ) {%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, \
        self.tileChar0, self.tileChar1, \
        self.endLine)
    kStr += "    unsigned int s%s = (wgMT%s/VECTOR_WIDTH)/SG%s;%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
    for r1 in range(1, kernel["VectorWidth"]):
      kStr += "    if (r%s == %u) {%s" % (self.tileChar1, r1, self.endLine)
      numVectors = kernel["ThreadTile1"]/kernel["VectorWidth"]
      for vIdx in range(0, numVectors):
        if vIdx == 0:
          kStr += "      "
        else:
          kStr += " else "
        if vIdx < numVectors-1:
          kStr += "if (s%s == %u) " % (self.tileChar1, vIdx)
        kStr += "{%s" % self.endLine

        for tt0 in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
          for s in range(0, r1):
            kStr += "        rC[%u+%u*(TT%s/VECTOR_WIDTH)*(VECTOR_WIDTH) + %u*(TT%s/VECTOR_WIDTH)] = rC[%u+%u*(TT%s/VECTOR_WIDTH)*(VECTOR_WIDTH) + %u*(TT%s/VECTOR_WIDTH)];%s" \
              % (tt0, vIdx, self.tileChar0, s, self.tileChar0, \
              tt0, vIdx, self.tileChar0, \
              s+kernel["VectorWidth"]-r1, self.tileChar0, self.endLine)

        kStr += "      }"
        if vIdx == numVectors-1:
          kStr += self.endLine

      kStr += "    }%s" % self.endLine
    kStr += "  }%s" % self.endLine
    return kStr

  ##############################################################################
  # Complex Declare Tmp Registers
  ##############################################################################
  def complexDeclareTmpRegisters(self, kernel):
    kStr = ""
    if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
      kStr += "  float type_mac_tmp;" + self.endLine
    if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
      kStr += "  double type_mac_tmp;" + self.endLine
    return kStr


  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  def localSplitULocalWrite(self, kernel):
    kStr = ""
    kStr += "  %sVECTOR_TYPE *localLocalSplitU = (%sVECTOR_TYPE *)(localMemory);%s" \
        % (self.sharedPtrStr, self.sharedPtrStr, self.endLine)
    for j in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          kStr += "%slocalLocalSplitU[lr%s + %u*SG%s + (MT%s/VECTOR_WIDTH)*(lr%s*VECTOR_WIDTH + %u + SG%s*VECTOR_WIDTH*%u) + (MT%s*MT%s/VECTOR_WIDTH)*sgId] = rC[%u+%u*(TT%s/VECTOR_WIDTH)+%u*TT%s];%s" \
              % (self.indent, self.tileChar0, i, self.tileChar0, \
              self.tileChar0, self.tileChar1, \
              s, self.tileChar1, j, self.tileChar0, self.tileChar1, i, s, \
              self.tileChar0, j, self.tileChar0, self.endLine)
    kStr += self.indent + self.syncStr + self.endLine
    """
    kStr += "    /* print Local state */" + self.endLine
    kStr += "    for (unsigned int i = serial; i < MT0I*MT1J*LOCAL_SPLITU; i+=NUM_THREADS) {%s" % self.endLine
    kStr += "      printf(\\\"localLocalSplitU[%%06u] = %%10.0f, %%10.0f\\\\n\\\", i, localLocalSplitU[i], localLocalSplitU[i]);%s" \
        % self.endLine
    kStr += "    }" + self.endLine
    """
    return kStr

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    kStr = ""
    for i in range(0, kernel["NumVectorsPerThread"]):
      kStr += "  rC[%3u] = localLocalSplitU[serial+%u*NUM_THREADS];%s" \
          % (i, i, self.endLine)
    kStr += self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  def localSplitUReduction(self, kernel):
    kStr = ""
    for s in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumVectorsPerThread"]):
        kStr += "  rC[%3u] += localLocalSplitU[serial+%u*NUM_THREADS + %u*(MT%s*MT%s/VECTOR_WIDTH)];%s" \
            % (i, i, s, self.tileChar0, self.tileChar1, self.endLine)
      kStr += self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    kStr = ""
    kStr += "  unsigned int localC%s = (serial %% (MT%s/VECTOR_WIDTH))*VECTOR_WIDTH;%s" \
        % (self.tileChar0, self.tileChar0, self.endLine)
    kStr += "  unsigned int localC%s = serial / (MT%s/VECTOR_WIDTH);%s" \
        % (self.tileChar1, self.tileChar0, self.endLine)
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int globalC%s = (" \
          % (self.indexChars[i])
      if i == kernel["ProblemType"]["Index1"] and kernel["GlobalSplitU"] > 1:
        kStr += "gsuWgIdx"
      else:
        kStr += "wg%s" % self.indexChars[i]
      kStr += ")"
      if i == kernel["ProblemType"]["Index0"]:
        kStr += "*MT%s + localC%s" \
            % (self.tileChar0, self.tileChar0)
      if i == kernel["ProblemType"]["Index1"]:
        kStr += "*MT%s + localC%s" \
            % (self.tileChar1, self.tileChar1)
      kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    kStr = ""
    for b in range(0, kernel["NumVectorsPerThread"]):
      for s in range(0, kernel["VectorWidth"]):
        if kernel["EdgeType"] != "None":
          kStr += "  if (globalC%s%s < size%s) {" \
              % (self.tileChar0, \
              ((" + %u" %s) if kernel["VectorWidth"]>1 else ""), \
              self.tileChar0)
          kStr += "  if (globalC%s + %u*CPSV < size%s) {" \
              % (self.tileChar1, b, self.tileChar1)

        kStr += "  TYPE_MAC_WRITE( C[ GLOBAL_C( (%s)" % self.uint64Str
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          kStr += " globalC%s" % self.indexChars[i]
          if i == kernel["ProblemType"]["Index0"] and kernel["VectorWidth"]>1:
            kStr += " + %u" %s
          if i == kernel["ProblemType"]["Index1"]:
            kStr += " + %u*CPSV" %b
          if i < kernel["ProblemType"]["NumIndicesC"]-1:
            kStr += ", (%s)" % self.uint64Str
        kStr += ") ]"
        kStr += ", alpha"
        kStr += ", rC[%d]%s" % (b, \
            ((".%s"%self.vectorComponents[s]) if kernel["VectorWidth"]>1 \
            else "") )

        if kernel["ProblemType"]["UseBeta"]:
          kStr += ", beta"
        kStr += ")"

        if kernel["EdgeType"] != "None":
          kStr += "} }"
        kStr += self.endLine
    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    kStr = ""
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int globalC" + self.indexChars[i] \
          + " = ("
      if i == kernel["ProblemType"]["Index1"] and kernel["GlobalSplitU"] > 1:
        kStr += "gsuWgIdx"
      else:
        kStr += "wg%s" % self.indexChars[i]
      kStr += ")"
      if i == kernel["ProblemType"]["Index0"]:
        kStr += "*MT%s + (serial %% SG%s)*VECTOR_WIDTH" \
            % (self.tileChar0, self.tileChar0)
      if i == kernel["ProblemType"]["Index1"]:
        kStr += "*MT%s + (serial / SG%s)*VECTOR_WIDTH" \
            % (self.tileChar1, self.tileChar0)
      kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    kStr = ""
    for b in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for a in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for s1 in range(0, kernel["VectorWidth"]):
          for s0 in range(0, kernel["VectorWidth"]):
            if kernel["EdgeType"] == "Branch":
              kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar0, a, self.tileChar0, self.tileChar0)
              kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar1, b, self.tileChar1, self.tileChar1)
            elif kernel["EdgeType"] == "ShiftPtr":
              kStr += "  if (globalC%s%s + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar0, \
                  ((" + %u"%s0) if kernel["VectorWidth"]>1 else ""), \
                  a, self.tileChar0, self.tileChar0)
              kStr += "  if (globalC%s%s + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar1, \
                  ((" + %u"%s1) if kernel["VectorWidth"]>1 else ""), \
                  b, self.tileChar1, self.tileChar1)

            kStr += "  TYPE_MAC_WRITE( C[ GLOBAL_C( (%s)" % self.uint64Str
            for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
              kStr += " globalC%s" % self.indexChars[i]
              if i == kernel["ProblemType"]["Index0"]:
                kStr += "%s + %u*SG%s*VECTOR_WIDTH" % (\
                    ((" + %u"%s0) if kernel["VectorWidth"]>1 else ""), \
                    a, self.tileChar0)
              if i == kernel["ProblemType"]["Index1"]:
                kStr += "%s + %u*SG%s*VECTOR_WIDTH" % (\
                    ((" + %u"%s1) if kernel["VectorWidth"]>1 else ""), \
                    b, self.tileChar1)
              if i < kernel["ProblemType"]["NumIndicesC"]-1:
                kStr += ", (%s)" % self.uint64Str
            kStr += ") ]"
            kStr += ", alpha"
            kStr += ", rC[%d+%d*(TT%s/VECTOR_WIDTH)+%d*TT%s]%s" \
                % (a, s1, self.tileChar0, b, self.tileChar0, \
                ((".%s"%self.vectorComponents[s0]) if kernel["VectorWidth"]>1\
                else "") )
            if kernel["ProblemType"]["UseBeta"]:
              kStr += ", beta"
            kStr += ")"

            if kernel["EdgeType"] != "None":
              kStr += " } }"
            kStr += self.endLine
    return kStr

  ##############################################################################
  # Function End
  ##############################################################################
  def functionEnd(self, kernel):
    kStr = ""
    kStr += self.endLine
    kStr += "}" + self.endLine
    return kStr

  ##############################################################################
  # Function Suffix
  ##############################################################################
  def functionSuffix(self, kernel):
    kStr = ""
    if globalParameters["MergeFiles"] and self.language == "HIP":
      kStr += "#undef UNROLL%s" % self.endLine
      kStr += "#undef LOCAL_SPLITU%s" % self.endLine
      kStr += "#undef LOCAL_DEPTHU%s" % self.endLine
      kStr += "#undef SG%s%s" % (self.tileChar0, self.endLine)
      kStr += "#undef SG%s%s" % (self.tileChar1, self.endLine)
      kStr += "#undef TT%s%s" % (self.tileChar0, self.endLine)
      kStr += "#undef TT%s%s" % (self.tileChar1, self.endLine)
      kStr += "#undef MT%s%s" % (self.tileChar0, self.endLine)
      kStr += "#undef MT%s%s" % (self.tileChar1, self.endLine)
      kStr += "#undef NLCA%s" % (self.endLine )
      kStr += "#undef NLCB%s" % (self.endLine )
      kStr += "#undef NLPA%s" % (self.endLine )
      kStr += "#undef NLPB%s" % (self.endLine )
      kStr += "#undef LSCA%s" % (self.endLine)
      kStr += "#undef LSPA%s" % (self.endLine)
      kStr += "#undef LSCB%s" % (self.endLine)
      kStr += "#undef LSPB%s" % (self.endLine)
      kStr += "#undef GLOBAL_C%s" % (self.endLine)
      kStr += "#undef GLOBAL_OFFSET_A%s" % (self.endLine)
      kStr += "#undef GLOBAL_OFFSET_B%s" % (self.endLine)
      kStr += "#undef DATA_TYPE%s" % (self.endLine)
      kStr += "#undef VECTOR_TYPE%s" % (self.endLine)
      kStr += "#undef LDS_OFFSET_B%s" % (self.endLine)
      kStr += "#undef LDS_OFFSET_BLK%s" % (self.endLine)
      kStr += "#undef LDS_NUM_ELEMENTS%s" % (self.endLine)
      kStr += "#undef NUM_THREADS%s" % (self.endLine)
      kStr += "#undef WORK_GROUP_MAPPING%s" % (self.endLine)
      kStr += "#undef VECTOR_WIDTH%s" % (self.endLine)
      kStr += "#undef TYPE_MAC%s" % (self.endLine)
      kStr += "#undef TYPE_MAC_WRITE%s" % (self.endLine)
      kStr += "#undef GLOBAL_SPLITU%s" % (self.endLine)
      # zero
      if kernel["ProblemType"]["DataType"].isHalf() \
          and kernel["VectorWidth"] > 1 \
          and (kernel["LoopTail"] or kernel["EdgeType"] == "Branch"):
        pass
      elif kernel["ProblemType"]["DataType"].isComplex():
        pass
      else:
        kStr += "#undef VECTOR_ZERO%s" % (self.endLine )
        kStr += "#undef SCALAR_ZERO%s" % (self.endLine )

      numMacs = 2 if kernel["PrefetchLocalRead"] else 1
      for m in range(0, numMacs):
        kStr += "#undef MAC_%ux%u" \
            % (kernel["ThreadTile0"], kernel["ThreadTile1"])
        if kernel["PrefetchLocalRead"]:
          kStr += ("" if m==0 else "_BLK")
        kStr += self.endLine

      firstStride = 0
      if kernel["ProblemType"]["UseInitialStrides"]:
        lastStrideC = 0
        lastStrideA = 0
        lastStrideB = 0
      else:
        lastStrideC = 1
        lastStrideA = 1
        lastStrideB = 1
      for i in range(firstStride, lastStrideC):
        kStr += "#undef strideC" + self.indexChars[i] + self.endLine
      for i in range(firstStride, lastStrideA):
        kStr += "#undef strideA" \
            + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]] \
            + self.endLine
      for i in range(firstStride, lastStrideB):
        kStr += "#undef strideB" \
            + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]] \
            + self.endLine
      kStr += self.endLine + self.endLine
    return kStr

  ##############################################################################
  # Kernel Body Prefix
  ##############################################################################
  def kernelBodyPrefix(self, kernel):
    kStr = ""
    kernelName = self.getKernelName(kernel)
    if not globalParameters["MergeFiles"]:
      kStr += "\n"
      kStr += "#include \"%s.h\"\n" % kernelName
      kStr += "\n"

    return kStr

  ##############################################################################
  # Kernel Body Suffix
  ##############################################################################
  def kernelBodySuffix(self, kernel):
    kStr = ""
    kernelName = self.getKernelName(kernel)

    if self.language == "OCL":
      kStr += "std::string %s_src_concatenated = \n  %s_src_0" \
          % (kernelName, kernelName)
      for i in range(1, self.stringIdx):
        kStr += "\n  + %s_src_%u" % (kernelName, i)
      kStr += ";\n"
      kStr += "const char * const %s_src = %s_src_concatenated.c_str();" \
          % (kernelName, kernelName)

    kStr += "\n"
    return kStr

  ##############################################################################
  # WaitCnt
  ##############################################################################
  def wait(self, kernel, globalRead, localWrite, localRead, comment):
    return ""

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def syncThreads(self, kernel):
    return self.indent + self.syncStr + self.endLine

  ##############################################################################
  #
  #   Beta-Only Kernel
  #
  ##############################################################################

  ##############################################################################
  # Function Signature
  ##############################################################################
  def functionSignatureBetaOnly(self, kernel):
    kernelName = self.getKernelNameBetaOnly(kernel)
    # determine chars for fast access
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.indexChars[kernel["ProblemType"]["Index0"]]
    self.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.indexChars[kernel["ProblemType"]["Index1"]]
    self.tileChar0 = self.indexChars[kernel["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[kernel["ProblemType"]["Index1"]]

    kStr = ""
    # kernel name
    if self.language == "OCL":
      kStr += "__attribute__((reqd_work_group_size(8,8,1)))"
      kStr += self.endLine
      kStr += "__kernel "
    else:
      kStr += "extern \"C\"\n"
      kStr += "__global__ "
    kStr += "void %s" % ( kernelName )
    kStr += "(" + self.endLine
    # pointers
    globalStr = "__global "
    if self.language == "HIP":
      kStr += "  hipLaunchParm lp," + self.endLine
      globalStr = ""
    #restrictStr = "restrict"
    #if self.language == "HIP":
    #  restrictStr = "__restrict__"
    ptrStr = kernel["ProblemType"]["DataType"].toDevice(self.language)
    kStr += "  " + globalStr + ptrStr \
        + " *C,"
    kStr += self.endLine

    # offsets
    kStr += "  unsigned int const offsetC,%s" % (self.endLine)

    # strides
    firstStride = 1
    if kernel["ProblemType"]["UseInitialStrides"]:
      firstStride = 0
    lastStrideC = kernel["ProblemType"]["NumIndicesC"]
    for i in range(firstStride, lastStrideC):
      kStr += "  unsigned int const strideC%s,%s" \
          % (self.indexChars[i], self.endLine)

    # sizes
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int const size%s" % self.indexChars[i]
      if i < kernel["ProblemType"]["NumIndicesC"]-1 \
          or kernel["ProblemType"]["UseBeta"]:
        kStr += ","
      else:
        kStr += ")"
      kStr += self.endLine
    # beta
    if kernel["ProblemType"]["UseBeta"]:
      kStr += "  %s const beta)%s" \
          % (kernel["ProblemType"]["DataType"].toDevice(self.language), \
          self.endLine )
    return kStr

  ##############################################################################
  # Kernel Body Beta-Only
  ##############################################################################
  def kernelBodyBetaOnly(self, kernel):
    kStr = ""
    kStr += "{%s" % self.endLine
    kStr += "  C += offsetC;%s" % self.endLine

    ########################################
    # defined initial strides
    firstStride = 0
    if kernel["ProblemType"]["UseInitialStrides"]:
      # no strides #defined
      lastStrideC = 0
    else:
      # #define initial stride
      kStr += "/* hard-coded initial strides */%s" \
          % self.endLine
      lastStrideC = 1
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + self.indexChars[i] + " 1" + self.endLine

    ########################################
    # GLOBAL_C()
    kStr += "#define GLOBAL_C(IDX%s" % self.indexChars[0]
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    ########################################
    # wg d0, d1
    #kStr += "  unsigned int wg" + self.tileChar0 + " = " \
    #    + self.getGroupIdStr + "(0);" + self.endLine
    #kStr += "  unsigned int wg" + self.tileChar1 + " = " \
    #    + self.getGroupIdStr + "(1);" + self.endLine
    ########################################
    # wg other
    nonTileFreeIndices = range(0, kernel["ProblemType"]["NumIndicesC"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index0"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index1"])
    for i in range(0, len(nonTileFreeIndices)):
      index = nonTileFreeIndices[i]
      kStr += "  unsigned int wg" + self.indexChars[index] \
          + " = ( " + self.getGroupIdStr + "(2)"
      for j in reversed( range( i+1, len(nonTileFreeIndices)) ):
        index2 = nonTileFreeIndices[j]
        kStr += " / size" + self.indexChars[index2]
      kStr += " ) % size" + self.indexChars[index] + ";" + self.endLine

    ########################################
    # C indices
    #kStr += "  unsigned int serial = %s(0);%s" \
    #    % (self.getLocalIdStr, self.endLine)
    # wg=8x8
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int globalC%s = " % self.indexChars[i]
      if i == kernel["ProblemType"]["Index0"]:
        kStr += "%s(%u)" % (self.getGlobalIdStr, \
            kernel["ProblemType"]["Index0"])
      elif i == kernel["ProblemType"]["Index1"]:
        kStr += "%s(%u)" % (self.getGlobalIdStr, \
            kernel["ProblemType"]["Index1"])
      else:
        kStr += "wg%s" % self.indexChars[i]
      kStr += ";" + self.endLine

    ########################################
    # C index
    kStr += "  %s idx = GLOBAL_C( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += " globalC%s" % self.indexChars[i]
      if i < kernel["ProblemType"]["NumIndicesC"]-1:
        kStr += ", "
    kStr += ");%s" % (self.endLine)
    #kStr += "printf(\\\"%%09llu\\\\n\\\", idx);%s" % (self.endLine)
    kStr += "  if (globalC%s < size%s && globalC%s < size%s) {%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar1, self.tileChar1, \
        self.endLine )

    ########################################
    # zero
    if kernel["ProblemType"]["DataType"].isComplex():
      kStr += "  DATA_TYPE SCALAR_ZERO;%s" % ( self.endLine )
      kStr += "  SCALAR_ZERO.s0 = 0;%s" % self.endLine
      kStr += "  SCALAR_ZERO.s1 = 0;%s" % self.endLine
    else:
      kStr += "#define SCALAR_ZERO %s%s" % ( kernel["ProblemType"][\
         "DataType"].zeroString(self.language, 1), \
         self.endLine )

    ########################################
    # zero
    kStr += "    C[idx]"
    if kernel["ProblemType"]["UseBeta"]:
      kStr += " *= beta;%s" % self.endLine
    else:
      kStr += " = SCALAR_ZERO;%s" % (self.endLine)
    kStr += "  }%s" % self.endLine

    ########################################
    # end
    kStr += "}%s" % self.endLine
    return kStr
