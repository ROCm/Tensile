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


from SolutionStructs import DataType, isPackedIndex
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
    self.language = globalParameters["RuntimeLanguage"]

    if self.language == "OCL":
      # everything escaped extra b/c string
      self.endLine = "\\n\"\n\""
      self.endLinePP = "\\\\" + self.endLine
      self.quote = "\\\""
      self.endLineQuote = "\\\\n\\\""
    else:
      self.endLine = "\n"
      self.endLinePP =  "\\" + self.endLine
      self.quote = "\""
      self.endLineQuote = "\\n\""

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
    self.useMagicNumber = True
    self.db={}
    self.db["PrintStagger"] = 0

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
  def initKernel(self, kernel, tPA, tPB ):
    super(KernelWriterSource, self).initKernel( kernel, tPA, tPB )
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

    kStr += self.endLine

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
    kStr += "#define VECTOR_WIDTH %u%s" % (kernel["VectorWidth"], self.endLine)
    kStr += "#define GLOBAL_LOAD_VECTOR_WIDTH_A %u%s" \
        % (kernel["GlobalLoadVectorWidthA"], self.endLine)
    kStr += "#define GLOBAL_LOAD_VECTOR_WIDTH_B %u%s" \
        % (kernel["GlobalLoadVectorWidthB"], self.endLine)
    kStr += "#define GLOBAL_WRITE_VECTOR_WIDTH %u%s" \
        % (kernel["GlobalWriteVectorWidth"], self.endLine)
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
    kStr += "#define PAD %u%s" % (kernel["LdsPadA"], self.endLine)  # TODO - ignore LdsPadB
    kStr += "#define WORK_GROUP_MAPPING %u%s" \
        % (abs(kernel["WorkGroupMapping"]), self.endLine)
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
    kStr += "#define LVCA (LSCA/GLOBAL_LOAD_VECTOR_WIDTH_A)%s" % (self.endLine)
    kStr += "#define LVCB (LSCB/GLOBAL_LOAD_VECTOR_WIDTH_B)%s" % (self.endLine)
    kStr += "#define LVPA (LSPA/GLOBAL_LOAD_VECTOR_WIDTH_A)%s" % (self.endLine)
    kStr += "#define LVPB (LSPB/GLOBAL_LOAD_VECTOR_WIDTH_B)%s" % (self.endLine)


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
    if kernel["WorkGroupMappingType"] == "Z":
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
    # D
    kStr += "#define GLOBAL_D(IDX%s" % self.indexChars[0]
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideD%s" % (indexChar, indexChar)
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideD%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine
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
    kStr += "#define DEST_DATA_TYPE %s%s" \
        % (kernel["ProblemType"]["DestDataType"].toDevice(self.language), \
        self.endLine)
    kStr += "#define COMPUTE_DATA_TYPE %s%s" \
        % (kernel["ProblemType"]["ComputeDataType"].toDevice(self.language), \
        self.endLine)
    #vecStr = kernel["ProblemType"]["DataType"].toDevice(self.language)
    #if kernel["VectorWidth"] > 1:
    #  vecStr += str(kernel["VectorWidth"])
    #kStr += "#define VECTOR_TYPE %s%s" % (vecStr, self.endLine)

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
        kStr += "template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>%s" % (self.endLine)
        kStr += "__device__ inline void atomicAddType(%s%sT *fPtr, T operand) {%s" \
            % (self.volatileStr, self.globalPtrStr, self.endLine)
        kStr += "  std::atomic<T> *aPtr = reinterpret_cast<std::atomic<T>*>(fPtr);%s" % (self.endLine)
        kStr += "  T oldValue, newValue;%s" % (self.endLine)
        kStr += "  oldValue = aPtr->load(std::memory_order_relaxed);%s" % (self.endLine)
        kStr += "  do {%s" % (self.endLine)
        kStr += "    newValue = oldValue + operand;%s" % (self.endLine)
        #kStr += "    prevReturn = %s(uPtr, prevVal.ui, newVal.ui);%s" \
        #    % (self.atomicCasStr, self.endLine)
        #kStr += "  } while ( !std::atomic_compare_exchange_weak_explicit(aPtr, &oldValue, newValue, std::memory_order_acq_rel, std::memory_order_release) );%s" % (self.endLine)
        kStr += "  } while ( !std::atomic_compare_exchange_weak_explicit(aPtr, &oldValue, newValue, std::memory_order_relaxed, std::memory_order_release) );%s" % (self.endLine)
        kStr += "}%s" % (self.endLine)

      kStr += "#endif%s" % self.endLine

    kStr += "#define MAGIC_DIV(dividend, magicNumber, magicShift) ((uint64_t)(dividend) * magicNumber >> magicShift)%s" % self.endLine

    ####################################
    # MACs
    kStr += self.endLine
    kStr += "/* MAC's */" + self.endLine

    if self.language == "OCL":
      kStr += "#define MAC(A,B,DST) mad(A,B,DST)"
    else:
      if kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["ProblemType"]["DataType"].isHalf():
        kStr += "#define MAC(A,B,DST) DST += static_cast<float>(A) * static_cast<float>(B)" 
      elif kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["ProblemType"]["DataType"].isInt8x4():
        kStr += "#define MAC(A,B,DST) DST = GenDot4(static_cast<int>(A), static_cast<int>(B), static_cast<int>(DST))"
      elif kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["ProblemType"]["DataType"].isBFloat16():
        kStr += "#define MAC(A,B,DST) DST = static_cast<tensile_bfloat16>(1.0f);"
      else:
        kStr += "#define MAC(A,B,DST) DST += A*B" 
    kStr += self.endLine

    if kernel["ProblemType"]["DataType"].isReal():
      # real data

      if ((kernel["ThreadTileA"] % 2 == 0) and (kernel["ProblemType"]["DataType"].isHalf())):
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          kStr += "#define TYPE_MAC(MULA0,MULB0,DST0,MULA1,MULB1,DST1) " + self.endLinePP
          kStr += " DST0 = MAC(MULA0,MULB0,DST0);" + self.endLinePP
          kStr += " DST1 = MAC(MULA1,MULB1,DST1);" + self.endLinePP
          kStr += self.endLine
        else:
          kStr += "#define TYPE_MAC(MULA0,MULB0,DST0,MULA1,MULB1,DST1) " + self.endLinePP
          kStr += "  a_pk_fma[0] = MULA0; %s " % (self.endLinePP)
          kStr += " a_pk_fma[1] = MULA1; %s " % (self.endLinePP)
          kStr += " b_pk_fma[0] = MULB0; %s " % (self.endLinePP)
          kStr += " b_pk_fma[1] = MULB1; %s " % (self.endLinePP)
          kStr += " c_pk_fma[0] = DST0; %s " % (self.endLinePP)
          kStr += " c_pk_fma[1] = DST1; %s " % (self.endLinePP)
          kStr += " c_pk_fma = tensile_fmadd_half2(a_pk_fma, b_pk_fma, c_pk_fma); %s " % (self.endLinePP)
          kStr += " DST0 = c_pk_fma[0]; %s " % (self.endLinePP)
          kStr += " DST1 = c_pk_fma[1]; %s " % (self.endLinePP)
          kStr += self.endLine
      else:
        kStr += "#define TYPE_MAC(MULA,MULB,DST) " \
            + "DST = MAC(MULA,MULB,DST);" + self.endLine

      # GSU
      if kernel["GlobalSplitU"] > 1: # 1st kernel will have taken care of B
        if kernel["ProblemType"]["UseBeta"]:
          kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) atomicAddType(&(DST), (ALPHA)*(REG));"
        else:
          kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) atomicAddType(&(DST), (ALPHA)*(REG));"

      else:
        if kernel["ProblemType"]["UseBeta"]:
          # dst = alpha*reg + dst*beta
          if kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["ProblemType"]["DataType"].isBFloat16():
            kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) DST = static_cast<tensile_bfloat16>(1.0f);" + self.endLine
          else:
            kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) " \
              + "DST = 0 != (BETA) ? (ALPHA)*(REG) + (BETA)*(SRC) : (ALPHA)*(REG);" + self.endLine
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
          "#define TYPE_MAC_WRITE( DST, SRC, ALPHA, REG, BETA ) "+self.endLinePP +
          "  /* (1) */ " + self.endLinePP +
          "  type_mac_tmp = REG.s0; " + self.endLinePP +
          "  REG.s0 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s0 = MAC( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLinePP +
          "  REG.s1 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s1 = MAC(  ALPHA.s1, type_mac_tmp, REG.s1 ); "+self.endLinePP+
          "  /* (2) */ " + self.endLinePP +
          "  if(BETA.s0 != 0) { " + self.endLinePP +
          "  REG.s0 = MAC(  BETA.s0, SRC.s0, REG.s0 ); " + self.endLinePP +
          "  REG.s1 = MAC(  BETA.s0, SRC.s1, REG.s1 ); " + self.endLinePP +
          "  } " + self.endLinePP +
          "  if (BETA.s1 != 0) { " + self.endLinePP +
          "  REG.s0 = MAC( -BETA.s1, SRC.s1, REG.s0 ); " + self.endLinePP +
          "  REG.s1 = MAC(  BETA.s1, SRC.s0, REG.s1 ); " + self.endLinePP +
          "  } " + self.endLinePP +
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
          strC = "rC[%d+%d*TT%s]" % (a, b, self.tileChar0 )
          strA = "rA[%d%s]" % (a, ("+TT%s"%self.tileCharA) if m>0 else "")
          strB = "rB[%d%s]" % (b, ("+TT%s"%self.tileCharB) if m>0 else "")
          if ((kernel["ThreadTileA"] % 2 == 0) and (kernel["ProblemType"]["DataType"].isHalf())):
            if a % 2 == 0:
              kStr += "  TYPE_MAC(%s,%s,%s , " % (strA, strB, strC)
            else:
              kStr += "%s,%s,%s); %s" % (strA, strB, strC, self.endLinePP)
          else:
            kStr += "  TYPE_MAC(%s,%s,%s); %s" % (strA, strB, strC, \
                self.endLinePP)

      if kernel["UnrollMemFence"]:
        kStr += "  " + self.fenceStr
      kStr += self.endLine

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
          #kStr += "  printf(\\\"T[%%u,%u,%u]: %s:%%.0f += %s:%%.0f * %s:%%.0f\\\\n\\\", serial, %s, %s, %s); %s" % (a, b, strC, strA, strB, strC, strA, strB, self.endLinePP)
          kStr += "  TYPE_MAC(%s,%s,%s); %s" % (strA, strB, strC, \
              self.endLinePP)
      if kernel["UnrollMemFence"]:
        kStr += "  " + self.fenceStr
      kStr += self.endLine
      """

    ####################################
    # preprocessor definitions of kernel arguments
    firstStride = 0
    if kernel["ProblemType"]["UseInitialStrides"]:
      # no strides #defined
      lastStrideD = 0
      lastStrideC = 0
      lastStrideA = 0
      lastStrideB = 0
    else:
      # #define initial stride
      kStr += "/* hard-coded initial strides */%s" \
          % self.endLine
      lastStrideD = 1
      lastStrideC = 1
      lastStrideA = 1
      lastStrideB = 1

    for i in range(firstStride, lastStrideD):
      kStr += "#define strideD" + self.indexChars[i] + " 1" + self.endLine
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
      #s += "  hipLaunchParm lp," + self.endLine
      globalStr = ""
    restrictStr = "restrict"
    if self.language == "HIP":
      restrictStr = "__restrict__"
    ptrStr = kernel["ProblemType"]["DestDataType"].toDevice(self.language)
    s += "  " + globalStr + ptrStr \
        + " *D,"
    s += self.endLine
    ptrStr = kernel["ProblemType"]["DestDataType"].toDevice(self.language)
    s += "  " + globalStr + ptrStr \
        + " const * " + restrictStr + " C,"
    s += self.endLine
    ptrStr = kernel["ProblemType"]["DataType"].toDevice(self.language)
    s += "  " + globalStr + ptrStr \
        + " const * " + restrictStr + " A,"
    s += self.endLine
    s += "  " + globalStr + ptrStr \
        + " const * " + restrictStr + " B"

    # alpha & beta
    s += "," + self.endLine + "  " \
        + kernel["ProblemType"]["ComputeDataType"].toDevice(self.language) + " const alpha"
    if kernel["ProblemType"]["UseBeta"]:
      s += "," + self.endLine + "  " \
          + kernel["ProblemType"]["ComputeDataType"].toDevice(self.language) + " const beta"

    # strides
    firstStride = 1
    if kernel["ProblemType"]["UseInitialStrides"]:
      firstStride = 0
    lastStrideD = kernel["ProblemType"]["NumIndicesC"]
    lastStrideC = kernel["ProblemType"]["NumIndicesC"]
    lastStrideA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    lastStrideB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    for i in range(firstStride, lastStrideD):
      s += "," + self.endLine + "  unsigned int const strideD" + self.indexChars[i]
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

    for idxChar in kernel["PackedC0Indices"][:-1]:
      s += ",%s  unsigned magicNumberSize%s" % (self.endLine, idxChar)
      s += ",%s  unsigned magicShiftSize%s" % (self.endLine, idxChar)
    for idxChar in kernel["PackedC1Indices"][:-1]:
      s += ",%s  unsigned magicNumberSize%s" % (self.endLine, idxChar)
      s += ",%s  unsigned magicShiftSize%s" % (self.endLine, idxChar)
    s += "," + self.endLine + "  unsigned int staggerUIterParm"

    # kernel["PersistentKernel"]:
    s += "," + self.endLine + "  unsigned int problemNumGroupTiles0"
    s += "," + self.endLine + "  unsigned int problemNumGroupTiles1"
    s += "," + self.endLine + "  unsigned int magicNumberProblemNumGroupTiles0"
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

    kStr += "  unsigned int serial = %s(0);%s" \
        % (self.getLocalIdStr, self.endLine)
    kStr += "  unsigned int sgId = serial / (SG%s*SG%s);%s" \
        % (self.tileChar0, self.tileChar1, self.endLine)

    ####################################
    # zero
    if kernel["ProblemType"]["DataType"].isHalf() \
        and kernel["VectorWidth"] > 1 \
        and (kernel["LoopTail"] or kernel["EdgeType"] == "Branch"):
      kStr += "#define SCALAR_ZERO 0%s" % self.endLine
    elif kernel["ProblemType"]["DataType"].isComplex():
      kStr += "  DATA_TYPE SCALAR_ZERO;%s" % ( self.endLine )
      kStr += "  SCALAR_ZERO.s0 = 0;%s" % self.endLine
      kStr += "  SCALAR_ZERO.s1 = 0;%s" % self.endLine
    elif kernel["ProblemType"]["DestDataType"].isBFloat16():
      kStr += "#define SCALAR_ZERO static_cast<tensile_bfloat16>(0.0f)%s" % self.endLine
    else:
      kStr += "#define SCALAR_ZERO %s%s" % ( kernel["ProblemType"][\
         "DataType"].zeroString(self.language, 1), \
         self.endLine )

    # TODO - use a different value for OOB data
    #        Currently use zero since Tensile already has handy functions to create zero in different types
    kStr += "#define SCALAR_OOB_DATA SCALAR_ZERO%s" % self.endLine

    kStr += "  /* registers for MAC's */" + self.endLine
    if kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["ProblemType"]["DataType"].isHalf():
        kStr += "  float rC[TT%s*TT%s];%s" \
            % (self.tileChar0, self.tileChar1, self.endLine )
    else:
        kStr += "  DEST_DATA_TYPE rC[TT%s*TT%s];%s" \
            % (self.tileChar0, self.tileChar1, self.endLine )

    # registers for valuAB
    kStr += "  DATA_TYPE rA[TT%s%s];%s" \
        % (self.tileChar0, ("*2" if kernel["PrefetchLocalRead"] else ""), \
        self.endLine)
    kStr += "  DATA_TYPE rB[TT%s%s];%s" \
        % (self.tileChar1, ("*2" if kernel["PrefetchLocalRead"] else ""), \
        self.endLine)

    ####################################
    # registers for global -> local load
    kStr += self.endLine
    kStr += "  /* registers for global->local */%s" % self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for sPerp in range(0, self.numReadsPerpVecCompA):
        for para in range(0, kernel["NumLoadsCoalescedA"]):
          for sPara in range(0, self.numReadsCoalVecCompA):
            kStr += "  DATA_TYPE a_%u_%u_%u_%u;%s" \
                % (para, sPara, perp, sPerp, self.endLine)
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for sPerp in range(0, self.numReadsPerpVecCompB):
        for para in range(0, kernel["NumLoadsCoalescedB"]):
          for sPara in range(0, self.numReadsCoalVecCompB):
            kStr += "  DATA_TYPE b_%u_%u_%u_%u;%s" \
                % (para, sPara, perp, sPerp, self.endLine)
    """
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
    """

    ####################################
    # allocate tensile_half2 memory
    if kernel["ProblemType"]["DataType"].isHalf():
      kStr += self.endLine
      kStr += "  /* allocate tensile_half2 memory */" + self.endLine
      kStr += "  tensile_half2 a_pk_fma;" + self.endLine
      kStr += "  tensile_half2 b_pk_fma;" + self.endLine
      kStr += "  tensile_half2 c_pk_fma;" + self.endLine

    ####################################
    # allocate local memory
    kStr += self.endLine
    kStr += "  /* allocate local memory */" + self.endLine
    kStr += "  %sDATA_TYPE localMemory[LDS_NUM_ELEMENTS];%s" \
        % (self.sharedDeclStr, self.endLine )

    return kStr

  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  def graWorkGroup(self, kernel):
    kStr = ""

    wg0 = "wg%s" % self.tileChar0
    wg1 = "wg%s" % self.tileChar1
    nwgg = kernel["WorkGroupMapping"] >= 0
    n0 = 0 if nwgg else 1
    n1 = 1 if nwgg else 0

    if kernel["PersistentKernel"]:
      kStr += "  %s serialWgIter = %s(0);%s" \
        % (self.uint64Str, self.getGroupIdStr, self.endLine)
      kStr += "  unsigned int n%s = problemNumGroupTiles0;%s" \
          % ( wg0, self.endLine)
      kStr += "  unsigned int n%s = problemNumGroupTiles1;%s" \
          % ( wg1, self.endLine)
      kStr += "  unsigned int %s;%s" % ( wg0, self.endLine)
      kStr += "  unsigned int %s;%s" % ( wg1, self.endLine)

      #kStr += "if (serial==0) printf(\"WG%%u_%%u probWG:%%u_%%u  %s\", hc_get_group_id(0), hc_get_group_id(1), %s, %s);" % (self.endLinePP, wg0, wg1)+ self.endLine
      kStr += "%swhile (1) { // persistent loop %s" % (self.endLine, self.endLine)
      kStr += "  %s  = serialWgIter %% problemNumGroupTiles0;%s" \
          % ( wg0, self.endLine)
      kStr += "  %s  = serialWgIter / problemNumGroupTiles0;%s" \
          % ( wg1, self.endLine)
    else:
      # optionally transpose work-group grid
      kStr += "  unsigned int %s = %s(%u);%s" \
          % ( wg0, self.getGroupIdStr, n0, self.endLine)
      kStr += "  unsigned int %s = %s(%u);%s" \
          % ( wg1, self.getGroupIdStr, n1, self.endLine)
      kStr += "  unsigned int n%s = %s(%u);%s" \
          % ( wg0, self.getNumGroupsStr, n0, self.endLine)
      kStr += "  unsigned int n%s = %s(%u);%s" \
          % ( wg1, self.getNumGroupsStr, n1, self.endLine)
      if kernel["GlobalSplitU"] > 1:
        kStr += "  n%s /= GLOBAL_SPLITU;%s" % (wg1, self.endLine)

    # split up work-group grid
    if kernel["GlobalSplitU"] > 1:
      kStr += "  unsigned int gsuSumIdx;%s" % self.endLine
      if kernel["GlobalSplitUWorkGroupMappingRoundRobin"]:
        kStr += "  gsuSumIdx = %s / n%s;%s" \
            % (wg1, wg1, self.endLine)
        kStr += "  %s = %s %% n%s;%s" \
            % (wg1, wg1, wg1, self.endLine)
      else:
        kStr += "  gsuSumIdx = %s %% GLOBAL_SPLITU;%s" \
            % (wg1, self.endLine)
        kStr += "  %s = %s / GLOBAL_SPLITU;%s" \
            % (wg1, wg1, self.endLine)

      ########################################
      # Blocked rows or columns
    if kernel["WorkGroupMappingType"] == "B" and abs(kernel["WorkGroupMapping"]) > 1:
      kStr += self.endLine
      kStr += "  %s wgSerial = %s + (%s %% WORK_GROUP_MAPPING) * n%s;// within block%s" \
        % (self.uint64Str, wg0, wg1, wg0, self.endLine)
      kStr += "  unsigned int block = %s / WORK_GROUP_MAPPING;%s" \
          % (wg1, self.endLine );
      kStr += "  unsigned int blockRemainder = (%s < n%s-(n%s %% WORK_GROUP_MAPPING) ) ? 0 : n%s %% WORK_GROUP_MAPPING;%s" % \
          ( wg1, wg1, wg1, wg1, self.endLine )
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
            % (wg0, blockWidth, self.endLine)
        kStr += "    %s = wgSerial %% %u + block*WORK_GROUP_MAPPING;%s" \
            % (wg1, blockWidth, self.endLine)
        kStr += "  }"
      kStr += "%s" % self.endLine


    ########################################
    # Generalized Z-Order
    elif kernel["WorkGroupMappingType"] == "Z":

      kStr += "  unsigned int nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
      kStr += "  unsigned int nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)

      if abs(kernel["WorkGroupMapping"]) == 1: # Generalized Z-Order
        kStr += "  generalized_z_order(&%s, &%s, %s, %s, 0, nwg0, nwg1);%s" \
            % ( wg0, wg1, wg0, wg1, self.endLine)

      elif abs(kernel["WorkGroupMapping"]) == 2: # Z-Order round up and return early
        kStr += "  unsigned int wgSerial = %s + %s * n%s;%s" % (wg0, wg1, wg0 if nwgg else wg1, self.endLine)
        kStr += "  z_order(&%s, &%s, wgSerial);%s" % (wg0, wg1, self.endLine)
        kStr += "  if (%s >= nwg0 || %s >= nwg1) return; // wg mapped out of bounds after z-ordering%s" \
            % (wg0, wg1, self.endLine)
      else:
        printExit("WorkGroupMappingType=Z and WorkGroupMapping=%u not supported"%kernel["WorkGroupMapping"])

    #kStr += "if (serial==0) printf(\"WG:%%u_%%u progWG:%%u_%%u \\n\", hc_get_group_id(0), hc_get_group_id(1), %s, %s);" \
    #      % (wg0, wg1)+ self.endLine

    if kernel["PersistentKernel"]:
      kStr += "  if ((%s >= n%s) || (%s >= n%s)) break; // persistent loop%s" \
        % (wg1, wg1, wg0, wg0, self.endLine)
      #kStr += "if (serial==0) printf(\"WG%%u_%%u probWG:%%u_%%u  probNumWG=%%u_%%u\\n%s\", hc_get_group_id(0), hc_get_group_id(1), %s, %s, problemNumGroupTiles0, problemNumGroupTiles1);" % (self.endLinePP, wg0, wg1)+ self.endLine
    return kStr


  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  ##############################################################################
  def graTileAssignment(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int globalReadOffset%s%s = (serial%s" \
        % (tP["tensorChar"], tP["tileChar"], ("%" if tP["grcg"] == tP["tlu"] else "/") )
    if tP["grcg"]:
      kStr += (tP["lvc"] if tP["grcv"] else tP["lsc"])
    else:
      kStr += (tP["lsp"] if tP["grcv"] else tP["lvp"])
    kStr += ")"
    if tP["grcv"] == tP["tlu"]:
      kStr += "*GLOBAL_LOAD_VECTOR_WIDTH_%s" % tP["tensorChar"]
    kStr += " + ("
    kStr += "wg%s" % (tP["tileChar"])
    kStr += ")*MT%s;%s" % (tP["tileChar"], self.endLine)
    return kStr


  ##############################################################################
  # Global Read Addresses: Unroll Assignment A/B
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int globalReadOffset%s%s = (serial%s" \
        % (tP["tensorChar"], self.unrollChar, ("/" if tP["grcg"] == tP["tlu"] else "%") )
    if tP["grcg"]:
      kStr += (tP["lvc"] if tP["grcv"] else tP["lsc"])
    else:
      kStr += (tP["lsp"] if tP["grcv"] else tP["lvp"])
    kStr += ")"
    if tP["grcv"] != tP["tlu"]:
      kStr += "*GLOBAL_LOAD_VECTOR_WIDTH_%s"% tP["tensorChar"]
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
    # packed free dims don't use 'wg' level vars for dims
    nonTileFreeIndices = range(0, kernel["ProblemType"]["NumIndicesC"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index0"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index1"])
    for i in range(0, len(nonTileFreeIndices)):
      index = nonTileFreeIndices[i]
      if isPackedIndex(kernel, index):
        continue
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
      index = i + kernel["ProblemType"]["NumIndicesC"]
      kStr += "#define globalReadOffsetA%s 0%s" \
          % (self.indexChars[index], self.endLine)
      kStr += "#define globalReadOffsetB%s 0%s" \
          % (self.indexChars[index], self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  def graTileOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    for l in range(0, tP["nrt"]):
      for s in range(0, 1 if tP["rc"] else tP["nrtv"]):
        firstGro = gro = "globalReadOffset%s%s_%u_%u" % (tc, tP["tileChar"], l, s)
        kStr += "  unsigned int %s = globalReadOffset%s%s + %u + %d*%s;%s" \
            % (gro, tc, tP["tileChar"], s, l, \
            (tP["lsc"] if tP["tlu"] else tP["lsp"]), \
            self.endLine)

        # Create additional vector address components for any packed dimensions
        lastGro = gro
        lastIdx = tP["idx"]
        dimXStr = ""
        tP["packedSizeList"] = ["size%s" % self.indexChars[lastIdx]]
        for idx in kernel["ProblemType"]["IndexAssignments%s"%tc]:
          if idx < kernel["ProblemType"]["NumIndicesC"] \
              and idx != tP["idx"]:  # tile index
            gro = None
            if isPackedIndex(kernel,idx, tP["PackBatchDims"]):
              gro = "globalReadOffset%s%s_%u_%u" % (tc, self.indexChars[idx], l, s)

            if gro != None:
              tP["packedSizeList"].append("size%s"%self.indexChars[idx])

              if self.useMagicNumber:
                c = globalParameters["IndexChars"][lastIdx]
                dimXStr += "  unsigned int %s = MAGIC_DIV(%s, magicNumberSize%s, magicShiftSize%s);%s" \
                        % (gro, lastGro, c, c, self.endLine)
                dimXStr += "  %s -= (%s*size%s);%s" \
                    % (lastGro, gro, self.indexChars[lastIdx], self.endLine)
              else:
                dimXStr += "  unsigned int %s = %s / size%s; // extract packed index%s" \
                        % (gro, lastGro, self.indexChars[lastIdx], self.endLine)
                dimXStr += "  %s %%= size%s;%s" % (lastGro, self.indexChars[lastIdx], self.endLine)
              #dimXStr += "printf(\"gro: serial:%%u wg0:%%u wg1:%%u %s:%%u\\n\", serial, wg0I, wg1J, %s);%s" % (gro, gro, self.endLine)
              lastGro = gro
              lastIdx = idx

          # Add check and then dimension extraction code, if we used any packed dims above:
          if dimXStr != "":
            sizeStr = " * ".join(tP["packedSizeList"])
            if 0 and tP["isA"]:
              kStr += "printf(\"gro-0: serial:%%u wg0:%%u wg1:%%u globalReadOffsetA0I_0_0:%%u\\n\", serial, wg0I, wg1J, globalReadOffsetA0I_0_0);%s" \
                      % (self.endLine)
            if 0 and tP["isB"]:
              kStr += "printf(\"gro-0: serial:%%u wg0:%%u wg1:%%u globalReadOffsetA0J_0_0:%%u\\n\", serial, wg0I, wg1J, globalReadOffsetA0J_0_0);%s" \
                      % (self.endLine)
            kStr += "  %s = (%s > (%s-1)) ? (%s-1):%s;%s" \
                % (firstGro, firstGro, sizeStr, sizeStr, firstGro, self.endLine)
            kStr += dimXStr;

    if 0 and tP["isB"]:
      kStr += "printf(\"gro-1: serial:%%u wg0:%%u wg1:%%u globalReadOffsetA0I_0_0:%%u globalReadOffsetB1J_0_0:%%u\\n\", serial, wg0I, wg1J, globalReadOffsetA0I_0_0, globalReadOffsetB1J_0_0);%s" \
        % (self.endLine)

    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  def graUnrollOffsets(self, kernel, tP):
    kStr = ""
    for l in range(0, tP["nru"]):
      for s in range(0, 1 if tP["rc"] else kernel["VectorWidth"]):
        kStr += "  unsigned int globalReadOffset%s%s_%u_%u = globalReadOffset%s%s + %u + %d*%s;%s" \
            % (tP["tensorChar"], self.unrollChar, l, s, \
            tP["tensorChar"], self.unrollChar, s, l, \
            (tP["lsp"] if tP["tlu"] else tP["lsc"]), \
            self.endLine)
      #else:
      #  kStr += "  unsigned int globalReadOffset%s%s_%u = globalReadOffset%s%s + %d*%s;%s" \
      #      % (tP["tensorChar"], self.unrollChar, l, tP["tensorChar"], self.unrollChar, l, \
      #      (tP["lsp"] if tP["tlu"] else tP["lsc"]), \
      #      self.endLine)
    return kStr


  ##############################################################################
  # Global Read Addresses: Branch A/B - TODO
  ##############################################################################
  def graBranch(self, kernel, tP):
    kStr = ""
    for l in range(0, tP["nrt"]):
      gro = "(globalReadOffset%s%s_%u_0%s)" \
          % (tP["tensorChar"], tP["tileChar"], l, \
          (" + (VECTOR_WIDTH-1)" if tP["rtc"] else "") )
      limit = "size%s" % (tP["tileChar"])
      kStr += "  bool inBounds%s_%u = %s < %s;%s" \
          % (tP["tensorChar"], l, gro, limit, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Shift A/B
  ##############################################################################
  def graShift(self, kernel, tP):
    kStr = ""
    for l in range(0, tP["nrt"]):
      for s in range(0, 1 if tP["rc"] else tP["nrtv"]):
        #gro = "globalReadOffset%s%s_%u_%u" \
        #    % (tP["tensorChar"], tP["tileChar"], l, s )

        #limit = "(size%s-GLOBAL_LOAD_VECTOR_WIDTH_%s)" % (tP["tileChar"], tP["tensorChar"] )

        #kStr += "  %s = (%s > %s) ? %s+%u : %s;%s" \
        #    % (gro, gro, limit, limit, s, gro, self.endLine)

        kStr += "  globalReadOffset%s%s_%u_%u" \
            % (tP["tensorChar"], tP["tileChar"], l, s )
        kStr += " = ("
        kStr += "  globalReadOffset%s%s_%u_%u" \
            % (tP["tensorChar"], tP["tileChar"], l, s )
        kStr += " > "
        kStr += "size%s-%s" % (tP["tileChar"], "GLOBAL_LOAD_VECTOR_WIDTH_%s+%u"%(tP["tensorChar"], s) if tP["rtv"] else "1")
        kStr += ") ? "
        kStr += "size%s-%s" % (tP["tileChar"], "GLOBAL_LOAD_VECTOR_WIDTH_%s+%u"%(tP["tensorChar"], s) if tP["rtv"] else "1")
        kStr += " : "
        kStr += "globalReadOffset%s%s_%u_%u" \
            % (tP["tensorChar"], tP["tileChar"], l, s )
        kStr += ";%s" % self.endLine

    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  def graFinalOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    problemType = kernel["ProblemType"]
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            # Pass parms to GLOBAL_OFFSET_ macro:
            gro = "globalReadOffset%s_%u_%u_%u_%u" \
                  % (tP["tensorChar"], para, sPara, perp, sPerp)

            kStr += "  %s %s = GLOBAL_OFFSET_%s( " \
                % (self.uint64Str, gro, tP["tensorChar"])
            for i in range(0, len(tP["ia"])):
              index = tP["ia"][i]
              if index < kernel["ProblemType"]["NumIndicesC"]:
                if index == tP["tileIdx"]:
                  kStr += "globalReadOffset%s%s_%u_%u" \
                      % (tP["tensorChar"], tP["tileChar"], \
                      (para if tP["tlu"] else perp), \
                      (sPara if tP["tlu"] else sPerp) )
                else:
                  if isPackedIndex(kernel, index):
                    # pass vector per-tensor-dim offset (rather than scalar wg)
                    if index in problemType["IndicesBatch"] and not tP["PackBatchDims"]:
                      # pass 0, this is is the non-packed batch dim and must be 0
                      kStr += "0"
                    else:
                      kStr += "globalReadOffset%s%s_%u_%u" \
                          % (tc, \
                          self.indexChars[index],
                          (para if tP["tlu"] else perp), \
                          (sPara if tP["tlu"] else sPerp) )
                  else:
                    # just a non-vector group index
                    kStr += "wg" + self.indexChars[index]
              else: # summation index
                if index == kernel["ProblemType"]["IndexUnroll"]:
                  kStr += "globalReadOffset%s%s_%u_%u" \
                      % (tP["tensorChar"], self.unrollChar, \
                      (perp if tP["tlu"] else para), \
                      (sPerp if tP["tlu"] else sPara) )
                else:
                  kStr += "globalReadOffset%s%s" \
                      % (tP["tensorChar"], self.indexChars[index])
              if i < len(tP["ia"])-1:
                kStr += ", "
            kStr += " );%s" % self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  def graAddresses(self, kernel, tP):
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            kStr += "  %sDATA_TYPE const *globalRead%s_%u_%u_%u_%u = %s + globalReadOffset%s_%u_%u_%u_%u;%s" \
                % (self.globalPtrStr, tP["tensorChar"], \
                para, sPara, perp, sPerp, \
                tP["tensorChar"], tP["tensorChar"], \
                para, sPara, perp, sPerp, \
                self.endLine)
        #else:
        #    kStr += "  %sVECTOR_TYPE const *globalRead%s_%u_%u = (%sVECTOR_TYPE const *)(%s + globalReadOffset%s_%u_%u);%s" \
        #        % (self.globalPtrStr, tP["tensorChar"], para, perp, self.globalPtrStr, \
        #        tP["tensorChar"], tP["tensorChar"], para, perp, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments A/B
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    kStr += "%s%s globalReadInc%s%s = (%s)stride%s%s" \
        % (self.indent, self.int64Str, tP["tensorChar"], loopChar, \
        self.int64Str, tP["tensorChar"], loopChar)
    if loopIdx==self.unrollIdx:
      kStr += "*LOCAL_DEPTHU"
      if kernel["GlobalSplitU"] > 1 \
          and kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        kStr += "*GLOBAL_SPLITU"
    else:
      for j in range(loopIdx+1, \
          min(loopIdx+2,kernel["ProblemType"]["NumIndicesSummation"]) ):
        tmpChar = self.indexChars[ \
            kernel["ProblemType"]["IndicesSummation"][j]]
        kStr += " - stride%s%s*(size%s%s" % (tP["tensorChar"], tmpChar, tmpChar, "/LOCAL_DEPTHU)*LOCAL_DEPTHU" if loopIdx == self.unrollIdx-1 else ")")
        #kStr += " - stride%s%s*(size%s-1)" % (tP["tensorChar"], tmpChar, tmpChar)
    kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  def lwaTileAssignment(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int lw%s%s = (serial%s" \
        % (tP["tensorChar"], tP["tileChar"], ("%" if tP["grcg"] \
        == tP["tlu"] else "/") )
    if tP["grcg"]:
      kStr += (tP["lvc"] if tP["grcv"] else tP["lsc"])
    else:
      kStr += (tP["lsp"] if tP["grcv"] else tP["lvp"])
    kStr += ")";
    if tP["grcv"] == tP["tlu"]:
      kStr += "*GLOBAL_LOAD_VECTOR_WIDTH_%s" % tP["tensorChar"]
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int lw%s%s = (serial%s" \
        % (tP["tensorChar"], self.unrollChar, ("/" if tP["grcg"] \
        == tP["tlu"] else "%") )
    if tP["grcg"]:
      kStr += (tP["lvc"] if tP["grcv"] else tP["lsc"])
    else:
      kStr += (tP["lsp"] if tP["grcv"] else tP["lvp"])
    kStr += ")";
    if tP["grcv"] != tP["tlu"]:
      kStr += "*GLOBAL_LOAD_VECTOR_WIDTH_%s" % tP["tensorChar"]
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int localWriteFirstOffset%s = lw%s%s + lw%s%s*(MT%s+PAD)%s;%s" \
        % (tP["tensorChar"], tP["tensorChar"], tP["tileChar"], \
        tP["tensorChar"], self.unrollChar, tP["tileChar"], \
        " + LDS_OFFSET_B" if tP["isB"] else "", self.endLine)
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A/B
  ##############################################################################
  def lwaFinalOffsets(self, kernel, tP):
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "  unsigned int localWriteOffset%s_%u_%u_%u_%u = localWriteFirstOffset%s + (%u + %d*%s)" \
                % (tP["tensorChar"], para, sPara, perp, sPerp, \
                tP["tensorChar"], sPara if tP["tlu"] else sPerp, para, \
                (tP["lsc"] if not tP["tlu"] else tP["lsc"]) )
            if not tP["tlu"]:
              kStr += "*(MT%s+PAD)" % (tP["tileChar"])
            kStr += " + (%u + %d*%s)" % (
                sPerp if tP["tlu"] else sPara, perp, \
                (tP["lsp"] if tP["tlu"] else tP["lsp"]) )
            if tP["tlu"]:
              kStr += "*(MT%s+PAD)" % (tP["tileChar"])
            kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Declare Addresses A/B
  ##############################################################################
  def lwaDeclareAddresses(self, kernel, tP):
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "  %sDATA_TYPE *localWrite%s_%u_%u_%u_%u;%s"\
                % (self.sharedPtrStr, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.endLine )
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment A
  ##############################################################################
  def lraTileAssignmentA(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int lr%s = (serial %% SG%s);%s" \
        % (tP["tileChar"], self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment B
  ##############################################################################
  def lraTileAssignmentB(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int lr%s = (serial / SG%s) %% SG%s;%s" \
        % (tP["tileChar"], self.tileChar0, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset A
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int localReadOffset%s = lr%s*VECTOR_WIDTH + sgId*(MT%s+PAD)%s;%s" \
        % ( tP["tensorChar"], tP["tileChar"], tP["tileChar"], \
        " + LDS_OFFSET_B" if tP["isB"] else "", self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    kStr = ""
    kStr += "  %sDATA_TYPE *localRead%s;%s" % (self.sharedPtrStr, \
        tP["tensorChar"], self.endLine)
    return kStr

  ##############################################################################
  # openShadowInit
  ##############################################################################
  def openShadowInit(self, kernel):
    return ""

  ##############################################################################
  # Initialize C
  ##############################################################################
  def initC(self, kernel):
    kStr = ""

    # init rC, in pf this is called twice
    kStr += self.endLine
    for i in range(0, kernel["ThreadTile0"]*kernel["ThreadTile1"]):
        kStr += "  rC[%u] = SCALAR_ZERO;%s" % (i, self.endLine)

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
  # Declare stagger parms used for both A and B
  # Input is the number of loop iterations
  # Defines staggerUIter
  # staggerUIter must be power-of-2 to simplify masking implementation
  ##############################################################################
  def declareStaggerParms(self, kernel):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]

    # Number of elements in U accessed by the unroll loop:
    # Does not include elements accessed in tail loop
    kStr += "  const unsigned origNumIter = numIter%s;%s" % (loopChar, self.endLine)

    if kernel["StaggerUMapping"] == 0:
      staggerInput = ("wg%s" % self.tileChar0)
    elif kernel["StaggerUMapping"] == 1:
      staggerInput = "wg%s" % self.tileChar1
    elif kernel["StaggerUMapping"] == 2:
      staggerInput = "wg2"
    elif kernel["StaggerUMapping"] == 3:
      staggerInput = "wgSerial"
    elif kernel["StaggerUMapping"] == 4:
      staggerInput = "0xffffffff" # all WG compute same stagger, this is test mode
    else:
      assert(0) # unsupported
    #kStr += "if (serial==0) printf(\"xWG:%u_%u progWG:%u_%u staggerUIterParm=%u\\n\", hc_get_group_id(0), hc_get_group_id(1), wg0I, wg1J, staggerUIterParm);"  + self.endLine
    kStr += "  unsigned staggerUIter = (%s & staggerUIterParm);%s" % (staggerInput, self.endLine)

    bpeAB = int(4*kernel["ProblemType"]["DataType"].numRegisters())
    kStr += "  staggerUIter = (staggerUIter << %u); // shift so each stagger has %u-byte stride%s" \
            % (kernel["_staggerStrideShift"], \
              (1<<kernel["_staggerStrideShift"])*kernel["DepthU"]*bpeAB, self.endLine)
    #kStr += "if (serial==0) printf(\"WG:%u_%u progWG:%u_%u staggerUIter=%u\\n\", hc_get_group_id(0), hc_get_group_id(1), wg0I, wg1J, staggerUIter);"  + self.endLine
    #kStr += "  staggerUIter = 0;\n"

    if self.db["PrintStagger"]:
      kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) == 0)%s" % \
              (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
      kStr += "  printf(%sStaggerOffset loop init: numIter=%%u, staggerUIter=%%u, globalReadIncAL=%%lu globalReadIncBL=%%lu %s,\
                        numIter%s, staggerUIter, globalReadIncAL, globalReadIncBL);%s" \
                      % (self.quote, self.endLineQuote, loopChar, self.endLine)
    return kStr

  ##############################################################################
  # Calculate and apply stagger offsets
  #
  # To help with cache and memory parking, offset the start position in the
  # summation dimension so each group starts at a different K
  ##############################################################################
  def calculateStagger(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    loopIdx = self.unrollIdx
    loopChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            gr = "globalRead%s_%u_%u_%u_%u" \
                  % (tc,   para, sPara, perp, sPerp)

            ti= "globalReadOffset%s%s_%u_%u" \
                % (tc, tP["tileChar"], \
                (para if tP["tlu"] else perp), \
                (sPara if tP["tlu"] else sPerp) )

            kStr += "  %s += (staggerUIter * globalReadInc%s%s); // apply stagger offset%s" \
                    % (gr, tc, loopChar, self.endLine)

            if self.db["PrintStagger"]:
              kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 16)%s" % \
                      (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
              # typecasting to work around hcc printf bugs:
              kStr += "printf(%sStaggerOffset init: gid=%%u.%%u.%%u, ti=0x%%x  %s-%s=0x%%x%s, \
                              %s(2),%s(1),%s(0), %s,  (unsigned)(size_t)(%s-%s));%s" \
                             % (self.quote,\
                                gr, tc,
                                self.endLineQuote,\
                                self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr,\
                                ti,  gr, tc, \
                                self.endLine)

    # if prefetching then wrap iteration needs to change since already prefetched
    # some tiles before entering the unroll loop
    kStr += self.endLine
    if tP["isB"]:
      kStr += "  staggerUIter += %u; // add PrefetchGlobalRead%s" \
          % (kernel["PrefetchGlobalRead"], self.endLine)
      # StaggerUIter is now the loop iteration where we should wrap the offset back to 0

    return kStr

  ##############################################################################
  # Remove the stagger offset for the kernel
  # (used in tail loop)
  ##############################################################################
  def removeStagger(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    loopIdx = self.unrollIdx
    loopChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            gr = "globalRead%s_%u_%u_%u_%u" \
                  % (tP["tensorChar"], para, sPara, perp, sPerp)

            if self.staggerU:
              kStr += "  %s += ((origNumIter - (staggerUIter - %u)) * globalReadInc%s%s); // remove stagger offset%s" \
                      % (gr, kernel["PrefetchGlobalRead"], tc, loopChar, self.endLine)

              if self.db["PrintStagger"]:
                kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 8)%s" % \
                        (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
                kStr += "printf(%sStaggerOffset remove: gid=%%u.%%u.%%u, origNumIter=%%u staggerUIter=%%u %s=%%p %s=%%p %s, \
                                %s(2),%s(1),%s(0), origNumIter, staggerUIter, %s, %s);%s" \
                               % (self.quote, \
                                  tc, gr, \
                                  self.endLineQuote, \
                                  self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr,
                                  tc, gr, \
                                  self.endLine)

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
      kStr += self.endLine + "  /* Compute tail loop num iter */" + self.endLine
      kStr += "%snumIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU);%s" \
          % (self.indent, self.unrollChar, self.unrollChar, self.endLine)
      if kernel["GlobalSplitU"] > 1:
        kStr += "%sif (gsuSumIdx != numIterPerWgRemainder) {%s" \
            % (self.indent, self.endLine)
        kStr += "%s  numIter%s = 0;%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%s}%s" % (self.indent, self.endLine)
        #kStr += "if (serial==0) printf(\\\"WG%u_%u TK:%u\\\\n\\\", get_group_id(0), get_group_id(1), numIterK);" + self.endLine
    else:
      kStr += self.endLine + "  /* Compute unroll loop num iter */" + self.endLine
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
  def closeLoop(self, kernel, loopIdx, finalLoop):
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
  def endSummation(self,kernel):
    return ""

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  def macIter(self, kernel, black, iuiCount):
    kStr = ""
    for iui in range(0,iuiCount):
        kStr += "%sMAC_%ux%u" % (self.indent, \
            kernel["ThreadTile0"],kernel["ThreadTile1"])
        if black:
          kStr += "_BLK"
        kStr += self.endLine
    return kStr

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel, prefetch):
    kStr = ""
    if kernel["GlobalSplitU"] > 1:
      kStr += "%sif (numIterMyWg >= 1) {%s" \
          % (self.indent, self.endLine)
    else:
      kStr += "%sif (size%s >= LOCAL_DEPTHU) {%s" \
          % (self.indent, self.unrollChar, self.endLine)
    self.indent += "  "
    return kStr

  def closeSumAtLeastUnroll(self, kernel, prefetch):
    kStr = ""
    self.indent = self.indent[2:]
    kStr += "%s} // end %s%s" % \
        (self.indent, "PrefetchGlobalRead" if prefetch else "unroll", self.endLine)
    if prefetch:
      kStr += "%selse { // still need to initC even if skipped prefetch%s" % (self.indent, self.endLine)
      kStr += self.initC(kernel)
      kStr += "%s}%s" % (self.indent, self.endLine)

    return kStr

  ##############################################################################
  # Global Read: Increment A/B
  ##############################################################################
  def globalReadIncrement(self, kernel, loopIdx, tP, prefetchIndex):
    kStr = ""
    tc = tP["tensorChar"]
    loopChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    kStr += self.comment("global read inc %s"%tc)
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            kStr += "%sglobalRead%s_%u_%u_%u_%u = (%sDATA_TYPE const *)( ((%sDATA_TYPE const *)globalRead%s_%u_%u_%u_%u) + globalReadInc%s%s);%s" \
                % (self.indent, tc, para, sPara, perp, sPerp, \
                self.globalPtrStr, self.globalPtrStr, tP["tensorChar"], \
                para, sPara, perp, sPerp, \
                tc, loopChar, self.endLine)

            if self.staggerU and loopIdx==self.unrollIdx:
              # Check to see if GRA wraps around edge:
              gr = "globalRead%s_%u_%u_%u_%u" \
                      % (tP["tensorChar"], para, sPara, perp, sPerp)

              kStr += "%sif ((numIter%s) == staggerUIter) {%s" \
                      % (self.indent, loopChar, self.endLine)

              if self.db["PrintStagger"]:
                # note loop counter numIterK/numIterL hard-coded, manually hack if needed
                kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 16)%s" % \
                        (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
                kStr += "printf(%sStaggerOffset wrap-gro: gid=%%u.%%u.%%u, old GR-%s=0x%%x numIter=%%u staggerUIter=%%u%s,\
                                  %s(2),%s(1),%s(0), (unsigned)(size_t)(%s-%s), numIterL, staggerUIter);%s" \
                                 % (self.quote, \
                                    tc, \
                                    self.endLineQuote, \
                                    self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, \
                                    gr,tc, \
                                    self.endLine)

              kStr += "  %s%s -= (origNumIter * globalReadInc%s%s); // wrap staggered offset back to row start%s" \
                      % (self.indent, \
                         gr,  tc, loopChar,
                         self.endLine)
              kStr += "%s}%s" % (self.indent, self.endLine)
              if self.db["PrintStagger"]:
                kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 8)%s" % \
                        (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
                kStr += "printf(%sStaggerOffset check-gro: gid=%%u.%%u.%%u, GR-%s=0x%%x %s, \
                                %s(2),%s(1),%s(0), (unsigned)(size_t)(%s-%s));%s" \
                               % (self.quote, \
                                  tc, \
                                  self.endLineQuote, \
                                  self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, \
                                  gr,tc, \
                                  self.endLine)

          #else:
          #  kStr += "%sglobalRead%s_%u_%u%s += globalReadInc%s%s%s;%s" \
          #      % (self.indent, tP["tensorChar"], para, perp, \
          #      (("_s%u"%s) if (tP["rtc"] \
          #      or tP["ruc"]) else ""), \
          #      tP["tensorChar"], loopChar, "" if (tP["rtc"] \
          #      or tP["ruc"]) else "/VECTOR_WIDTH", \
          #      self.endLine)
    return kStr

  ##############################################################################
  # Global Read: Do It A/B
  ##############################################################################
  def globalReadDo(self, kernel, mode, tP):
    kStr = ""
    tc = tP["tensorChar"]

    guardK = (mode==2)
    kStr += self.comment("global read %s")%tc

    #for perp in range(0, tP["nrp"]):
    #  for para in range(0, tP["nrc"]):
    #    for s in range(0, numUnrollVectorComponents):
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]):
            dest ="%s_%u_%u_%u_%u" \
                % (tP["tensorChar"].lower(), \
                para, sPara, perp, sPerp )
            kStr += "%s%s = " % (self.indent, dest)
            # guard around K
            if guardK:
              kStr += "( globalReadOffset%s%s_%u_%u + %u >= (size%s %% LOCAL_DEPTHU%s)%s )" \
                  % (tP["tensorChar"], self.unrollChar, \
                  (perp if tP["tlu"] else para), \
                  (sPerp if tP["tlu"] else 0), (0 if tP["tlu"] else sPara), self.unrollChar, \
                  (" + LOCAL_DEPTHU*gsuSumIdx" if kernel["GlobalSplitU"]>1 \
                  else ""), (" || !numIter%s"%self.unrollChar) \
                  if kernel["GlobalSplitU"] > 1 else "")
            # guard around edge
            if kernel["EdgeType"] == "Branch":
              if guardK:
                kStr += " || "
              kStr += "( !inBounds%s_%u )" % ( \
                  (tP["tensorChar"], para if tP["tlu"] else perp) )
            if kernel["EdgeType"] == "Branch" or guardK:
              kStr += " ? SCALAR_OOB_DATA : "
            kStr += "*(globalRead%s_%u_%u_%u_%u + %u);%s" \
                % (tP["tensorChar"], para, 0 if tP["rc"] else sPara, perp, sPerp, sPara if tP["rc"] else 0, \
                self.endLine)

            #if self.db["PrintStagger"] and tP["isA"]:
            if 0 and self.db["PrintStagger"]:
              kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 16)%s" % \
                      (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
              kStr += "  printf(%sglobalRead: gid=%%u.%%u.%%u, %s loaded:%%.0f%s, \
                                %s(2),%s(1),%s(0), %s );%s" \
                             % (self.quote,\
                                tc,
                                self.endLineQuote, \
                                self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr,\
                                dest, \
                                self.endLine)
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, tP):
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "%slocalWriteOffset%s_%u_%u_%u_%u = (localWriteOffset%s_%u_%u_%u_%u + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
                % (self.indent, tP["tensorChar"], \
                para, sPara, perp, sPerp, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.endLine )
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets A/B
  ##############################################################################
  def localWriteResetOffsets(self, kernel, tP):
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "%slocalWriteOffset%s_%u_%u_%u_%u %%= LDS_OFFSET_BLK;%s" \
                % (self.indent, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.endLine )
    return kStr

  ##############################################################################
  # Local Write: Init Pointers A/B
  ##############################################################################
  def localWriteInitPointers(self, kernel, tP):
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "%slocalWrite%s_%u_%u_%u_%u = (%sDATA_TYPE *)(localMemory + localWriteOffset%s_%u_%u_%u_%u);%s"\
                % (self.indent, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.sharedPtrStr, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.endLine)
    return kStr

  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  def localWriteDo(self, kernel, tP):
    kStr = ""
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wconditional-uninitialized\"" + self.endLine
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nwcv"]):
            kStr += "%s*(localWrite%s_%u_%u_%u_%u + %u) = %s_%u_%u_%u_%u;%s" \
                % (self.indent, tP["tensorChar"], \
                para, 0, perp, sPerp, sPara, \
                tP["tensorChar"].lower(), \
                para, \
                sPara if tP["tlu"] else sPerp, \
                perp, \
                sPerp if tP["tlu"] else sPara, \
                self.endLine)
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine
    if False and tP["isB"]:
      kStr += "%s%s" % (self.syncStr, self.endLine)
      kStr += "    /* print Local state */" + self.endLine
      kStr += "    for (unsigned int i = serial; i < LDS_NUM_ELEMENTS; i+=NUM_THREADS) {%s" % self.endLine
      kStr += "      printf(\\\"lds[%%06u] = %%.0f\\\\n\\\", i, localMemory[i]);%s" % self.endLine
      kStr += "    }" + self.endLine
    return kStr

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    kStr = ""
    kStr += "%slocalReadOffset%s = (localReadOffset%s + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
        % (self.indent, tP["tensorChar"], tP["tensorChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A/B
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    kStr = ""
    kStr += "%slocalReadOffset%s %%= LDS_OFFSET_BLK;%s" \
        % (self.indent, tP["tensorChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  def localReadInitPointers(self, kernel, tP):
    kStr = ""
    kStr += "%slocalRead%s = (%sDATA_TYPE *)(localMemory + localReadOffset%s);%s" \
        % (self.indent, tP["tensorChar"], self.sharedPtrStr, \
        tP["tensorChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  def localReadInc(self, kernel, iui, tP):
    kStr = ""
    kStr += "%slocalRead%s += LOCAL_SPLITU*(MT%s+PAD);%s" \
        % (self.indent, tP["tensorChar"], tP["tileChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Do It A/B
  ##############################################################################
  def localReadDo(self, kernel, black, iui, tP):
    kStr = ""
    for r in range(0, kernel[tP["tt"]]/kernel["VectorWidth"]):
      for s in range(0, kernel["VectorWidth"]):
        kStr += "%sr%s[%u*VECTOR_WIDTH+%u%s] = localRead%s[%u*SG%s*VECTOR_WIDTH + %u]; %s" \
            % (self.indent, tP["tensorChar"], r, s, \
            (("+TT%s"%tP["tileChar"]) if black else ""), \
            tP["tensorChar"], r, tP["tileChar"], s, self.endLine)
    return kStr

  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponents(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int wgMT%s = size%s - wg%s*MT%s;%s" \
        % (tP["tileChar"], tP["tileChar"], tP["tileChar"], \
        tP["tileChar"], self.endLine)
    kStr += "  if (wgMT%s > MT%s) wgMT%s = MT%s;%s" \
        %(tP["tileChar"], tP["tileChar"], tP["tileChar"], \
        tP["tileChar"], self.endLine)
    kStr += "  unsigned int r%s = wgMT%s %% GLOBAL_LOAD_VECTOR_WIDTH_%s;%s" \
        % (tP["tileChar"], tP["tileChar"], tP["tensorChar"], self.endLine)
    kStr += "  if (r%s > 0 && ((wgMT%s/VECTOR_WIDTH) %% SG%s) == (serial %s SG%s)%s ) {%s" \
        % (tP["tileChar"], tP["tileChar"], tP["tileChar"], "%" if tP["isA"] else "/", \
        self.tileChar0, (" %% SG%s"%self.tileChar1) if tP["isB"] else "", self.endLine)

    # old
    #kStr += "    unsigned int s%s = (wgMT%s/VECTOR_WIDTH)/SG%s;%s" \
    #    % (tP["tileChar"], tP["tileChar"], tP["tileChar"], self.endLine)
    # new
    # (wgMT/(SG0*VW))*(VW/glvw) + (wgMT%VW) / glvw
    kStr += "    unsigned int s%s = (wgMT%s%%VECTOR_WIDTH)/GLOBAL_LOAD_VECTOR_WIDTH_%s + (wgMT%s/(SG%s*VECTOR_WIDTH))*(VECTOR_WIDTH/GLOBAL_LOAD_VECTOR_WIDTH_%s);%s" \
        % (tP["tileChar"], tP["tileChar"], tP["tensorChar"], \
        tP["tileChar"], tP["tileChar"], tP["tensorChar"], self.endLine)

    for r in range(1, tP["glvw"]):
      kStr += "    if (r%s == %u) {%s" % (tP["tileChar"], r, self.endLine)
      numVectors = kernel["ThreadTile%s"%tP["tileIdx"]]/tP["glvw"]
      for vIdx in range(0, numVectors):
        if vIdx == 0:
          kStr += "      "
        else:
          kStr += " else "
        if vIdx < numVectors-1:
          kStr += "if (s%s == %u) " % (tP["tileChar"], vIdx)
        kStr += "{%s" % self.endLine
        for tt in range(0, kernel["ThreadTile%u"%((tP["tileIdx"]+1)%2)]):
          for s in range(0, r):
            if tP["isA"]:
              kStr += "        rC[%u + %u*GLOBAL_LOAD_VECTOR_WIDTH_A + %u*TT%s] = rC[%u + %u*GLOBAL_LOAD_VECTOR_WIDTH_A + %u*TT%s];%s" \
                % (s, vIdx, tt, self.tileChar0, \
                s+tP["glvw"]-r, vIdx, tt, self.tileChar0, \
                self.endLine)
            else:
              kStr += "        rC[%u + %u*TT%s*GLOBAL_LOAD_VECTOR_WIDTH_B + %u*TT%s] = rC[%u + %u*TT%s*GLOBAL_LOAD_VECTOR_WIDTH_B + %u*TT%s];%s" \
                % (tt, vIdx, self.tileChar0, s, self.tileChar0, \
                tt, vIdx, self.tileChar0, \
                s+tP["glvw"]-r, self.tileChar0, self.endLine)
        #kStr += "printf(\\\"sv %u %u\\\");%s" % (r, vIdx, self.endLine)
        kStr += "      }"
        if vIdx == numVectors-1:
          kStr += self.endLine
      kStr += "    }%s" % self.endLine
    kStr += "  }%s" % self.endLine
    return kStr

  ##############################################################################
  # Shift Vectors Components d1
  ##############################################################################
  def shiftVectorComponents1(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int wgMT%s = size%s - %s*MT%s;%s" \
        % (self.tileChar1, self.tileChar1, "wg%s"%self.tileChar1, \
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
    for r1 in range(1, tP["glvw"]):
      kStr += "    if (r%s == %u) {%s" % (self.tileChar1, r1, self.endLine)
      numVectors = kernel["ThreadTile1"]/tP["glvw"]
      for vIdx in range(0, numVectors):
        if vIdx == 0:
          kStr += "      "
        else:
          kStr += " else "
        if vIdx < numVectors-1:
          kStr += "if (s%s == %u) " % (self.tileChar1, vIdx)
        kStr += "{%s" % self.endLine

        for s in range(0, r1):
          for tt0 in range(0, kernel["ThreadTile0"]):
            kStr += "        rC[%u+%u*TT%s*VECTOR_WIDTH + %u*TT%s] = rC[%u+%u*TT%s*VECTOR_WIDTH + %u*TT%s];%s" \
              % (tt0, vIdx, self.tileChar0, s, self.tileChar0, \
              tt0, vIdx, self.tileChar0, \
              s+tP["glvw"]-r1, self.tileChar0, self.endLine)

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
    kStr += "  %sDATA_TYPE *localLocalSplitU = (%sDATA_TYPE *)(localMemory);%s" \
        % (self.sharedPtrStr, self.sharedPtrStr, self.endLine)
    for j in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          for vc in range(0, kernel["VectorWidth"]):
            kStr += "%slocalLocalSplitU[%u + (lr%s + %u*SG%s + (MT%s/VECTOR_WIDTH)*(lr%s*VECTOR_WIDTH + %u + SG%s*VECTOR_WIDTH*%u) + (MT%s*MT%s/VECTOR_WIDTH)*sgId)*VECTOR_WIDTH] = rC[%u + (%u+%u*(TT%s/VECTOR_WIDTH)+%u*TT%s)*VECTOR_WIDTH];%s" \
                % (self.indent, vc, self.tileChar0, i, self.tileChar0, \
                self.tileChar0, self.tileChar1, \
                s, self.tileChar1, j, self.tileChar0, self.tileChar1, vc, i, s, \
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
    for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for s in range(0, kernel["VectorWidth"]):
        kStr += "  rC[%u + %3u*GLOBAL_WRITE_VECTOR_WIDTH] = localLocalSplitU[%u + (serial+%u*NUM_THREADS)*GLOBAL_WRITE_VECTOR_WIDTH];%s" \
            % (s, i, s, i, self.endLine)
    kStr += self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  def localSplitUReduction(self, kernel):
    kStr = ""
    for r in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"]):
          kStr += "  rC[%u + %3u*GLOBAL_WRITE_VECTOR_WIDTH] += localLocalSplitU[(%u + serial*GLOBAL_WRITE_VECTOR_WIDTH+%u*NUM_THREADS*GLOBAL_WRITE_VECTOR_WIDTH + %u*MT%s*MT%s)];%s" \
              % (s, i, s, i, r, self.tileChar0, self.tileChar1, self.endLine)
      kStr += self.endLine
    return kStr



  ##############################################################################
  # extractGlobalCDims: Extract the packed dims from mask(s)
  #
  # indexMask:
  #   - 0x1: add index0 extraction, if index0 has packed dims
  #   - 0x2: add index1 extraction, if index1 has packed dims
  ##############################################################################
  def extractGlobalCDims(self, kernel, base, indexMask):
    kStr = ""
    problemType = kernel["ProblemType"]
    index0 = kernel["ProblemType"]["Index0"]
    index1 = kernel["ProblemType"]["Index1"]
    lastIndex = None
    if indexMask & 0x1:
      flattenedGlobalC = "flattenedGlobalC0"
      lastIndex = index0
      targetIndices = problemType["IndexAssignmentsA"]
    if indexMask & 0x2:
      flattenedGlobalC = "flattenedGlobalC1"
      lastIndex = index1
      targetIndices = problemType["IndexAssignmentsB"]

    if lastIndex != None:
      if base != "":
        kStr += "  globalC%s = %s + %s;%s" \
            % (self.indexChars[lastIndex], flattenedGlobalC, base, self.endLine)

        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          if i != index0 and i != index1 and i in targetIndices and isPackedIndex(kernel,i,indexMask):
            #kStr += "printf(\"pre: serial:%%u wg0:%%u wg1:%%u globalC0I:%%u globalC1J:%%u\\n\", serial, wg0I, wg1J, globalC0I, globalC1J);%s" % (self.endLine)
            kStr += "  globalC%s = " % (self.indexChars[i])
            if self.useMagicNumber:
              c = globalParameters["IndexChars"][lastIndex]
              kStr += "MAGIC_DIV(globalC%s, magicNumberSize%s, magicShiftSize%s);%s" \
                      % (self.indexChars[lastIndex], c, c, self.endLine)
              kStr += "  globalC%s -= (globalC%s*size%s);%s" \
                      % (self.indexChars[lastIndex], self.indexChars[i], \
                         self.indexChars[lastIndex], self.endLine)
            else:
              kStr += "(globalC%s) / size%s;%s" % (self.indexChars[lastIndex], self.indexChars[lastIndex], self.endLine)
              kStr += "  globalC%s %%= size%s;%s" % (self.indexChars[lastIndex], self.indexChars[lastIndex], self.endLine)
            lastIndex = i
            #kStr += "printf(\"post: serial:%%u wg0:%%u wg1:%%u globalC0I:%%u globalCK=%%u\\n\", serial, wg0I, wg1J, globalC0I, globalCK);%s" % (self.endLine)

    return kStr

  ##############################################################################
  # globalWriteWorkGroupInit:
  ##############################################################################
  def globalWriteWorkGroupInit(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    kStr = ""

    # Add Index0
    index0 = kernel["ProblemType"]["Index0"]
    kStr += "  unsigned int localC%s = (serial %% (MT%s/GLOBAL_WRITE_VECTOR_WIDTH))*GLOBAL_WRITE_VECTOR_WIDTH;%s" \
        % (self.tileChar0, self.tileChar0, self.endLine)
    kStr += "  unsigned int globalC%s = (wg%s)" % (self.indexChars[index0], self.indexChars[index0])
    kStr += "*MT%s + localC%s;%s" % (self.tileChar0, self.tileChar0, self.endLine)
    # Save original flattened C0 before extracting batch components:
    kStr += "  unsigned int flattenedGlobalC0 = globalC%s;%s" \
        % (self.indexChars[index0], self.endLine)

    # Add Index1
    index1 = kernel["ProblemType"]["Index1"]
    kStr += "  unsigned int localC%s = serial / (MT%s/GLOBAL_WRITE_VECTOR_WIDTH);%s" \
        % (self.tileChar1, self.tileChar0, self.endLine)
    kStr += "  unsigned int globalC%s = (wg%s)" % (self.indexChars[index1], self.indexChars[index1])
    kStr += "*MT%s + localC%s;%s" % (self.tileChar1, self.tileChar1, self.endLine)
    kStr += "  unsigned int flattenedGlobalC1 = globalC%s;%s" \
        % (self.indexChars[index1], self.endLine)

    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      if i != index0 and i != index1:
        kStr += "  unsigned int globalC%s = " \
            % (self.indexChars[i])
        if isPackedIndex(kernel,i):
          kStr += "0; // define, will be set below%s" % (self.endLine)
        else:
          kStr += "(wg%s);%s" % (self.indexChars[i], self.endLine)
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    kStr = ""
    packGranularity = kernel["PackGranularity"]
    addTensorDimCheck0 = addTensorDimCheck1 = 0
    for b in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      loadOffset1 = " %u*CPSV" %b
      if packGranularity==2:
        addTensorDimCheck1 = 1
        base1 = loadOffset1
        loadOffset1 = "0"
      for s in range(0, kernel["GlobalWriteVectorWidth"]):
        loadOffset0 = "%u" % (s)
        if packGranularity==2:
          addTensorDimCheck0 = 1
          base0 = loadOffset0
          loadOffset0 = "0"
        if kernel["EdgeType"] != "None":
          if addTensorDimCheck0 or addTensorDimCheck1:
            kStr += self.endLine
          if addTensorDimCheck0:
            kStr += "  /* new 0 offset - inc and extract tensor dims */%s" % (self.endLine)
            kStr += self.extractGlobalCDims(kernel, base0, 0x1)
            addTensorDimCheck0 = 0
          if addTensorDimCheck1:
            kStr += "  /* new 1 offset - inc and extract tensor dims */%s" % (self.endLine)
            kStr += self.extractGlobalCDims(kernel, base1, 0x2)
            addTensorDimCheck1 = 0

          ### Bounds checks:
          # if packed, check flattened against product of all packed sizes
          # The flattened base never changes so add all address offsets before comparison
          globalC0ForCheck = "flattenedGlobalC0"
          size0ForCheck = " * ".join(self.tPA["packedSizeList"])
          kStr += "  if (%s%s < %s) {" \
              % (globalC0ForCheck, \
              ((" + %u" %s) if kernel["GlobalWriteVectorWidth"]>1 else ""), \
              size0ForCheck)

          globalC1ForCheck = "flattenedGlobalC1"
          size1ForCheck = " * ".join(self.tPB["packedSizeList"])
          kStr += "  if (%s + %u*CPSV < %s) {" \
              % (globalC1ForCheck, b, size1ForCheck)

        kStr += "  TYPE_MAC_WRITE( D[ GLOBAL_D( (%s)" % self.uint64Str
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          kStr += " globalC%s" % self.indexChars[i]
          if i == kernel["ProblemType"]["Index0"] and kernel["GlobalWriteVectorWidth"]>1:
            kStr += " + %s" % (loadOffset0)
          if i == kernel["ProblemType"]["Index1"]:
            kStr += " + %s" % (loadOffset1)
          if i < kernel["ProblemType"]["NumIndicesC"]-1:
            kStr += ", (%s)" % self.uint64Str
        kStr += ") ]"

        if kernel["ProblemType"]["UseBeta"]:
          kStr += ", C[ GLOBAL_C( (%s)" % self.uint64Str
          for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
            kStr += " globalC%s" % self.indexChars[i]
            if i == kernel["ProblemType"]["Index0"] and kernel["GlobalWriteVectorWidth"]>1:
              kStr += " + %s" % (loadOffset0)
            if i == kernel["ProblemType"]["Index1"]:
              kStr += " + %s" % (loadOffset1)
            if i < kernel["ProblemType"]["NumIndicesC"]-1:
              kStr += ", (%s)" % self.uint64Str
          kStr += ") ]"

        kStr += ", alpha"
        kStr += ", rC[%u + %u*GLOBAL_WRITE_VECTOR_WIDTH]" % (s, b )

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

    # Add Index0 and Index1:
    index0 = kernel["ProblemType"]["Index0"]
    kStr += "  unsigned int globalC%s = " % self.indexChars[index0]
    kStr += "(wg%s)*MT%s + (serial %% SG%s)*VECTOR_WIDTH;%s" \
            % (self.indexChars[index0], self.tileChar0, self.tileChar0, self.endLine)
    # Save original flattened C0 before extracting batch components:
    kStr += "  unsigned int flattenedGlobalC0 = globalC%s;%s" \
        % (self.indexChars[index0], self.endLine)

    index1 = kernel["ProblemType"]["Index1"]
    kStr += "  unsigned int globalC%s = " % self.indexChars[index1]
    kStr += "(wg%s)*MT%s + (serial / SG%s)*VECTOR_WIDTH;%s" \
            % (self.indexChars[index1], self.tileChar1, self.tileChar0, self.endLine)
    # Save original flattened C0 before extracting batch components:
    kStr += "  unsigned int flattenedGlobalC1 = globalC%s;%s" \
        % (self.indexChars[index1], self.endLine)

    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      if i != index0 and i != index1:
        kStr += "  unsigned int globalC%s = " % self.indexChars[i]
        if isPackedIndex(kernel,i):
          kStr += "0; // define, will be set below%s" % (self.endLine)
        else:
          #kStr += "printf(\"pre: serial:%%u wg0:%%u wg1:%%u globalC0I:%%u globalC1J:%%u\\n\", serial, wg0I, wg1J, globalC0I, globalC1J);%s" % (self.endLine)
          kStr += "(wg%s);%s" % (self.indexChars[i], self.endLine)

    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    kStr = ""
    packGranularity = kernel["PackGranularity"]
    addTensorDimCheck0 = addTensorDimCheck1 = 0

    for b in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for a in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        if packGranularity==2:
          addTensorDimCheck0 = 1
          base0 = " %u*SG%s*VECTOR_WIDTH" % (a,self.tileChar0)
        for s1 in range(0, kernel["VectorWidth"]):
          if packGranularity==2:
            addTensorDimCheck1 = 1
            base1 = "%u + %u*SG%s*VECTOR_WIDTH" % (s1, b,self.tileChar1)
            offsetS1 = ""
          else:
            offsetS1 = ((" + %u"%s1) if kernel["VectorWidth"]>1 else "")
          for s0 in range(0, kernel["VectorWidth"]):
            # set default offsets, may be overridden in packed mode:
            offsetS0 = ((" + %u"%s0) if kernel["VectorWidth"]>1 else "")
            offset0 = "%s + %u*SG%s*VECTOR_WIDTH" \
                      % (offsetS0, a, self.tileChar0)
            offset1 = "%s + %u*SG%s*VECTOR_WIDTH" % (\
                ((" + %u"%s1) if kernel["VectorWidth"]>1 else ""), \
                b, self.tileChar1)

            if kernel["EdgeType"] == "Branch":
              kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar0, a, self.tileChar0, self.tileChar0)
              kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar1, b, self.tileChar1, self.tileChar1)
            elif kernel["EdgeType"] == "ShiftPtr":
              if addTensorDimCheck0 or addTensorDimCheck1:
                kStr += self.endLine
              if addTensorDimCheck0:
                kStr += "  /* new vw0 offset - inc and extract tensor dims */%s" % (self.endLine)
                kStr += self.extractGlobalCDims(kernel, base0, 0x1)
                addTensorDimCheck0 = 0
              if addTensorDimCheck1:
                kStr += "  /* new vw1 offset - inc and extract tensor dims */%s" % (self.endLine)
                kStr += self.extractGlobalCDims(kernel, base1, 0x2)
                addTensorDimCheck1 = 0

              ### Bounds checks:
              # if packed, check flattened against product of all packed sizes
              # The flattened base never changes so add all address offsets before comparison
              if packGranularity == 2:
                # base contains some addressing components, so just offset here:
                offset0 = offsetS0
              globalC0ForCheck = "flattenedGlobalC0"
              size0ForCheck = " * ".join(self.tPA["packedSizeList"])

              # Check 0 dimension against appropriate size limit
              kStr += "  if (%s%s + %u*SG%s*VECTOR_WIDTH < %s) {" \
                  % (globalC0ForCheck,
                  ((" + %u"%s0) if kernel["VectorWidth"]>1 else ""), \
                  a, self.tileChar0, size0ForCheck)

              if packGranularity == 2:
                offset1 = offsetS1
              globalC1ForCheck = "flattenedGlobalC1"
              size1ForCheck = " * ".join(self.tPB["packedSizeList"])

              kStr += "  if (%s%s + %u*SG%s*VECTOR_WIDTH < %s) {" \
                  % (globalC1ForCheck,
                  ((" + %u"%s1) if kernel["VectorWidth"]>1 else ""), \
                  b, self.tileChar1, size1ForCheck)

            # Write the result
            kStr += "  TYPE_MAC_WRITE( D[ GLOBAL_D( (%s)" % self.uint64Str
            for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
              kStr += " globalC%s" % self.indexChars[i]
              if i == kernel["ProblemType"]["Index0"]:
                kStr += offset0
              if i == kernel["ProblemType"]["Index1"]:
                kStr += offset1
              if i < kernel["ProblemType"]["NumIndicesC"]-1:
                kStr += ", (%s)" % self.uint64Str
            kStr += ") ]"

            if kernel["ProblemType"]["UseBeta"]:
              kStr += ", C[ GLOBAL_C( (%s)" % self.uint64Str
              for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
                kStr += " globalC%s" % self.indexChars[i]
                if i == kernel["ProblemType"]["Index0"]:
                  kStr += offset0
                if i == kernel["ProblemType"]["Index1"]:
                  kStr += offset1
                if i < kernel["ProblemType"]["NumIndicesC"]-1:
                  kStr += ", (%s)" % self.uint64Str
              kStr += ") ]"

            kStr += ", alpha"
            #kStr += ", rC[%d+%d*(TT%s/VECTOR_WIDTH)+%d*TT%s]%s" \
            #    % (a, s1, self.tileChar0, b, self.tileChar0, \
            #    ((".%s"%self.vectorComponents[s0]) if kernel["VectorWidth"]>1\
            #    else "") )
            kStr += ", rC[%u*VECTOR_WIDTH+%u + (%u*VECTOR_WIDTH+%u)*TT%s]" \
                % (a, s0, b, s1, self.tileChar0 )
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

    if kernel["PersistentKernel"]:
      kStr += "  serialWgIter += %s(0);%s" \
        % (self.getNumGroupsStr, self.endLine)
      kStr += "} // End Persistent Loop" + self.endLine



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
      kStr += "#undef DEST_DATA_TYPE%s" % (self.endLine)
      #kStr += "#undef VECTOR_TYPE%s" % (self.endLine)
      kStr += "#undef LDS_OFFSET_B%s" % (self.endLine)
      kStr += "#undef LDS_OFFSET_BLK%s" % (self.endLine)
      kStr += "#undef LDS_NUM_ELEMENTS%s" % (self.endLine)
      kStr += "#undef NUM_THREADS%s" % (self.endLine)
      kStr += "#undef WORK_GROUP_MAPPING%s" % (self.endLine)
      kStr += "#undef VECTOR_WIDTH%s" % (self.endLine)
      kStr += "#undef GLOBAL_LOAD_VECTOR_WIDTH_A%s" % (self.endLine)
      kStr += "#undef GLOBAL_LOAD_VECTOR_WIDTH_B%s" % (self.endLine)
      kStr += "#undef GLOBAL_WRITE_VECTOR_WIDTH%s" % (self.endLine)
      kStr += "#undef MAC%s" % (self.endLine)
      kStr += "#undef TYPE_MAC%s" % (self.endLine)
      kStr += "#undef TYPE_MAC_WRITE%s" % (self.endLine)
      kStr += "#undef GLOBAL_SPLITU%s" % (self.endLine)
      # zero
      kStr += "#undef SCALAR_ZERO%s" % (self.endLine )

      numMacs = 2 if kernel["PrefetchLocalRead"] else 1
      for m in range(0, numMacs):
        kStr += "#undef MAC_%ux%u" \
            % (kernel["ThreadTile0"], kernel["ThreadTile1"])
        if kernel["PrefetchLocalRead"]:
          kStr += ("" if m==0 else "_BLK")
        kStr += self.endLine
      # initial strides
      firstStride = 0
      if kernel["ProblemType"]["UseInitialStrides"]:
        lastStrideD = 0
        lastStrideC = 0
        lastStrideA = 0
        lastStrideB = 0
      else:
        lastStrideD = 1
        lastStrideC = 1
        lastStrideA = 1
        lastStrideB = 1
      for i in range(firstStride, lastStrideD):
        kStr += "#undef strideD" + self.indexChars[i] + self.endLine
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
      # other summation indices
      for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
        index = i + kernel["ProblemType"]["NumIndicesC"]
        kStr += "#undef globalReadOffsetA%s%s" \
            % (self.indexChars[index], self.endLine)
        kStr += "#undef globalReadOffsetB%s%s" \
            % (self.indexChars[index], self.endLine)
      kStr += self.endLine + self.endLine
    return kStr

  ##############################################################################
  # Kernel Body Prefix
  ##############################################################################
  def kernelBodyPrefix(self, kernel, tPA, tPB ):
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
  def kernelBodySuffix(self, kernel, tPA, tPB ):
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
  def wait(self, kernel, tPA, tPB, globalRead, localWrite, localRead, comment):
    return ""

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def syncThreads(self, kernel, comment=""):
    return self.indent + self.syncStr + " //" + comment + self.endLine

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
      kStr += self.endLine
      kStr += "extern \"C\"\n"
      kStr += "__global__ "
    kStr += "void %s" % ( kernelName )
    kStr += "(" + self.endLine
    # pointers
    globalStr = "__global "
    if self.language == "HIP":
      #kStr += "  hipLaunchParm lp," + self.endLine
      globalStr = ""
    restrictStr = "restrict"
    if self.language == "HIP":
      restrictStr = "__restrict__"
    ptrStr = kernel["ProblemType"]["DestDataType"].toDevice(self.language)
    kStr += "  " + globalStr + ptrStr \
        + " *D,"
    kStr += self.endLine
    kStr += "  " + globalStr + ptrStr \
        + " const * " + restrictStr + " C,"
    kStr += self.endLine

    # strides
    firstStride = 1
    if kernel["ProblemType"]["UseInitialStrides"]:
      firstStride = 0
    lastStrideC = kernel["ProblemType"]["NumIndicesC"]
    for i in range(firstStride, lastStrideC):
      kStr += "  unsigned int const strideD%s,%s" \
          % (self.indexChars[i], self.endLine)
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
          % (kernel["ProblemType"]["ComputeDataType"].toDevice(self.language), \
          self.endLine )
    return kStr

  ##############################################################################
  # Kernel Body Beta-Only
  ##############################################################################
  def kernelBodyBetaOnly(self, kernel):
    kStr = ""
    kStr += "{%s" % self.endLine

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
      kStr += "#define strideD" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + self.indexChars[i] + " 1" + self.endLine

    ########################################
    # GLOBAL_D()
    kStr += "#define GLOBAL_D(IDX%s" % self.indexChars[0]
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideD%s" % (indexChar, indexChar)
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideD%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine
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

    ######################################## # C indices
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
    # D index
    kStr += "  %s idxD = GLOBAL_D( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += " globalC%s" % self.indexChars[i]
      if i < kernel["ProblemType"]["NumIndicesC"]-1:
        kStr += ", "
    kStr += ");%s" % (self.endLine)
    # C index
    kStr += "  %s idxC = GLOBAL_C( (%s)" % (self.uint64Str, self.uint64Str)
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
    if kernel["ProblemType"]["UseBeta"]:
      if kernel["ProblemType"]["DataType"].isComplex():
        kStr += "    if((beta.s0 == 0) && (beta.s1 == 0)) {%s" % self.endLine
      else:
        kStr += "    if(beta == SCALAR_ZERO) {%s" % self.endLine
      kStr += "      D[idxD] = SCALAR_ZERO;%s" % self.endLine
      kStr += "    } else {%s" % self.endLine
      kStr += "      D[idxD] = C[idxC]*beta;%s" % self.endLine
      kStr += "    }%s" % self.endLine
    else:
      kStr += "    D[idxD] = SCALAR_ZERO;%s" % (self.endLine)
    kStr += "  }%s" % self.endLine



    ########################################
    # end
    kStr += "}%s" % self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)
    kStr += "#undef GLOBAL_C%s" % (self.endLine)
    if kernel["ProblemType"]["DataType"].isReal():
      kStr += "#undef SCALAR_ZERO%s" % ( self.endLine)
    return kStr
