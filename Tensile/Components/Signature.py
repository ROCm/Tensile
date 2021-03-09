################################################################################
# Copyright 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import Signature
from ..Common import globalParameters

srcValueTypeDict = {
    "f16":  "Struct",
    "i8":   "I8",
    "f32":  "F32",
    "f64":  "F64",
    "f32c": "F64",
    "f64c": "F64",
    "bf16": "Struct"
}

dstValueTypeDict = {
    "f16":  "Struct",
    "i8":   "I32",
    "f32":  "F32",
    "f64":  "F64",
    "f32c": "F64",
    "f64c": "F64",
    "bf16": "Struct"
}

cptValueTypeDict = {
    "f16":  "F16",
    "i8":   "I32",
    "f32":  "F32",
    "f64":  "F64",
    "f32c": "F64",
    "f64c": "Struct",
    "bf16": "F32"
}

def getSrcValueType(kernel, cov):
    srcValueType = srcValueTypeDict[kernel["ProblemType"]["DataType"].toNameAbbrev()]
    if kernel["ProblemType"]["DataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
        srcValueType = "F16"
    if cov == "V3":
        srcValueType = srcValueType.lower()
    return srcValueType

def getDstValueType(kernel, cov):
    dstValueType = dstValueTypeDict[kernel["ProblemType"]["DataType"].toNameAbbrev()]
    if kernel["ProblemType"]["DataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
        dstValueType = "F16"
    if cov == "V3":
        dstValueType = dstValueType.lower()
    return dstValueType

def getCptValueType(kernel, cov):
    cptValueType = cptValueTypeDict[kernel["ProblemType"]["DataType"].toNameAbbrev()]
    if cov == "V3":
        cptValueType = cptValueType.lower()
    return cptValueType

def getCptByte(kernel):
    cptByte = 4
    if kernel["ProblemType"]["DataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
        cptByte = 2
    elif kernel["ProblemType"]["DataType"].isDouble() or kernel["ProblemType"]["DataType"].isSingleComplex():
        cptByte = 8
    elif kernel["ProblemType"]["DataType"].isDoubleComplex():
        cptByte = 16
    return cptByte

def getCptSize(kernel):
    return str(getCptByte(kernel))

def getCptAlign(kernel):
    return str(getCptByte(kernel))

class SignatureCOV2(Signature):
    kernel = {"CodeObjectVersion": "V2"}

    def v2Argument(self, name, size, align, valueKind, valueType, AddrSpaceQual = None):
        kStr = ""
        kStr += "      - Name:            %s\n" % name
        kStr += "        Size:            %s\n" % size
        kStr += "        Align:           %s\n" % align
        kStr += "        ValueKind:       %s\n" % valueKind
        kStr += "        ValueType:       %s\n" % valueType
        if AddrSpaceQual != None:
            kStr += "        AddrSpaceQual:   %s\n" % AddrSpaceQual
        return kStr

    def __call__(self, writer):
        kernel = writer.kernel

        kStr = self.commentHeader()

        # begin kernel descriptor
        kStr += ".hsa_code_object_version %s,0%s" \
            % (globalParameters["CodeObjectVersion"][1], writer.endLine)
        kStr += ".hsa_code_object_isa %u, %u, %u, \"AMD\", \"AMDGPU\" %s" \
            % (writer.version[0], writer.version[1], writer.version[2], writer.endLine)

        kStr += ".text%s" % writer.endLine
        kStr += ".protected %s%s" % (writer.kernelName, writer.endLine)
        kStr += ".globl %s%s" % (writer.kernelName, writer.endLine)
        kStr += ".p2align 8%s" % writer.endLine
        kStr += ".type %s,@function%s" % (writer.kernelName, writer.endLine)

        tWord = "amdgpu_hsa_kernel"
        kStr += ".%s %s%s" % (tWord, writer.kernelName, writer.endLine)
        kStr += "%s:%s" % (writer.kernelName, writer.endLine)
        kStr += ".amd_kernel_code_t%s" % writer.endLine
        kStr += "  is_ptr64 = 1%s" % writer.endLine
        tWord = "enable_sgpr_kernarg_segment_ptr ="
        kStr += "  %s 1%s" % (tWord, writer.endLine)

        # kern arg size
        kernArgReg = 0
        kernArgReg += 3*writer.rpga
        kernArgReg += max(1,int(writer.bpeAB/4)) # alpha
        if kernel["ProblemType"]["UseBeta"]:
            kernArgReg += max(1,int(writer.bpeCexternal/4)) # beta
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsA"]) # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsB"]) # strides
        if not kernel["ProblemType"]["UseInitialStridesAB"]:
            kernArgReg -= 2 # strides
        if not kernel["ProblemType"]["UseInitialStridesCD"]:
            kernArgReg -= 2 # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesSummation"]
        kernArgReg += kernel["ProblemType"]["NumIndicesC"]
        if globalParameters["DebugKernel"]:
            kernArgReg += writer.rpga # debug buffer
        kernArgBytes = kernArgReg * 4 # bytes/reg
        kStr += "  kernarg_segment_byte_size = %u // bytes of kern args%s" \
            % (kernArgBytes, writer.endLine)

        # register allocation
        totalVgprs = writer.vgprPool.size()
        totalSgprs = writer.sgprPool.size()
        tWord = "workitem_vgpr_count ="
        kStr += "  %s %u // vgprs%s" % (tWord, totalVgprs, writer.endLine)
        tWord = "wavefront_sgpr_count ="
        kStr += "  %s %u // sgprs%s" % (tWord, totalSgprs, writer.endLine)

        kStr += "  compute_pgm_rsrc1_vgprs = %u // floor((%u-1)/4)%s" \
            % ( (totalVgprs-1)//4, totalVgprs, writer.endLine)
        kStr += "  compute_pgm_rsrc1_sgprs = %u // floor((%u-1)/8)%s" \
            % ( 1+(totalSgprs-1)//8, totalSgprs, writer.endLine)

        # work-group dimensions
        kStr += "  compute_pgm_rsrc2_tidig_comp_cnt = 0 // 1D wg%s" % writer.endLine

        # grid dimensions
        kStr += "  compute_pgm_rsrc2_tgid_x_en = 1 // wg.x%s" % writer.endLine
        kStr += "  compute_pgm_rsrc2_tgid_y_en = 1 // wg.y%s" % writer.endLine
        if kernel["ProblemType"]["NumIndicesC"] > 2:
            kStr += "  compute_pgm_rsrc2_tgid_z_en = %u // wg.z%s" % (1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0, writer.endLine)
        #if abs(kernel["WorkGroupMapping"]) > 1:
        #  kStr += "  enable_sgpr_grid_workgroup_count_x = 1 // nwg0%s" % writer.endLine
        #  kStr += "  enable_sgpr_grid_workgroup_count_y = 1 // nwg1%s" % writer.endLine

        # lds size
        #kStr += "  compute_pgm_rsrc2_lds_size = 1 // ?%s" % writer.endLine # don't use, it eats up 512 bytes of LDS
        #jgolds HACK
        # only want to enable this for cases we know it helps: 4x4 TT size and 16x16 WG size. Feel free to add more
        # cases after validating performance

        tWord = "workgroup_group_segment_byte_size ="
        if kernel["AggressivePerfMode"]>=2 and kernel["ProblemType"]["DataType"].isDouble() and \
            kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 4 and kernel["WorkGroup"] == [16,16,1]:
            group_segment_size = 32768 # Pad LDS to ensure we run exactly two waves
        else:
            group_segment_size = kernel["LdsNumElements"] * writer.bpeAB
        kStr += "  %s %u // lds bytes%s" % ( tWord, group_segment_size, writer.endLine )

        if writer.archCaps["HasWave32"]:
            if kernel["WavefrontSize"] == 32:
                kStr += "  wavefront_size = 5 // 32-thread wavefronts%s" % writer.endLine
            else:
                kStr += "  wavefront_size = 6 // 64-thread wavefronts%s" % writer.endLine

        # other
        kStr += "  compute_pgm_rsrc2_user_sgpr = 2 // vcc%s" % writer.endLine
        kStr += "  kernarg_segment_alignment = 4%s" % writer.endLine
        kStr += "  group_segment_alignment = 4%s" % writer.endLine
        kStr += "  private_segment_alignment = 4%s" % writer.endLine
        kStr += ".end_amd_kernel_code_t%s" % writer.endLine

        kStr += writer.comment3("Optimizations and Config:")
        kStr += writer.comment1("ThreadTile= %u x %u" % (kernel["ThreadTile0"], kernel["ThreadTile1"]))
        kStr += writer.comment1("SubGroup= %u x %u" % (kernel["SubGroup0"], kernel["SubGroup1"]))
        kStr += writer.comment1("VectorWidth=%u" % (kernel["VectorWidth"]))
        kStr += writer.comment1("GlobalLoadVectorWidthA=%u, GlobalLoadVectorWidthB=%u" % (kernel["GlobalLoadVectorWidthA"], kernel["GlobalLoadVectorWidthB"]))
        kStr += writer.comment1("DirectToLdsA=%s" % kernel["DirectToLdsA"])
        kStr += writer.comment1("DirectToLdsB=%s" % kernel["DirectToLdsB"])
        kStr += writer.comment1("UseSgprForGRO=%s" % kernel["_UseSgprForGRO"])

        srcValueType = getSrcValueType(kernel, "V2")
        dstValueType = getDstValueType(kernel, "V2")
        cptValueType = getCptValueType(kernel, "V2")
        cptByte = getCptByte(kernel)
        # cptSize = getCptSize(kernel)
        # cptAlign = getCptAlign(kernel)

        # Codeobject V2 metadata
        kStr += ".amd_amdgpu_hsa_metadata\n"
        kStr += "Version: [ 1, 0 ]\n"
        kStr += "Kernels:\n"
        kStr += "  - Name: %s%s" % (writer.kernelName, writer.endLine)
        kStr += "    SymbolName: '%s@kd'%s" % (writer.kernelName, writer.endLine)
        kStr += "    Language: OpenCL C\n"
        kStr += "    LanguageVersion: [ 2, 0 ]\n"
        kStr += "    Args:\n"
        ka_size = 0

        if globalParameters["DebugKernel"]:
            kStr += self.v2Argument(                    'AddressDbg',     '8',      '8', "GlobalBuffer",     "Struct", "Generic"); ka_size += 8

        kStr += self.v2Argument(                           'sizeC',     '8',      '8',      "ByValue",        "I64"); ka_size += 8
        kStr += self.v2Argument(                           'sizeA',     '8',      '8',      "ByValue",        "I64"); ka_size += 8
        kStr += self.v2Argument(                           'sizeB',     '8',      '8',      "ByValue",        "I64"); ka_size += 8

        kStr += self.v2Argument(                               'D',     '8',      '8', "GlobalBuffer", dstValueType, "Generic"); ka_size += 8
        kStr += self.v2Argument(                               'C',     '8',      '8', "GlobalBuffer", dstValueType, "Generic"); ka_size += 8
        kStr += self.v2Argument(                               'A',     '8',      '8', "GlobalBuffer", srcValueType, "Generic"); ka_size += 8
        kStr += self.v2Argument(                               'B',     '8',      '8', "GlobalBuffer", srcValueType, "Generic"); ka_size += 8

        useSize = max(4, cptByte)
        useAlign = useSize
        kStr += self.v2Argument(                             "alpha", useSize, useAlign,      "ByValue", cptValueType); ka_size += useSize
        if kernel["ProblemType"]["UseBeta"]:
            kStr += self.v2Argument(                          "beta", useSize, useAlign,      "ByValue", cptValueType); ka_size += useSize

        for i in range(0, writer.numSgprStridesD):
            kStr += self.v2Argument(                   "strideD%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        for i in range(0, writer.numSgprStridesC):
            kStr += self.v2Argument(                   "strideC%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        for i in range(0, writer.numSgprStridesA):
            kStr += self.v2Argument(                   "strideA%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        for i in range(0, writer.numSgprStridesB):
            kStr += self.v2Argument(                   "strideB%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        for i in range(0, writer.numSgprSizesFree):
            kStr += self.v2Argument(                 "SizesFree%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        for i in range(0, writer.numSgprSizesSum):
            kStr += self.v2Argument(                  "SizesSum%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        for magicName in writer.sumMagicParms:
            kStr += self.v2Argument(     "MagicNumberSize%s"%magicName,     '4',      '4',      "ByValue",        "U32"); ka_size += 4
            kStr += self.v2Argument(      "MagicShiftSize%s"%magicName,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        for idxChar in kernel["PackedC0IdxChars"][:-1]:
            kStr += self.v2Argument(     "MagicNumberSize%s"%idxChar,     '4',      '4',      "ByValue",        "U32"); ka_size += 4
            kStr += self.v2Argument(      "MagicShiftSize%s"%idxChar,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        for idxChar in kernel["PackedC1IdxChars"][:-1]:
            kStr += self.v2Argument(     "MagicNumberSize%s"%idxChar,     '4',      '4',      "ByValue",        "U32"); ka_size += 4
            kStr += self.v2Argument(      "MagicShiftSize%s"%idxChar,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        for idx in kernel["ProblemType"]["IndicesSummation"]:
          for tc in ('A','B'):
            for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
              (freeDim, sumDim, padStart, padEnd) = zp
              if sumDim == idx:
                freeDimChar = globalParameters["IndexChars"][freeDim]
                sumDimChar  = globalParameters["IndexChars"][sumDim]
                kStr += self.v2Argument("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar), '4', '4', "ByValue", "U32"); ka_size += 4
                kStr += self.v2Argument("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar), '4', '4', "ByValue", "U32"); ka_size += 4

        kStr += self.v2Argument(                "OrigStaggerUIter",     '4',      '4',      "ByValue",        "I32"); ka_size += 4

        kStr += self.v2Argument(                  "NumWorkGroups0",     '4',      '4',      "ByValue",        "U32"); ka_size += 4
        kStr += self.v2Argument(                  "NumWorkGroups1",     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        if kernel["PersistentKernel"]:
            kStr += self.v2Argument("MagicNumberProblemNumGroupTiles0",     '4',    '4',      "ByValue",      "U32"); ka_size += 4
            kStr += self.v2Argument("MagicShiftProblemNumGroupTiles0",      '4',    '4',      "ByValue",      "U32"); ka_size += 4
            kStr += self.v2Argument(              "GridNumWorkGroups0",     '4',    '4',      "ByValue",      "U32"); ka_size += 4
            if kernel["PersistentKernelAlongBatch"]:
                kStr += self.v2Argument(                "NumWorkGroups2",     '4',    '4',    "ByValue",      "U32"); ka_size += 4
                kStr += self.v2Argument("MagicNumProblemNumGroupTiles0By1",   '4',    '4',    "ByValue",      "U32"); ka_size += 4
                kStr += self.v2Argument("MagicShiftProblemNumGroupTiles0By1", '4',    '4',    "ByValue",      "U32"); ka_size += 4

        kStr += self.v2Argument(                   "NumFullBlocks",     '4',      '4',      "ByValue",        "U32"); ka_size += 4
        kStr += self.v2Argument(                   "WgmRemainder1",     '4',      '4',      "ByValue",        "U32"); ka_size += 4
        kStr += self.v2Argument(        "MagicNumberWgmRemainder1",     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        kStr += self.v2Argument("OffsetD", '4', '4', "ByValue", "U32"); ka_size += 4
        kStr += self.v2Argument("OffsetC", '4', '4', "ByValue", "U32"); ka_size += 4
        kStr += self.v2Argument("OffsetA", '4', '4', "ByValue", "U32"); ka_size += 4
        kStr += self.v2Argument("OffsetB", '4', '4', "ByValue", "U32"); ka_size += 4

        kStr += self.v2Argument(                         "padding",     '4',      '4',      "ByValue",        "U32"); ka_size += 4

        kStr += "    CodeProps:\n"
        kStr += "      KernargSegmentSize: %u%s" % (ka_size, writer.endLine)
        kStr += "      GroupSegmentFixedSize: %u%s" % ( group_segment_size, writer.endLine )
        kStr += "      PrivateSegmentFixedSize: %u%s" % ( 0, writer.endLine )
        kStr += "      KernargSegmentAlign:  %u%s" % ( 8, writer.endLine )
        kStr += "      WavefrontSize:        %u%s" % ( kernel["WavefrontSize"], writer.endLine )
        kStr += "      NumSGPRs:             %u%s" % ( totalSgprs, writer.endLine )
        kStr += "      NumVGPRs:             %u%s" % ( totalVgprs, writer.endLine )
        kStr += "      MaxFlatWorkGroupSize: %u%s" % ( kernel["SubGroup0"] * kernel["SubGroup1"] * kernel["LocalSplitU"], writer.endLine )
        kStr += ".end_amd_amdgpu_hsa_metadata\n"

        return kStr


class SignatureCOV3(Signature):
    kernel = {"CodeObjectVersion": "V3"}

    def v3Argument(self, name, size, offset, valueKind, valueType, AddrSpaceQual = None):
        kStr = ""
        kStr += "      - .name:            %s\n" % name
        kStr += "        .size:            %s\n" % size
        kStr += "        .offset:          %s\n" % offset
        kStr += "        .value_kind:      %s\n" % valueKind
        kStr += "        .value_type:      %s\n" % valueType
        if AddrSpaceQual != None:
            kStr += "        .address_space:   %s\n" % AddrSpaceQual
        return kStr

    def __call__(self, writer):
        kernel = writer.kernel

        kStr = self.commentHeader()

        # begin kernel descriptor
        kStr += ".amdgcn_target \"amdgcn-amd-amdhsa--gfx%s\"%s" \
            % ("".join(map(str,writer.version)), writer.endLine)

        kStr += ".text%s" % writer.endLine
        kStr += ".protected %s%s" % (writer.kernelName, writer.endLine)
        kStr += ".globl %s%s" % (writer.kernelName, writer.endLine)
        kStr += ".p2align 8%s" % writer.endLine
        kStr += ".type %s,@function%s" % (writer.kernelName, writer.endLine)

        kStr += ".section .rodata,#alloc%s" % writer.endLine
        kStr += ".p2align 6%s" % writer.endLine
        tWord = "amdhsa_kernel"
        kStr += ".%s %s%s" % (tWord, writer.kernelName, writer.endLine)
        tWord = ".amdhsa_user_sgpr_kernarg_segment_ptr"
        kStr += "  %s 1%s" % (tWord, writer.endLine)

        # kern arg size
        kernArgReg = 0
        kernArgReg += 3*writer.rpga
        kernArgReg += max(1,int(writer.bpeAB/4)) # alpha
        if kernel["ProblemType"]["UseBeta"]:
            kernArgReg += max(1,int(writer.bpeCexternal/4)) # beta
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsA"]) # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsB"]) # strides
        if not kernel["ProblemType"]["UseInitialStridesAB"]:
            kernArgReg -= 2 # strides
        if not kernel["ProblemType"]["UseInitialStridesCD"]:
            kernArgReg -= 2 # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesSummation"]
        kernArgReg += kernel["ProblemType"]["NumIndicesC"]
        if globalParameters["DebugKernel"]:
            kernArgReg += writer.rpga # debug buffer
        # kernArgBytes = kernArgReg * 4 # bytes/reg

        # register allocation
        totalVgprs = writer.vgprPool.size()
        totalSgprs = writer.sgprPool.size()
        tWord = ".amdhsa_next_free_vgpr"
        kStr += "  %s %u // vgprs%s" % (tWord, totalVgprs, writer.endLine)
        tWord = ".amdhsa_next_free_sgpr"
        kStr += "  %s %u // sgprs%s" % (tWord, totalSgprs, writer.endLine)

        tWord = ".amdhsa_group_segment_fixed_size"
        if kernel["AggressivePerfMode"]>=2 and kernel["ProblemType"]["DataType"].isDouble() and \
            kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 4 and kernel["WorkGroup"] == [16,16,1]:
            group_segment_size = 32768 # Pad LDS to ensure we run exactly two waves
        else:
            group_segment_size = kernel["LdsNumElements"] * writer.bpeAB
        kStr += "  %s %u // lds bytes%s" % ( tWord, group_segment_size, writer.endLine )

        if writer.archCaps["HasWave32"]:
            if kernel["WavefrontSize"] == 32:
                kStr += "  .amdhsa_wavefront_size32 1 // 32-thread wavefronts%s" % writer.endLine
            else:
                kStr += "  .amdhsa_wavefront_size32 0 // 64-thread wavefronts%s" % writer.endLine

        # other
        kStr += "  .amdhsa_private_segment_fixed_size 0%s" % writer.endLine
        kStr += "  .amdhsa_system_sgpr_workgroup_id_x 1%s" % writer.endLine
        kStr += "  .amdhsa_system_sgpr_workgroup_id_y 1%s" % writer.endLine
        kStr += "  .amdhsa_system_sgpr_workgroup_id_z %u%s" % (1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0, writer.endLine)
        kStr += "  .amdhsa_system_vgpr_workitem_id 0%s" % writer.endLine
        kStr += ".end_amdhsa_kernel%s" % writer.endLine
        kStr += ".text%s" % writer.endLine

        kStr += writer.comment3("Optimizations and Config:")
        kStr += writer.comment1("ThreadTile= %u x %u" % (kernel["ThreadTile0"], kernel["ThreadTile1"]))
        kStr += writer.comment1("SubGroup= %u x %u" % (kernel["SubGroup0"], kernel["SubGroup1"]))
        kStr += writer.comment1("VectorWidth=%u" % (kernel["VectorWidth"]))
        kStr += writer.comment1("GlobalLoadVectorWidthA=%u, GlobalLoadVectorWidthB=%u" % (kernel["GlobalLoadVectorWidthA"], kernel["GlobalLoadVectorWidthB"]))
        kStr += writer.comment1("DirectToLdsA=%s" % kernel["DirectToLdsA"])
        kStr += writer.comment1("DirectToLdsB=%s" % kernel["DirectToLdsB"])
        kStr += writer.comment1("UseSgprForGRO=%s" % kernel["_UseSgprForGRO"])

        srcValueType = getSrcValueType(kernel, "V3")
        dstValueType = getDstValueType(kernel, "V3")
        cptValueType = getCptValueType(kernel, "V3")
        cptByte = getCptByte(kernel)
        # cptSize = getCptSize(kernel)
        # cptAlign = getCptAlign(kernel)

        # Codeobject V3 metadata
        kStr += ".amdgpu_metadata\n"
        kStr += "---\n"
        kStr += "amdhsa.version:\n"
        kStr += "  - 1\n"
        kStr += "  - 0\n"
        kStr += "amdhsa.kernels:\n"
        kStr += "  - .name: %s%s" % (writer.kernelName, writer.endLine)
        kStr += "    .symbol: '%s.kd'%s" % (writer.kernelName, writer.endLine)
        kStr += "    .language:                   %s%s" % ("OpenCL C", writer.endLine)
        kStr += "    .language_version:%s" % writer.endLine
        kStr += "      - 2%s" % writer.endLine
        kStr += "      - 0%s" % writer.endLine
        kStr += "    .args:%s" % writer.endLine
        offset = 0

        if globalParameters["DebugKernel"]:
            kStr += self.v3Argument(                    'AddressDbg',     '8', offset, "global_buffer","struct", "generic"); offset += 8

        kStr += self.v3Argument(                           'sizeC',     '8', offset,      "by_value",        "u64"); offset += 8
        kStr += self.v3Argument(                           'sizeA',     '8', offset,      "by_value",        "u64"); offset += 8
        kStr += self.v3Argument(                           'sizeB',     '8', offset,      "by_value",        "u64"); offset += 8

        kStr += self.v3Argument(                               'D',     '8', offset, "global_buffer", dstValueType, "generic"); offset += 8
        kStr += self.v3Argument(                               'C',     '8', offset, "global_buffer", dstValueType, "generic"); offset += 8
        kStr += self.v3Argument(                               'A',     '8', offset, "global_buffer", srcValueType, "generic"); offset += 8
        kStr += self.v3Argument(                               'B',     '8', offset, "global_buffer", srcValueType, "generic"); offset += 8

        useSize = max(4, cptByte)
        kStr += self.v3Argument(                             "alpha", useSize, offset,      "by_value", cptValueType); offset += useSize
        if kernel["ProblemType"]["UseBeta"]:
            kStr += self.v3Argument(                          "beta", useSize, offset,      "by_value", cptValueType); offset += useSize

        for i in range(0, writer.numSgprStridesD):
            kStr += self.v3Argument(                   "strideD%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

        for i in range(0, writer.numSgprStridesC):
            kStr += self.v3Argument(                   "strideC%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

        for i in range(0, writer.numSgprStridesA):
            kStr += self.v3Argument(                   "strideA%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

        for i in range(0, writer.numSgprStridesB):
            kStr += self.v3Argument(                   "strideB%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

        for i in range(0, writer.numSgprSizesFree):
            kStr += self.v3Argument(                 "SizesFree%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

        for i in range(0, writer.numSgprSizesSum):
            kStr += self.v3Argument(                  "SizesSum%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

        for magicName in writer.sumMagicParms:
            kStr += self.v3Argument(     "MagicNumberSize%s"%magicName,     '4', offset,      "by_value",        "u32"); offset += 4
            kStr += self.v3Argument(      "MagicShiftSize%s"%magicName,     '4', offset,      "by_value",        "u32"); offset += 4

        for idxChar in kernel["PackedC0IdxChars"][:-1]:
            kStr += self.v3Argument(     "MagicNumberSize%s"%idxChar,     '4', offset,      "by_value",        "u32"); offset += 4
            kStr += self.v3Argument(      "MagicShiftSize%s"%idxChar,     '4', offset,      "by_value",        "u32"); offset += 4

        for idxChar in kernel["PackedC1IdxChars"][:-1]:
            kStr += self.v3Argument(     "MagicNumberSize%s"%idxChar,     '4', offset,      "by_value",        "u32"); offset += 4
            kStr += self.v3Argument(      "MagicShiftSize%s"%idxChar,     '4', offset,      "by_value",        "u32"); offset += 4

        for idx in kernel["ProblemType"]["IndicesSummation"]:
          for tc in ('A','B'):
            for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
              (freeDim, sumDim, padStart, padEnd) = zp
              if sumDim == idx:
                freeDimChar = globalParameters["IndexChars"][freeDim]
                sumDimChar  = globalParameters["IndexChars"][sumDim]
                # These will eventually be read as kernel args:
                kStr += self.v3Argument(   "PadStart%s%s%s"%(tc, freeDimChar, sumDimChar),     '4', offset,      "by_value",        "u32"); offset += 4
                kStr += self.v3Argument(     "PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar),     '4', offset,      "by_value",        "u32"); offset += 4

        kStr += self.v3Argument(              "OrigStaggerUIter",       '4', offset,      "by_value",        "i32"); offset += 4

        kStr += self.v3Argument(                  "NumWorkGroups0",     '4', offset,      "by_value",        "u32"); offset += 4
        kStr += self.v3Argument(                  "NumWorkGroups1",     '4', offset,      "by_value",        "u32"); offset += 4

        if kernel["PersistentKernel"]:
            kStr += self.v3Argument("MagicNumberProblemNumGroupTiles0",   '4', offset,    "by_value",        "u32"); offset += 4
            kStr += self.v3Argument("MagicShiftProblemNumGroupTiles0",    '4', offset,    "by_value",        "u32"); offset += 4
            kStr += self.v3Argument(              "GridNumWorkGroups0",   '4', offset,    "by_value",        "u32"); offset += 4
            if kernel["PersistentKernelAlongBatch"]:
                kStr += self.v3Argument(                "NumWorkGroups2",   '4', offset,  "by_value",        "u32"); offset += 4
                kStr += self.v3Argument("MagicNumProblemNumGroupTiles0By1", '4', offset,  "by_value",        "u32"); offset += 4
                kStr += self.v3Argument("MagicShiftProblemNumGroupTiles0By1", '4', offset,"by_value",        "u32"); offset += 4

        kStr += self.v3Argument(                   "NumFullBlocks",     '4', offset,      "by_value",        "u32"); offset += 4
        kStr += self.v3Argument(                   "WgmRemainder1",     '4', offset,      "by_value",        "u32"); offset += 4
        kStr += self.v3Argument(        "MagicNumberWgmRemainder1",     '4', offset,      "by_value",        "u32"); offset += 4

        kStr += self.v3Argument("OffsetD", '4', offset, "by_value", "u32"); offset += 4
        kStr += self.v3Argument("OffsetC", '4', offset, "by_value", "u32"); offset += 4
        kStr += self.v3Argument("OffsetA", '4', offset, "by_value", "u32"); offset += 4
        kStr += self.v3Argument("OffsetB", '4', offset, "by_value", "u32"); offset += 4

        kStr += self.v3Argument(                         "padding",     '4', offset,      "by_value",        "u32"); offset += 4
        kStr += "    .group_segment_fixed_size:   %u%s" % ( group_segment_size, writer.endLine ) #XXXXXX
        kStr += "    .kernarg_segment_align:      %u%s" % ( 8, writer.endLine )
        kStr += "    .kernarg_segment_size:       %u%s" % (((offset+7)//8)*8, writer.endLine) # round up to .kernarg_segment_align
        kStr += "    .max_flat_workgroup_size:    %u%s" % ( kernel["SubGroup0"] * kernel["SubGroup1"] * kernel["LocalSplitU"], writer.endLine )
        kStr += "    .private_segment_fixed_size: %u%s" % ( 0, writer.endLine )
        kStr += "    .sgpr_count:                 %u%s" % ( totalSgprs, writer.endLine )
        kStr += "    .sgpr_spill_count:           %u%s" % ( 0, writer.endLine )
        kStr += "    .vgpr_count:                 %u%s" % ( totalVgprs, writer.endLine )
        kStr += "    .vgpr_spill_count:           %u%s" % ( 0, writer.endLine )
        kStr += "    .wavefront_size:             %u%s" % ( kernel["WavefrontSize"], writer.endLine )

        kStr += "...\n"

        kStr += ".end_amdgpu_metadata\n"

        kStr += "%s:%s" % (writer.kernelName, writer.endLine)

        return kStr
