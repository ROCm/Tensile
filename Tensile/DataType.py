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

from __future__ import print_function
from Common import printExit

class dt:
    """ 
    Data Type (new, called dt for now)
    Uses a nested dictionary to organize the DataTypes and Properties
    """
    single        = 0
    double        = 1
    complexSingle = 2
    complexDouble = 3
    half          = 4
    int8x4        = 5
    int32         = 6
    num           = 7
    none          = 8

    # data type properties
    # idxChar    = 0
    # idxReg     = 1
    # idxOpenCL  = 2
    # idxHIP     = 3
    # idxLibType = 4
    # idxLibEnum = 5
    #    char, reg,    ocl,       hip,       libType,                 libEnum
    properties = { 0: {'char': 'S', 'reg': 1, 'ocl': 'float', 'hip': 'float', 'libType': 'float', 'libEnum': 'tensileDataTypeFloat'},
                 1: {'char': 'D', 'reg': 2, 'ocl': 'double', 'hip': 'double', 'libType': 'double', 'libEnum': 'tensileDataTypeDouble'},
                 2: {'char': 'C', 'reg': 2, 'ocl': 'float2', 'hip': 'float2', 'libType': 'TensileComplexFloat', 'libEnum': 'tensileDataTypeComplexFloat'},
                 3: {'char': 'Z', 'reg': 4, 'ocl': 'double2', 'hip': 'double2', 'libType': 'TensileComplexDouble', 'libEnum': 'tensileDataTypeComplexDouble'},
                 4: {'char': 'H', 'reg': 0.5, 'ocl': 'ERROR', 'hip': 'tensile_half', 'libType': 'TensileHalf', 'libEnum': 'tensileDataTypeHalf'},
                 5: {'char': '4xi8', 'reg': 1, 'ocl': 'ERROR', 'hip': 'uint32_t', 'libType': 'TensileInt8x4', 'libEnum': 'tensileDataTypeInt8x4'},
                 6: {'char': 'I', 'reg': 1, 'ocl': 'ERROR', 'hip': 'int32_t', 'libType': 'TensileInt32', 'libEnum': 'tensileDataTypeInt32'}, 
                 7: {'char': 'Num', 'reg': 2, 'ocl': '?', 'hip': '?', 'libType': '?', 'libEnum': '?'},
                 8: {'char': 'None', 'reg': 0, 'ocl': '?', 'hip': '?', 'libType': '?', 'libEnum': '?'}
               }
      # ["S",    1,   "float",   "float",        "float",                "tensileDataTypeFloat"        ],
      # ["D",    2,   "double",  "double",       "double",               "tensileDataTypeDouble"       ],
      # ["C",    2,   "float2",  "float2",       "TensileComplexFloat",  "tensileDataTypeComplexFloat" ],
      # ["Z",    4,   "double2", "double2",      "TensileComplexDouble", "tensileDataTypeComplexDouble"],
      # ["H",    0.5, "ERROR",   "tensile_half", "TensileHalf",          "tensileDataTypeHalf"         ],
      # ["4xi8", 1,   "ERROR",   "uint32_t",     "TensileInt8x4",        "tensileDataTypeInt8x4"       ],
      # ["I",    1,   "ERROR",   "int32_t",      "TensileInt32",         "tensileDataTypeInt32"        ]
  

    ########################################
    def __init__( self, value ):
        if isinstance(value, int):
            self.value = value
        elif isinstance(value, basestring):
            for propertiesIdx in range(0,8): #probably should be 6
                for dataTypeIdx in range(0,self.num):
                    if value.lower() == self.properties[dataTypeIdx][propertiesIdx].lower():
                        self.value = dataTypeIdx
                        return
        elif isinstance(value, DataType):
            self.value = value.value
        else:
            printExit("initializing DataType to %s %s" % (str(type(value)), str(value)) )


    ########################################
    def toChar(self):
        return self.properties[self.value]['char']
    def toOpenCL(self):
        return self.properties[self.value]['ocl']
    def toHIP(self):
        return self.properties[self.value]['hip']
    def toDevice(self, language):
        if language == "OCL":
            return self.toOpenCL()
        else:
            return self.toHIP()
    def toCpp(self):
        return self.properties[self.value]['idxLibType']
    def getLibString(self):
        return self.properties[self.value]['idxLibEnum']

    ########################################
    def zeroString(self, language, vectorWidth):
        if language == "HIP":
            if self.value == list(self.properties.keys())[2]: #complex float, can also just set this to 2
                return "make_float2(0.f, 0.f)"
            if self.value == list(self.properties.keys())[3]: #self.complexDouble:
                return "make_double2(0.0, 0.0)"

        zeroString = "("
        zeroString += self.toDevice(language)
        if vectorWidth > 1:
            zeroString += str(vectorWidth)
        zeroString += ")("

        """
        if self.value == self.half:
            single = "0"
            vectorWidth = 1
        elif self.value == self.single:
            single = "0.f"
        elif self.value == self.double:
            single = "0.0"
        elif self.value == self.complexSingle:
            single = "0.f, 0.f"
        elif self.value == self.complexDouble:
            single = "0.0, 0.0"
        """
        zeroString += "0"
        zeroString += ")"
        return zeroString

  ########################################
    def isReal(self):
        if self.value == 4 or self.value == 0 or self.value == 1 or self.value == 5 or self.value == 6 
        #self.half or self.value == self.single or self.value == self.double or self.value == self.int8x4 or self.value == self.int32:
            return True
        else:
            return False
    def isComplex(self):
        return not self.isReal()
    def isDouble(self):
        return self.value == 1 or self.value == 3
    def isSingle(self):
        return self.value == 0
    def isHalf(self):
        return self.value == 4
    def isInt32(self):
        return self.value == 6
    def isInt8x4(self):
        return self.value == 5
    def isNone(self):
        return self.value == 8

  ########################################
    def numRegisters(self):
        return self.properties[self.value]['reg']
    def numBytes(self):
        return int(self.numRegisters() * 4)
    def flopsPerMac(self):
        return 2 if self.isReal() else 8

    def __str__(self):
        return self.toChar()

    def __repr__(self):
        return self.__str__()

    def getAttributes(self):
        return (self.value)
    def __hash__(self):
        return hash(self.getAttributes())
    def __eq__(self, other):
        return isinstance(other, DataType) and self.getAttributes() == other.getAttributes()
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result