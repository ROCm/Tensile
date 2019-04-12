################################################################################
# Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

try:
  basestring
except NameError:
  basestring = str

class DataType:
    """ 
    Data Type (new)
    Uses a nested dictionary to organize the DataType and Properties
    """
    # single        = 0
    # double        = 1
    # complexSingle = 2
    # complexDouble = 3
    # half          = 4
    # int8x4        = 5
    # int32         = 6
    # num           = 7
    # none          = 8

    # data type properties
    # idxChar    = 0
    # idxReg     = 1
    # idxOpenCL  = 2
    # idxHIP     = 3
    # idxLibType = 4
    # idxLibEnum = 5
    #    char, reg,    ocl,       hip,       libType,                 libEnum
    
    """
    Changed older properties list of lists to list of dictionaries
    While the inner keys (char, reg, etc) correspond with the data type properties values
    """

     ########################################
    """
    For init, note that value is a letter
    """
    def __init__( self, value ):
        #print("init value:", value)
        if isinstance(value, int):
            self.value = value
        elif isinstance(value, basestring):
            import pdb
            pdb.set_trace()
            self.value = lookup[value.lower()]
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
        return self.properties[self.value]['libType']
    def getLibString(self):
        return self.properties[self.value]['libEnum']

    ########################################
    def zeroString(self, language, vectorWidth):
        """
        Returns a string containing the data output format, depending on programming language 
        and in the case of complex numbers, the vector width
        """
        if language == "HIP":
            if self.value == 2: #complex float, can also just set this to 2
                return "make_float2(0.f, 0.f)"
            if self.value == 3: #self.complexDouble: previously was the value
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
        if self.value == dt.half or self.value == dt.single or self.value == dt.double or self.value == dt.int8x4 or self.value == dt.int32: 
        #self.half or self.value == self.single or self.value == self.double or self.value == self.int8x4 or self.value == self.int32:
            return True
        else:
            return False
    def isComplex(self):
        return not self.isReal()
    def isDouble(self):
        return self.value == dt.double or self.value == dt.complexDouble
    def isSingle(self):
        return self.value == dt.single or self.value == dt.complexSingle
    def isHalf(self):
        return self.value == dt.half
    def isInt32(self):
        return self.value == dt.int32
    def isInt8x4(self):
        return self.value == dt.int8x4
    def isNone(self):
        return self.value == None

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
        
      # ["S",    1,   "float",   "float",        "float",                "tensileDataTypeFloat"        ],
      # ["D",    2,   "double",  "double",       "double",               "tensileDataTypeDouble"       ],
      # ["C",    2,   "float2",  "float2",       "TensileComplexFloat",  "tensileDataTypeComplexFloat" ],
      # ["Z",    4,   "double2", "double2",      "TensileComplexDouble", "tensileDataTypeComplexDouble"],
      # ["H",    0.5, "ERROR",   "tensile_half", "TensileHalf",          "tensileDataTypeHalf"         ],
      # ["4xi8", 1,   "ERROR",   "uint32_t",     "TensileInt8x4",        "tensileDataTypeInt8x4"       ],
      # ["I",    1,   "ERROR",   "int32_t",      "TensileInt32",         "tensileDataTypeInt32"        ]
properties = [{'char': 'S', 'name': 'single', 'enum': 'Float', 'reg': 1, 'ocl': 'float', 'hip': 'float', 'libType': 'float', 'libEnum': 'tensileDataTypeFloat'},
        {'char': 'D', 'name': 'double', 'enum': 'Double', 'reg': 2, 'ocl': 'double', 'hip': 'double', 'libType': 'double', 'libEnum': 'tensileDataTypeDouble'},
        {'char': 'C', 'name': 'complexSingle', 'enum': 'ComplexFloat', 'reg': 2, 'ocl': 'float2', 'hip': 'float2', 'libType': 'TensileComplexFloat', 'libEnum': 'tensileDataTypeComplexFloat'},
        {'char': 'Z', 'name': 'complexDouble', 'enum': 'ComplexDouble', 'reg': 4, 'ocl': 'double2', 'hip': 'double2', 'libType': 'TensileComplexDouble', 'libEnum': 'tensileDataTypeComplexDouble'},
        {'char': 'H', 'name': 'half', 'enum': 'Half', 'reg': 0.5, 'ocl': 'ERROR', 'hip': 'tensile_half', 'libType': 'TensileHalf', 'libEnum': 'tensileDataTypeHalf'},
        {'char': '4xi8', 'name': 'int8x4', 'enum': 'Int8', 'reg': 1, 'ocl': 'ERROR', 'hip': 'uint32_t', 'libType': 'TensileInt8x4', 'libEnum': 'tensileDataTypeInt8x4'},
        {'char': 'I', 'name': 'int32', 'enum': 'Int32', 'reg': 1, 'ocl': 'ERROR', 'hip': 'int32_t', 'libType': 'TensileInt32', 'libEnum': 'tensileDataTypeInt32'}]
lookup = {}
dt = DataType(properties[0]['char'])
for i,e in enumerate(properties):
    setattr(dt, e['name'], i)
    for k in ['char','name','enum']:
        lookup[e[k]] = i