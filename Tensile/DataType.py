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

import functools

@functools.total_ordering
class DataType:
    """ 
    Data Type (new)
    Uses a list of dictionaries to organize the DataType and Properties for the kernels
    Changed older properties list of lists to list of dictionaries
    The inner keys (char, reg, etc) correspond with the data type properties values
    Lookup table is used to store row numbers of a specific property
    """

    properties = [
        {
            'char': 'S',
            'name': 'single',
            'nameAbbrev': 'f32',
            'enum': 'Float',
            'reg': 1,
            'ocl': 'float',
            'hip': 'float',
            'libType': 'float',
            'libEnum': 'tensileDataTypeFloat',
            'isIntegral': False,
            'isComplex': False,
            'packing': 1,
        },
        {
            'char': 'D',
            'name': 'double',
            'nameAbbrev': 'f64',
            'enum': 'Double',
            'reg': 2,
            'ocl': 'double',
            'hip': 'double',
            'libType': 'double',
            'libEnum': 'tensileDataTypeDouble',
            'isIntegral': False,
            'isComplex': False,
            'packing': 1,
        },
        {
            'char': 'C',
            'name': 'complexSingle',
            'nameAbbrev': 'f32c',
            'enum': 'ComplexFloat',
            'reg': 2,
            'ocl': 'float2',
            'hip': 'TensileComplexFloat',
            'libType': 'TensileComplexFloat',
            'libEnum': 'tensileDataTypeComplexFloat',
            'isIntegral': False,
            'isComplex': True,
            'packing': 1,
        },
        {
            'char': 'Z',
            'name': 'complexDouble',
            'nameAbbrev': 'f64c',
            'enum': 'ComplexDouble',
            'reg': 4,
            'ocl': 'double2',
            'hip': 'TensileComplexDouble',
            'libType': 'TensileComplexDouble',
            'libEnum': 'tensileDataTypeComplexDouble',
            'isIntegral': False,
            'isComplex': True,
            'packing': 1,
        },
        {
            'char': 'H',
            'name': 'half',
            'nameAbbrev': 'f16',
            'enum': 'Half',
            'reg': 0.5,
            'ocl': 'ERROR',
            'hip': 'tensile_half',
            'libType': 'TensileHalf',
            'libEnum': 'tensileDataTypeHalf',
            'isIntegral': False,
            'isComplex': False,
            'packing': 1,
        },
        {
            'char': '4xi8',
            'name': 'int8x4',
            'nameAbbrev': 'i8',
            'enum': 'Int8x4',
            'reg': 1,
            'ocl': 'ERROR',
            'hip': 'uint32_t',
            'libType': 'TensileInt8x4',
            'libEnum': 'tensileDataTypeInt8x4',
            'isIntegral': True,
            'isComplex': False,
            'packing': 4,
        },
        {
            'char': 'I',
            'name': 'int32',
            'nameAbbrev': 'i32',
            'enum': 'Int32',
            'reg': 1,
            'ocl': 'ERROR',
            'hip': 'int32_t',
            'libType': 'TensileInt32',
            'libEnum': 'tensileDataTypeInt32',
            'isIntegral': True,
            'isComplex': False,
            'packing': 1,
        },
        {
            'char': 'B',
            'name': 'bfloat16',
            'nameAbbrev': 'bf16',
            'enum': 'BFloat16',
            'reg': 0.5,
            'ocl': 'ERROR',
            'hip': 'tensile_bfloat16',
            'libType': 'tensile_bfloat16',
            'libEnum': 'tensileDataTypeBFloat16',
            'isIntegral': False,
            'isComplex': False,
            'packing': 1,
        },
    ]
    lookup = {}    
            
    def __init__(self, value):
        if isinstance(value, int):
            self.value = value
        elif isinstance(value, str):
            self.value = DataType.lookup[value.lower()]
        elif isinstance(value, DataType):
            self.value = value.value
        else:
            raise RuntimeError("initializing DataType to {0} {1}".format(str(type(value)), str(value)))

        self.properties = DataType.properties[self.value]
   
    def toChar(self):
        return self.properties['char']
    def toName(self):
        return self.properties['name']
    def toNameAbbrev(self):
        return self.properties['nameAbbrev']
    def toEnum(self):
        return self.properties['enum']
    def toOpenCL(self):
        return self.properties['ocl']
    def toHIP(self):
        return self.properties['hip']
    def toDevice(self, language):
        if language == "OCL":
            return self.toOpenCL()
        else:
            return self.toHIP()
    def toCpp(self):
        return self.properties['libType']
    def getLibString(self):
        return self.properties['libEnum']

    ########################################
    def zeroString(self, language, vectorWidth):
        """
        Returns a string containing the data output format, depending on programming language 
        and in the case of complex numbers, the vector width. 
        """
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

    def isReal(self):
        return not self.isComplex()
    def isComplex(self):
        return self.properties['isComplex']
    def isDoubleComplex(self):
        return self.value == DataType.complexDouble
    def isSingleComplex(self):
        return self.value == DataType.complexSingle
    def isDouble(self):
        return self.value == DataType.double
    def isSingle(self):
        return self.value == DataType.single
    def isHalf(self):
        return self.value == DataType.half
    def isInt32(self):
        return self.value == DataType.int32
    def isInt8x4(self):
        return self.value == DataType.int8x4
    def isBFloat16(self):
        return self.value == DataType.bfloat16
    def isNone(self):
        return self.value == None

    def numRegisters(self):
        return self.properties['reg']
    def numBytes(self):
        return int(self.numRegisters() * 4)
    def flopsPerMac(self):
        return 2 if self.isReal() else 8

    def state(self): return self.toEnum()

    def __str__(self):
        return self.toChar()
    def __repr__(self):
        return self.__str__()

    def getAttributes(self):
        return (self.value,)

    def __hash__(self):
        return hash(self.getAttributes())

    def __eq__(self, other):
        if not isinstance(other, DataType):
            return NotImplemented

        return self.getAttributes() == other.getAttributes()

    def __lt__(self, other):
        if not isinstance(other, DataType):
            return NotImplemented

        return self.getAttributes() < other.getAttributes()

    # Other operands are provided by functools.total_ordering.

def populateLookupTable(properties,lookup):
    """
    Populates Lookup Table with the corresponding row number for each DataType. The row number
    is assigned to self.value when a DataType object is called 
    """
    for i,e in enumerate(properties):
        setattr(DataType, e['name'], i)
        for k in ['name','char','enum','libEnum']:
            lookupKey = e[k].lower()
            if lookupKey in lookup and lookup[lookupKey] != i:
                raise RuntimeError("Duplicate key {1} in property '{0}'".format(k,lookupKey)) 
            lookup[lookupKey] = i        
    
populateLookupTable(DataType.properties,DataType.lookup)

