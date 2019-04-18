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
import pytest
from Tensile.DataType import DataType

def test_init_single():
    assert DataType('S').value == DataType('single').value == DataType('Float').value == DataType('tensileDataTypeFloat').value == 0

def test_init_double():
    assert DataType('D').value == DataType('double').value == DataType('Double').value == DataType('tensileDataTypeDouble').value == 1
    
def test_init_complexSingle():
    assert DataType('C').value == DataType('complexSingle').value == DataType('complexFloat').value == DataType('tensileDataTypeComplexFloat').value == 2

def test_init_complexDouble():
    assert DataType('Z').value == DataType('complexDouble').value == DataType('complexDouble').value == DataType('tensileDataTypeComplexDouble').value == 3

def test_init_half():
    assert DataType('H').value == DataType('half').value == DataType('Half').value == DataType('tensileDataTypeHalf').value == 4

def test_init_i8():
    assert DataType('4xi8').value == DataType('int8x4').value == DataType('Int8').value == DataType('tensileDataTypeInt8x4').value == 5

def test_init_i32():
    assert DataType('I').value == DataType('int32').value == DataType('Int32').value == DataType('tensileDataTypeInt32').value == 6

def test_single():
    obj = DataType(0)
    assert obj.toChar() == 'S'
    assert obj.toName() == 'single'
    assert obj.toEnum() == 'Float'
    assert obj.toOpenCL() == 'float'
    assert obj.toHIP() == 'float'
    assert obj.toDevice("") == 'float'
    assert obj.toCpp() == 'float'
    assert obj.getLibString() == 'tensileDataTypeFloat'
    assert obj.numBytes() == 4
    assert obj.isReal()
    
def test_double():
    obj = DataType(1)
    assert obj.toChar() == 'D'
    assert obj.toName() == 'double'
    assert obj.toEnum() == 'Double'
    assert obj.toOpenCL() == 'double'
    assert obj.toHIP() == 'double'
    assert obj.toDevice("") == 'double'
    assert obj.toCpp() == 'double'
    assert obj.getLibString() == 'tensileDataTypeDouble'
    assert obj.numBytes() == 8    
    assert obj.isReal()
    
def test_complexSingle():
    obj = DataType(2)
    assert obj.toChar() == 'C'
    assert obj.toName() == 'complexSingle'
    assert obj.toEnum() == 'ComplexFloat'
    assert obj.toOpenCL() == 'float2'
    assert obj.toHIP() == 'float2'
    assert obj.toDevice("") == 'float2'
    assert obj.toCpp() == 'TensileComplexFloat'
    assert obj.getLibString() == 'tensileDataTypeComplexFloat'
    assert obj.numBytes() == 8
    assert not obj.isReal()
    
def test_complexDouble():
    obj = DataType(3)
    assert obj.toChar() == 'Z'
    assert obj.toName() == 'complexDouble'
    assert obj.toEnum() == 'ComplexDouble'
    assert obj.toOpenCL() == 'double2'
    assert obj.toHIP() == 'double2'
    assert obj.toDevice("") == 'double2'
    assert obj.toCpp() == 'TensileComplexDouble'
    assert obj.getLibString() == 'tensileDataTypeComplexDouble'
    assert obj.numBytes() == 16
    assert not obj.isReal()
    
def test_half():
    obj = DataType(4)
    assert obj.toChar() == 'H'
    assert obj.toName() == 'half'
    assert obj.toEnum() == 'Half'
    assert obj.toOpenCL() == 'ERROR'
    assert obj.toHIP() == 'tensile_half'
    assert obj.toDevice("OCL") == 'ERROR'
    assert obj.toDevice("") == 'tensile_half'
    assert obj.toCpp() == 'TensileHalf'
    assert obj.getLibString() == 'tensileDataTypeHalf'   
    assert obj.numBytes() == 2    
    assert obj.isReal()
    
def test_int8():
    obj = DataType(5)
    assert obj.toChar() == '4xi8'
    assert obj.toName() == 'int8x4'
    assert obj.toEnum() == 'Int8'
    assert obj.toOpenCL() == 'ERROR'
    assert obj.toHIP() == 'uint32_t'
    assert obj.toDevice("OCL") == 'ERROR'
    assert obj.toDevice("") == 'uint32_t'
    assert obj.toCpp() == 'TensileInt8x4'
    assert obj.getLibString() == 'tensileDataTypeInt8x4'
    assert obj.numBytes() == 4
    assert obj.isReal()
    
def test_int32():
    obj = DataType(6)
    assert obj.toChar() == 'I'
    assert obj.toName() == 'int32'
    assert obj.toEnum() == 'Int32'
    assert obj.toOpenCL() == 'ERROR'
    assert obj.toHIP() == 'int32_t'
    assert obj.toDevice("OCL") == 'ERROR'
    assert obj.toDevice("") == 'int32_t'
    assert obj.toCpp() == 'TensileInt32'
    assert obj.getLibString() == 'tensileDataTypeInt32'
    assert obj.numBytes() == 4
    assert obj.isReal()

def test_eq():
    assert DataType('single').__eq__(DataType(0))
    assert not DataType('double').__eq__(DataType('tensileDataTypeComplexFloat'))
    assert DataType('ComplexDouble').__eq__(DataType('Z'))
    assert not DataType('tensileDataTypeInt8x4').__eq__(DataType('int32'))

def test_bounds():
    with pytest.raises(Exception) as e_info:
        DataType(10)