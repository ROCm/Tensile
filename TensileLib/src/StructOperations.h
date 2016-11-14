/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#ifndef STRUCT_OPERATIONS_H
#define STRUCT_OPERATIONS_H

#include "Tensile.h"
#include "Tools.h"
//#include "Solution.h"
#include <string>

namespace Tensile {

/*******************************************************************************
 * enum toString
 ******************************************************************************/
std::string toString( TensileStatus code );
std::string toString( TensileDataType dataType );
std::string toString( TensileOperationType type );
std::string toString( TensileProblem problem );

template<typename T>
std::string tensorElementToString( T element );

// printing objects
template<typename DataType>
std::ostream& appendElement(std::ostream& os, const DataType& element);

size_t sizeOf( TensileDataType type );

size_t flopsPerMadd( TensileDataType type );

} // namespace


/*******************************************************************************
 * comparators for STL
 ******************************************************************************/
bool operator<(const TensileDimension & l, const TensileDimension & r);
bool operator<(const TensileControl & l, const TensileControl & r);

bool operator==(const TensileDimension & l, const TensileDimension & r);
bool operator==(const TensileComplexFloat & l, const TensileComplexFloat & r);
bool operator==(const TensileComplexDouble & l, const TensileComplexDouble & r);

#endif

