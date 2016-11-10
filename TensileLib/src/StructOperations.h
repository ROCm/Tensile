/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


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

