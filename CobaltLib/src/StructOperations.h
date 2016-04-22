
#ifndef STRUCT_OPERATIONS_H
#define STRUCT_OPERATIONS_H

#include "Cobalt.h"
#include "Tools.h"
//#include "Solution.h"
#include <string>

namespace Cobalt {

/*******************************************************************************
 * enum toString
 ******************************************************************************/
std::string toString( CobaltStatus code );
std::string toString( CobaltDataType dataType );
std::string toString( CobaltOperationType type );
std::string toString( CobaltProblem problem );

template<typename T>
std::string tensorElementToString( T element );

// printing objects
template<typename DataType>
std::ostream& appendElement(std::ostream& os, const DataType& element);

size_t sizeOf( CobaltDataType type );

} // namespace


/*******************************************************************************
 * comparators for STL
 ******************************************************************************/
bool operator<(const CobaltDimension & l, const CobaltDimension & r);
bool operator<(const CobaltControl & l, const CobaltControl & r);

bool operator==(const CobaltDimension & l, const CobaltDimension & r);
bool operator==(const CobaltComplexFloat & l, const CobaltComplexFloat & r);
bool operator==(const CobaltComplexDouble & l, const CobaltComplexDouble & r);

#endif
