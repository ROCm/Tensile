
#ifndef STRUCT_OPERATIONS_H
#define STRUCT_OPERATIONS_H

#include "Cobalt.h"
//#include "Solution.h"
#include <string>

namespace Cobalt {

/*******************************************************************************
 * enum toString
 ******************************************************************************/
std::string toString( CobaltStatus code );
std::string toString( CobaltDataType dataType );
std::string toString( CobaltOperationType type );
std::string toString( CobaltProblem problem ); // TODO move to problem.h

std::string toString( CobaltTensorData data, CobaltTensor tensor );
template<typename T>
std::string toStringTemplate( CobaltTensorData data, CobaltTensor tensor );

template<typename T>
std::string tensorElementToString( T element );

/*******************************************************************************
 * struct toString
 ******************************************************************************/
//std::string toStringXML( const CobaltProblem problem, size_t indentLevel );
//std::string toStringXML( const Cobalt::Solution *solution, size_t indentLevel ); // move to 
//std::string toStringXML( const CobaltOperation operation, size_t indentLevel );
//std::string toStringXML( const CobaltDeviceProfile deviceProfile,
//    size_t indentLevel );
//std::string toStringXML( const CobaltDevice device, size_t indentLevel );
//std::string toStringXML( const CobaltTensor tensor, size_t indentLevel );
  

/*******************************************************************************
 * comparators for STL
 ******************************************************************************/
bool operator<(const CobaltDimension & l, const CobaltDimension & r);
bool operator<(const CobaltTensor & l, const CobaltTensor & r);
bool operator<(const CobaltDevice & l, const CobaltDevice & r);
bool operator<(const CobaltDeviceProfile & l, const CobaltDeviceProfile & r);
//bool operator<(const CobaltOperation & l, const CobaltOperation & r);
//bool operator<(const CobaltProblem & l, const CobaltProblem & r);
bool operator<(const CobaltControl & l, const CobaltControl & r);
bool operator<(const Cobalt::Solution & l, const Cobalt::Solution & r);
struct CobaltSolutionPtrComparator
    : std::binary_function<const Cobalt::Solution *,
    const Cobalt::Solution *, bool> {
  bool  operator() (const Cobalt::Solution *l, const Cobalt::Solution *r) const {
    return *l < *r;
  }
};

size_t getCobaltDataTypeSize( CobaltDataType type );

} // namespace

#endif
