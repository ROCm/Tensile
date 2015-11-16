
#ifndef STRUCT_OPERATIONS_H
#define STRUCT_OPERATIONS_H

#include "Cobalt.h"
#include "Solution.h"
#include <string>


/*******************************************************************************
 * enum toString
 ******************************************************************************/
std::string toString( CobaltCode code );
std::string toString( CobaltPrecision precision );
std::string toString( CobaltOperationType type );
std::string toString( CobaltOperationIndexAssignmentType type );
std::string toString( CobaltProblem problem );

/*******************************************************************************
 * struct toString
 ******************************************************************************/
std::string toStringXML( const CobaltProblem problem, size_t indentLevel );
std::string toStringXML( const CobaltSolution *solution, size_t indentLevel );
std::string toStringXML( const CobaltOperation operation, size_t indentLevel );
std::string toStringXML( const CobaltDeviceProfile deviceProfile,
    size_t indentLevel );
std::string toStringXML( const CobaltDevice device, size_t indentLevel );
std::string toStringXML( const CobaltTensor tensor, size_t indentLevel );
std::string toStringXML( const CobaltStatus status, size_t indentLevel );
  
/*******************************************************************************
 * xml tags for toString
 ******************************************************************************/
std::string indent(size_t level);

/*******************************************************************************
 * comparators for STL
 ******************************************************************************/
bool operator<(const CobaltStatus & a, const CobaltStatus & b );
bool operator<(const CobaltDimension & l, const CobaltDimension & r);
bool operator<(const CobaltTensor & l, const CobaltTensor & r);
bool operator<(const CobaltDevice & l, const CobaltDevice & r);
bool operator<(const CobaltDeviceProfile & l, const CobaltDeviceProfile & r);
bool operator<(const CobaltOperationIndexAssignment & l,
    const CobaltOperationIndexAssignment & r);
bool operator<(const CobaltOperation & l, const CobaltOperation & r);
bool operator<(const CobaltProblem & l, const CobaltProblem & r);
bool operator<(const CobaltControl & l, const CobaltControl & r);
bool operator<(const CobaltSolution & l, const CobaltSolution & r);
struct CobaltSolutionPtrComparator
    : std::binary_function<const CobaltSolution *,
    const CobaltSolution *, bool> {
  bool  operator() (const CobaltSolution *l, const CobaltSolution *r) const {
    return *l < *r;
  }
};

#endif
