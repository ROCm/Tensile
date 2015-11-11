
#ifndef STRUCT_OPERATIONS_H
#define STRUCT_OPERATIONS_H

#include "Cobalt.h"
#include "Solution.h"
#include <string>

 std::string indent(size_t level);

/*******************************************************************************
 * enum toString
 ******************************************************************************/
std::string toString( CobaltCode code );
std::string toString( CobaltPrecision precision );
std::string toString( CobaltOperationType type );
std::string toString( CobaltOperationIndexAssignmentType type );
  

/*******************************************************************************
 * struct toString
 ******************************************************************************/
std::string toString( const CobaltProblem problem, size_t indentLevel );
std::string toString( const CobaltSolution *solution, size_t indentLevel );
std::string toString( const CobaltOperation operation, size_t indentLevel );
std::string toString( const CobaltDeviceProfile deviceProfile, size_t indentLevel );
std::string toString( const CobaltDevice device, size_t indentLevel );
std::string toString( const CobaltTensor tensor, size_t indentLevel );
std::string toString( const CobaltStatus status, size_t indentLevel );
  
/*******************************************************************************
 * xml tags for toString
 ******************************************************************************/
extern const std::string tensorTag;
extern const std::string dimensionTag;
extern const std::string dimPairTag;
extern const std::string operationTag;
extern const std::string deviceTag;
extern const std::string deviceProfileTag;
extern const std::string problemTag;
extern const std::string solutionTag;
extern const std::string statusTag;
extern const std::string traceEntryTag;
extern const std::string traceTag;
extern const std::string getSummaryTag;
extern const std::string enqueueSummaryTag;
extern const std::string documentTag;
extern const std::string numDimAttr;
extern const std::string operationAttr;
extern const std::string dimNumberAttr;
extern const std::string dimStrideAttr;
extern const std::string nameAttr;
extern const std::string typeEnumAttr;
extern const std::string typeStringAttr;


bool operator<(const CobaltStatus & a, const CobaltStatus & b );
bool operator<(const CobaltDimension & l, const CobaltDimension & r);
bool operator<(const CobaltTensor & l, const CobaltTensor & r);
bool operator<(const CobaltDevice & l, const CobaltDevice & r);
bool operator<(const CobaltDeviceProfile & l, const CobaltDeviceProfile & r);
bool operator<(const CobaltOperationIndexAssignment & l, const CobaltOperationIndexAssignment & r);
bool operator<(const CobaltOperation & l, const CobaltOperation & r);
bool operator<(const CobaltProblem & l, const CobaltProblem & r);
bool operator<(const CobaltControl & l, const CobaltControl & r);
bool operator<(const CobaltSolution & l, const CobaltSolution & r);
struct CobaltSolutionPtrComparator : std::binary_function<const CobaltSolution *, const CobaltSolution *, bool> {
  bool  operator() (const CobaltSolution *l, const CobaltSolution *r) const {
    return *l < *r;
  }
};

#endif
