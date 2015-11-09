
#pragma once

#include <vector>

namespace Cobalt {
  
/*******************************************************************************
 * DimensionPair
 * - pair of dimensions contracted over
 * - dimension a of tensor A will be multiplied by dimension b of tensor B
 ******************************************************************************/
typedef struct DimensionPair {
  size_t a, b;
  
/*******************************************************************************
 * comparison operator for STL
 ******************************************************************************/
  bool operator< ( const DimensionPair & other ) const;
  
/*******************************************************************************
 * toString
 * - for writing xml format
 ******************************************************************************/
  std::string DimensionPair::toString( size_t indentLevel ) const;

} DimensionPair;

/*******************************************************************************
 * OperationType
 * - type of operation used in C = A op B, such as convolution
 ******************************************************************************/
enum OperationType {
  TensorContraction,
  Convolution
};

/*******************************************************************************
 * OperationDescriptor
 * - exact operation used in C = A op B
 ******************************************************************************/
class OperationDescriptor {
public:
  
/*******************************************************************************
 * state
 ******************************************************************************/
  OperationType type;
  std::vector<DimensionPair> dimensions;
  // stride
  // handling edge
  // interpolation stuff?
  
/*******************************************************************************
 * constructor
 ******************************************************************************/
  OperationDescriptor( OperationType inputOperationType, size_t inputNumDims );
  
/*******************************************************************************
 * comparison operator for STL
 ******************************************************************************/
  bool operator< ( const OperationDescriptor & other ) const;

/*******************************************************************************
 * toString
 * - for writing xml format
 ******************************************************************************/
  std::string toString( size_t indentLevel ) const;

}; // class Operation

} // namespace Cobalt