
#pragma once

#include <vector>

namespace Cobalt {
  
  
/*******************************************************************************
 * Precision
 ******************************************************************************/
typedef enum Precision { S, D, C, Z } Precision;
std::string precisionToString( Precision precision);
  
/*******************************************************************************
 * Tensor Data - OpenCL 1.2
 ******************************************************************************/
#if Cobalt_BACKEND_OPENCL12

typedef struct TensorData {
  cl_mem clMem;
  size_t offset;
} TensorData;

/*******************************************************************************
 * Tensor Data - OpenCL 2.0
 ******************************************************************************/
#elif Cobalt_BACKEND_OPENCL20
typedef enum {
  openCLBufferType_clMem,
  openClBufferType_SVM
} OpenCLBufferType;

typedef struct TensorData {
  void *data;
  OpenCLBufferType bufferType;
  size_t offset;
} TensorData;

/*******************************************************************************
 * Tensor Data - HCC
 ******************************************************************************/
#elif Cobalt_BACKEND_HCC
typedef void* TensorData;

/*******************************************************************************
 * Tensor Data - HSA
 ******************************************************************************/
#elif Cobalt_BACKEND_HSA  
typedef void* TensorData;

#endif


/*******************************************************************************
 * Dimension Descriptor
 * - every dimension of a tensor has a size and a stide
 ******************************************************************************/
  typedef struct DimensionDescriptor {
  size_t size;
  size_t stride;
  
/*******************************************************************************
 * comparison method for STL
 ******************************************************************************/
  bool operator< ( const DimensionDescriptor & other ) const;

/*******************************************************************************
 * toString for writting to xml
 ******************************************************************************/
  std::string toString( size_t indentLevel ) const;
  
} DimensionDescriptor;
  
/*******************************************************************************
 * Tensor Descriptor
 ******************************************************************************/
class TensorDescriptor {
public:
  
/*******************************************************************************
 * precision
 ******************************************************************************/
  Precision precision;

/*******************************************************************************
 * list of dimensions
 ******************************************************************************/
  std::vector<DimensionDescriptor> dimensions;
  
/*******************************************************************************
 * constructor - numDims
 * - user manually sets dimensions
 ******************************************************************************/
  TensorDescriptor( size_t inputNumDims );

/*******************************************************************************
 * constructor - list of sizes
 * - dimensions are auto-populated from compactSizes
 ******************************************************************************/
  TensorDescriptor( std::vector<size_t> compactSizes );

/*******************************************************************************
 * comparison operator for STL
 ******************************************************************************/
  bool operator< ( const TensorDescriptor & other ) const;
  
/*******************************************************************************
 * toString for writing xml
 ******************************************************************************/
  std::string toString( size_t indentLevel ) const;
  
/*******************************************************************************
 * convert a multidimensional coordinate to a serial index (for this tensor)
 ******************************************************************************/
  size_t coordsToSerial( std::vector<size_t> coords ) const;

/*******************************************************************************
 * convert a serial index to a multidimensional coordinate (for this tensor)
 ******************************************************************************/
  std::vector<size_t> serialToCoords( size_t serial ) const;

/*******************************************************************************
 * convert a list of sizes (compact) to a list of dimension descriptors
 ******************************************************************************/
  static std::vector<DimensionDescriptor> compactSizesToDimensions( std::vector<size_t> compactSizes );
  
};

} // namespace Cobalt