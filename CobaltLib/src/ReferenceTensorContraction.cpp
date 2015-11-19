
#include "ReferenceTensorContraction.h"
#include "StructOperations.h"
#include <assert.h>



/*******************************************************************************
 * constructor
 ******************************************************************************/
ReferenceTensorContraction::ReferenceTensorContraction(
    CobaltProblem inputProblem )
  : CobaltSolution(inputProblem) {
}

size_t coordsToSerial( CobaltTensor tensor, size_t *coords ) {
  size_t serial = 0;
  for (size_t i = 0; i < tensor.numDimensions; i++) {
    serial += coords[i] * tensor.dimensions[i].stride;
  }
  return serial;
}

/*******************************************************************************
 * enqueue
 ******************************************************************************/
CobaltStatus ReferenceTensorContraction::enqueue(
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltTensorData tensorDataC,
    CobaltControl & ctrl ) {
  INIT_STATUS

  // pointers to data
  float *dataA = (float *)tensorDataA.data;
  dataA += tensorDataA.offset;
  float *dataB = (float *)tensorDataB.data;
  dataB += tensorDataB.offset;
  float *dataC = (float *)tensorDataC.data;
  dataC += tensorDataC.offset;

#if 0
  // indices
  size_t numFreeIndices = problem.tensorC.numDimensions;
  size_t numFreeIndicesA = 0;
  for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
    if (problem.operation.operationIndexAssignmentsA[i].type == cobaltOperationIndexAssignmentTypeFree) {
      numFreeIndicesA++;
    }
  }
  size_t numFreeIndicesB = numFreeIndicesA;
  size_t numBoundIndices = problem.tensorA.numDimensions - numFreeIndices;
  size_t *freeIndicesC = new size_t[numFreeIndices];
  size_t *freeIndicesA = new size_t[numFreeIndices]; // TODO A doesn't have all free indices
  size_t *freeIndicesB = new size_t[numFreeIndices]; // TODO B doesn't have all free indices
  size_t *boundIndicesA = new size_t[numBoundIndices];
  size_t *boundIndicesB = new size_t[numBoundIndices];
  bool *boundIndexAssigned = new bool[problem.tensorA.numDimensions];
  for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
    boundIndexAssigned[i] = false;
  }

  // order bound indices
  for ( size_t b = 0; b < numBoundIndices; b++) {
    // find largest-stride bound index (using A as identifier)
    size_t winStride = 0;
    size_t winIndex;
    for ( size_t i = 0; i < problem.tensorA.numDimensions; i++) {
      if (problem.operation.operationIndexAssignmentsA[i].type
        == cobaltOperationIndexAssignmentTypeBound && !boundIndexAssigned[i]) {
          size_t stride = problem.tensorA.dimensions[i].stride
            + problem.tensorB.dimensions[problem.operation.operationIndexAssignmentsA[i].index].stride;
        // inner-most summation loop will have the smallest combined stride for fast read
        if (stride > winStride) {
          winStride = stride;
          winIndex = i;
        }
      }
    }
    boundIndexAssigned[winIndex] = true;
    boundIndicesA[b] = winIndex;
    boundIndicesB[b] = problem.operation.operationIndexAssignmentsA[winIndex].index;
  }

  // order free indices
  bool *freeIndexAssigned = new bool[numFreeIndices];
  for (size_t i = 0; i < numFreeIndices; i++) freeIndexAssigned[i] = false;

  for (size_t f = 0; f < numFreeIndices; f++) {
    // find largest-stride free index
    size_t winStride = 0;
    size_t winIndex;
    for ( size_t i = 0; i < numFreeIndices; i++) {
      if (!freeIndexAssigned[i]) {
        size_t stride = problem.tensorC.dimensions[i].stride;
        if (stride > winStride) {
          winStride = stride;
          winIndex = i;
        }
      }
    }
    freeIndexAssigned[winIndex] = true;
    freeIndicesC[f] = winIndex;
    for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
      if (problem.operation.operationIndexAssignmentsA[i].type == cobaltOperationIndexAssignmentTypeFree
        && problem.operation.operationIndexAssignmentsA[i].index == winIndex) {
        freeIndicesA[f] = i;
      }
    }
    for (size_t i = 0; i < problem.tensorB.numDimensions; i++) {
      if (problem.operation.operationIndexAssignmentsB[i].type == cobaltOperationIndexAssignmentTypeFree
        && problem.operation.operationIndexAssignmentsB[i].index == winIndex) {
        freeIndicesB[f] = i;
      }
    }
  }
#endif

  size_t *freeCoord = new size_t[problem.operation.numFreeIndices];
  //size_t *freeIndexSizes = new size_t[problem.operation.numFreeIndices];
  for (size_t i = 0; i < problem.operation.numFreeIndices; i++) {
    freeCoord[i] = 0;
  //  freeIndexSizes[i] = problem.tensorC.dimensions[freeIndicesC[i]].size;
  }
  size_t *boundCoord = new size_t[problem.operation.numBoundIndices];
  size_t *boundIndexSizes = new size_t[numBoundIndices];
  for (size_t i = 0; i < numBoundIndices; i++) {
    boundCoord[i] = 0;
    boundIndexSizes[i] = problem.tensorA.dimensions[boundIndicesA[i]].size;
  }
  size_t *coordsA = new size_t[problem.tensorA.numDimensions];
  size_t *coordsB = new size_t[problem.tensorB.numDimensions];
  size_t *coordsC = new size_t[problem.tensorC.numDimensions];

  // iterate over entire free index range
  while (true) {

    // next free element is along last free dimension
    freeCoord[problem.tensorC.numDimensions-1]++;
    for (size_t f = problem.tensorC.numDimensions-1; f > 0; f--) {
      if (freeCoord[f] >= problem.tensorC.dimensions[f].size) {
        freeCoord[f] = 0;
        freeCoord[f-1]++;
      }
    }
    if (freeCoord[0] >= problem.tensorC.dimensions[0].size) {
      break; // done with last free element of C
    }

    // iterate over entire bound index 
    float sumC = 0.f;
    while (true) {
      boundCoord[problem.operation.numBoundIndices-1]++;
      for ( size_t b = problem.operation.numBoundIndices-1; b > 0; b--) {
        if ( boundCoord[b] >= boundIndexSizes[b]) {
          boundCoord[b] = 0;
          boundCoord[b-1]++;
        }
      }
      if (boundCoord[0] >= boundIndexSizes[0]) {
        break; // done with last element
      }
      for (size_t f = 0; f < problem.operation.numFreeIndices; f++) {
        coordsA[problem.operation.freeIndicesA[f]] = freeCoord[f];
        coordsB[problem.operation.freeIndicesB[f]] = freeCoord[f];
      }
      for (size_t b = 0; b < problem.operation.numBoundIndices; b++) {
        coordsA[problem.operation.boundIndicesA[b]] = boundCoord[b];
        coordsB[problem.operation.boundIndicesB[b]] = boundCoord[b];
      }
      
      size_t serialIdxA = coordsToSerial( problem.tensorA, coordsA);
      float valueA = dataA[serialIdxA];

      size_t serialIdxB = coordsToSerial( problem.tensorB, coordsB);
      float valueB = dataB[serialIdxB];

      sumC += valueA * valueB;

    } // bound range
    size_t serialIdxC = coordsToSerial(problem.tensorC, freeCoord);
    dataC[serialIdxC] = sumC; // TODO - or += allow split among k

  } // free range

  RETURN_STATUS
} // referenceTensorContraction


/*******************************************************************************
 * toString
 ******************************************************************************/
std::string ReferenceTensorContraction::toString( size_t indentLevel ) const {
  return "TODO";
}