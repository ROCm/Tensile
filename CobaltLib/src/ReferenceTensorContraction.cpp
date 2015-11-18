
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

  // indices
  size_t numFreeIndices = problem.tensorC.numDimensions;
  size_t numBoundIndices = problem.tensorA.numDimensions - numFreeIndices;
  size_t *freeIndicesC = new size_t[numFreeIndices];
  size_t *freeIndicesA = new size_t[numFreeIndices];
  size_t *freeIndicesB = new size_t[numFreeIndices];
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

  size_t *freeCoord = new size_t[numFreeIndices];
  size_t *freeIndexSizes = new size_t[numFreeIndices];
  for (size_t i = 0; i < numFreeIndices; i++) {
    freeCoord[i] = 0;
    freeIndexSizes[i] = problem.tensorC.dimensions[freeIndicesC[i]].size;
  }
  size_t *boundCoord = new size_t[numBoundIndices];
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
    freeCoord[numFreeIndices-1]++;
    for (size_t f = numFreeIndices-1; f > 0; f--) {
      if (freeCoord[f] >= freeIndexSizes[f]) {
        freeCoord[f] = 0;
        freeCoord[f-1]++;
      }
    }
    if (freeCoord[0] >= freeIndexSizes[0]) {
      break; // done with last free element of C
    }

    // iterate over entire bound index 
    float sumC = 0.f;
    while (true) {
      boundCoord[numBoundIndices-1]++;
      for ( size_t b = numBoundIndices-1; b > 0; b--) {
        if ( boundCoord[b] >= boundIndexSizes[b]) {
          boundCoord[b] = 0;
          boundCoord[b-1]++;
        }
      }
      if (boundCoord[0] >= boundIndexSizes[0]) {
        break; // done with last element
      }
      
      size_t coordsA[] = {i, y, k, x, z, j}; // TODO - lookup order
      size_t serialIdxA = coordsToSerial( problem.tensorA, coordsA);
      float valueA = dataA[serialIdxA];

      size_t coordsB[] = {y, i, j, x, k, z}; // TODO - lookup order
      size_t serialIdxB = coordsToSerial( problem.tensorB, coordsB);
      float valueB = dataB[serialIdxB];

      sumC += valueA * valueB;

    } // bound range

  } // free range



  // for free indices
  for (size_t i = 0; i < problem.tensorC.dimensions[0].size; i++) {
    for (size_t j = 0; j < problem.tensorC.dimensions[1].size; j++) {
      for (size_t k = 0; k < problem.tensorC.dimensions[2].size; k++) {
        size_t coordsC[] = {i, j, k}; // TODO - lookup order
        float sumC = 0;

        // for bound indices
        for (size_t z = 0; z < boundIndexSizes[0]; z++) {
          for (size_t y = 0; y < boundIndexSizes[1]; y++) {
            for (size_t x = 0; x < boundIndexSizes[2]; x++) {
              size_t coordsA[] = {i, y, k, x, z, j}; // TODO - lookup order
              size_t serialIdxA = coordsToSerial( problem.tensorA, coordsA);
              float valueA = dataA[serialIdxA];

              size_t coordsB[] = {y, i, j, x, k, z}; // TODO - lookup order
              size_t serialIdxB = coordsToSerial( problem.tensorB, coordsB);
              float valueB = dataB[serialIdxB];

              sumC += valueA * valueB;
            }
          }
        }
        size_t serialIdxC = coordsToSerial(problem.tensorC, coordsC);
        dataC[serialIdxC] = sumC; // TODO - or += allow split among k
      }
    }
  }


  //A[0,1,2];
  //B[0,1,2];
  //C[0,1];
  //operation.contractedIndices(1,0);
  //operation.cIndices[0](A,0);
  //operation.cIndices[1](B,1);

  //for ( size_t i = 0; i < 
  RETURN_STATUS
} // referenceTensorContraction


/*******************************************************************************
 * toString
 ******************************************************************************/
std::string ReferenceTensorContraction::toString( size_t indentLevel ) const {
  return "TODO";
}