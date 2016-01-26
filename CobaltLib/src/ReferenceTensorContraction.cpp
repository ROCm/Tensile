
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

  // pointers to data
  // TODO need OpenCL read buffer here
  float *dataA = 0; //  = (float *)tensorDataA.clMem;
  dataA += tensorDataA.offset;
  float *dataB = 0; //  = (float *)tensorDataB.clMem;
  dataB += tensorDataB.offset;
  float *dataC = 0; // = (float *)tensorDataC.clMem;
  dataC += tensorDataC.offset;
  
  size_t numIndicesFreeC = problem.tensorC.numDimensions;
  size_t numIndicesSummation = problem.operation.numIndicesSummation;
  size_t numIndicesFreeAB = problem.tensorA.numDimensions - numIndicesSummation;

  size_t *freeCoord = new size_t[numIndicesFreeC];
  //size_t *freeIndexSizes = new size_t[problem.operation.numIndicesFree];
  for (size_t i = 0; i < numIndicesFreeC; i++) {
    freeCoord[i] = 0;
  //  freeIndexSizes[i] = problem.tensorC.dimensions[freeIndicesC[i]].size;
  }
  size_t *boundCoord = new size_t[numIndicesSummation];
  for (size_t b = 0; b < numIndicesSummation; b++) boundCoord[b] = 0;
  size_t *boundIndexSizes = new size_t[numIndicesSummation];
  for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
    if ( problem.operation.indexAssignmentsA[i] >= numIndicesFreeC) {
      boundIndexSizes[problem.operation.indexAssignmentsA[i]-numIndicesFreeC] = problem.tensorA.dimensions[i].size;
      // TODO - verify
    }
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
      boundCoord[problem.operation.numIndicesSummation-1]++;
      for ( size_t b = problem.operation.numIndicesSummation-1; b > 0; b--) {
        if ( boundCoord[b] >= boundIndexSizes[b]) {
          boundCoord[b] = 0;
          boundCoord[b-1]++;
        }
      }
      if (boundCoord[0] >= boundIndexSizes[0]) {
        break; // done with last element
      }
      // convert free/bound coord into tensorA,B 
      for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
        if (problem.operation.indexAssignmentsA[i] < numIndicesFreeC) {
          coordsA[i] = freeCoord[problem.operation.indexAssignmentsA[i]];
        } else {
          coordsA[i] = boundCoord[problem.operation.indexAssignmentsA[i]-numIndicesFreeC];
        }
      }
      for (size_t i = 0; i < problem.tensorB.numDimensions; i++) {
        if (problem.operation.indexAssignmentsB[i] < numIndicesFreeC) {
          coordsB[i] = freeCoord[problem.operation.indexAssignmentsB[i]];
        } else {
          coordsB[i] = boundCoord[problem.operation.indexAssignmentsB[i]-numIndicesFreeC];
        }
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

  return cobaltStatusSuccess;
} // referenceTensorContraction


/*******************************************************************************
 * toString
 ******************************************************************************/
std::string ReferenceTensorContraction::toString( size_t indentLevel ) const {
  return "TODO";
}