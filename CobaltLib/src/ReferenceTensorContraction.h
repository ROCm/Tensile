#ifndef REFERENCE_TENSOR_CONTRACTION_H
#define REFERENCE_TENSOR_CONTRACTION_H
#include "Cobalt.h"
#include "Solution.h"
#include <assert.h>


/*******************************************************************************
 * CobaltSolutionTensorContractionCPU
 * - compute tensor contraction on cpu using simple/slow loops
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
class CobaltSolutionTensorContractionCPU : public CobaltSolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  CobaltSolutionTensorContractionCPU( CobaltProblem inputProblem );


  CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltScalarData alpha,
      CobaltScalarData beta,
      CobaltControl & ctrl );
  
  std::string toString( size_t indentLevel ) const;
 


  //CobaltStatus gemm_batched(
  //  CobaltTensorData tensorDataC,
  //  CobaltTensorData tensorDataA,
  //  CobaltTensorData tensorDataB,
  //  CobaltScalarData alpha,
  //  CobaltScalarData beta,
  //  CobaltControl & ctrl );
  //
  //CobaltStatus gemm(
  //  CobaltTensorData tensorDataC,
  //  CobaltTensorData tensorDataA,
  //  CobaltTensorData tensorDataB,
  //  CobaltScalarData alpha,
  //  CobaltScalarData beta,
  //  CobaltControl & ctrl );
};


#if 0
  template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
  CobaltSolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::CobaltSolutionTensorContractionCPU( CobaltProblem inputProblem )
      : CobaltSolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>(inputProblem) {
}

  template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
CobaltStatus CobaltSolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::enqueue(
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl & ctrl ) {

  // GEMM
  //if (problem.operation.numIndicesFree == 2
  //    && problem.operation.numIndicesBatch == 0
  //    && problem.operation.numIndicesSummation == 1) {
  //  return gemm(
  //      tensorDataC,
  //      tensorDataA,
  //      tensorDataB,
  //      alpha,
  //      beta,
  //      ctrl );
  //} else if (problem.operation.numIndicesFree == 2
  //    && problem.operation.numIndicesBatch == 1
  //    && problem.operation.numIndicesSummation == 1) {
  //  return gemm_batched(
  //      tensorDataC,
  //      tensorDataA,
  //      tensorDataB,
  //      alpha,
  //      beta,
  //      ctrl );
  //}

  // pointers to data
  float *dataA = (float *)tensorDataA.data;
  dataA += tensorDataA.offset;
  float *dataB = (float *)tensorDataB.data;
  dataB += tensorDataB.offset;
  float *dataC = (float *)tensorDataC.data;
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


template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
std::string CobaltSolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toString( size_t indentLevel ) const {
  return "CobaltSolutionTensorContractionCPU";
} 
#endif







/*******************************************************************************
 * cobaltGetSolution
 * need to list all wanted template variants for compiler in this file
 ******************************************************************************/
CobaltStatus cobaltGetSolutionCPU(
    CobaltProblem problem,
    CobaltSolution **solution );
#if 0
{
  bool problemIsTensorContraction = true;

  if (problemIsTensorContraction) {
    switch(problem.tensorC.dataType) {
    case cobaltDataTypeSingle:
      (*solution)->pimpl = new CobaltSolutionTensorContractionCPU<float,float,float,float,float>( problem );
      break;
    case cobaltDataTypeDouble:
      (*solution)->pimpl = new CobaltSolutionTensorContractionCPU<double,double,double,double,double>( problem );
      break;
    case cobaltDataTypeComplexSingle:
      (*solution)->pimpl = new CobaltSolutionTensorContractionCPU<CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat>( problem );
      break;
    case cobaltDataTypeComplexDouble:
      (*solution)->pimpl = new CobaltSolutionTensorContractionCPU<CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble>( problem );
      break;
    default:
      (*solution)->pimpl = nullptr;
      return cobaltStatusProblemNotSupported;
    }

    return cobaltStatusSuccess;
  } else {
  // TODO - reorganize to include CPU convolution also
    return cobaltStatusProblemNotSupported;
  }
}
#endif


#endif