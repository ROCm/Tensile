
#include "ReferenceTensorContraction.h"
#include "StructOperations.h"
#include <assert.h>
#include <algorithm>


/*******************************************************************************
 * constructor
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
CobaltSolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::CobaltSolutionTensorContractionCPU( CobaltProblem inputProblem )
  : CobaltSolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>(inputProblem) {
}


/*******************************************************************************
 * enqueue
 ******************************************************************************/
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


#if 0
/*******************************************************************************
 * gemm_batched
 ******************************************************************************/
CobaltStatus CobaltSolutionTensorContractionCPU::gemm_batched(
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alphaScalar,
    CobaltScalarData betaScalar,
    CobaltControl & ctrl ) {

  // find batch index
  unsigned int batchIdxC;
  unsigned int batchIdxA;
  unsigned int batchIdxB;
  for (unsigned int i = 0; i < problem.tensorC.numDimensions; i++) {
    unsigned int idxA = *std::find(problem.operation.indexAssignmentsA, problem.operation.indexAssignmentsA+3, i);
    unsigned int idxB = *std::find(problem.operation.indexAssignmentsB, problem.operation.indexAssignmentsB+3, i);
    if (idxA < 3 && idxB < 3) {
      batchIdxC = i;
      batchIdxA = idxA;
      batchIdxB = idxB;
      break;
    }
  }

  CobaltStatus status = cobaltStatusSuccess;
  for ( unsigned int batch = 0; batch < problem.tensorC.dimensions[batchIdxC].size; batch++) {
    tensorDataC.data = (void *)((float *)tensorDataC.data + problem.tensorC.dimensions[batchIdxC].stride);
    tensorDataA.data = (void *)((float *)tensorDataA.data + problem.tensorA.dimensions[batchIdxA].stride);
    tensorDataB.data = (void *)((float *)tensorDataB.data + problem.tensorB.dimensions[batchIdxB].stride);
    status = gemm( tensorDataC,
        tensorDataA,
        tensorDataB,
        alphaScalar,
        betaScalar,
        ctrl );
  }
  return status;
}

/*******************************************************************************
 * gemm
 ******************************************************************************/
CobaltStatus CobaltSolutionTensorContractionCPU::gemm(
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alphaScalar,
    CobaltScalarData betaScalar,
    CobaltControl & ctrl ) {

  float  alpha = *((float *)alphaScalar.data);
  float  beta  = *((float *)betaScalar.data);
  float *dataC =   (float *)tensorDataC.data;
  
  unsigned int tensorAIdxSummation = (unsigned int) (std::find(problem.operation.indexAssignmentsA, problem.operation.indexAssignmentsA+problem.tensorA.numDimensions, 2) - problem.operation.indexAssignmentsA);
  unsigned int tensorBIdxSummation = (unsigned int) (std::find(problem.operation.indexAssignmentsB, problem.operation.indexAssignmentsB+problem.tensorB.numDimensions, 2) - problem.operation.indexAssignmentsB);
  unsigned int tensorAIdxFree;
  unsigned int tensorAStrideFree;
  unsigned int tensorBIdxFree;
  unsigned int tensorBStrideFree;
  for (unsigned int i = 0; i < problem.tensorA.numDimensions; i++) {
    if (problem.operation.indexAssignmentsA[i] == 0) {
      tensorAIdxFree = i;
      tensorAStrideFree = problem.tensorA.dimensions[0].stride;
      break;
    }
    if (problem.operation.indexAssignmentsA[i] == 1) {
      tensorAIdxFree = i;
      tensorAStrideFree = problem.tensorA.dimensions[1].stride;
      break;
    }
  }
  for (unsigned int i = 0; i < problem.tensorB.numDimensions; i++) {
    if (problem.operation.indexAssignmentsB[i] == 0) {
      tensorBIdxFree = i;
      tensorBStrideFree = problem.tensorB.dimensions[0].stride;
      break;
    }
    if (problem.operation.indexAssignmentsB[i] == 1) {
      tensorBIdxFree = i;
      tensorBStrideFree = problem.tensorB.dimensions[1].stride;
      break;
    }
  }
  unsigned int sumMax = problem.tensorA.dimensions[tensorAIdxSummation].size;
  unsigned int d[2];
  for (d[0] = 0; d[0] < problem.tensorC.dimensions[0].size; d[0]++) {
    for (d[1] = 0; d[1] < problem.tensorC.dimensions[1].size; d[1]++) {
      float sum = 0.f;
      for (unsigned int sumIdx = 0; sumIdx < sumMax; sumIdx++) {
        unsigned int idxA = (d[tensorAIdxFree])*tensorAStrideFree
          + sumIdx*problem.tensorA.dimensions[tensorAIdxSummation].stride;
        unsigned int idxB = d[tensorBIdxFree]*tensorBStrideFree
          + sumIdx*problem.tensorB.dimensions[tensorBIdxSummation].stride;
        float elementA = ((float *) tensorDataA.data)[idxA];
        float elementB = ((float *) tensorDataB.data)[idxB];
        sum += elementA * elementB;
      }
      unsigned int idxC = d[0]*problem.tensorC.dimensions[0].stride
        + d[1]*problem.tensorC.dimensions[1].stride;
      dataC[idxC] = alpha*sum + beta*dataC[idxC];
    }
  }
  return cobaltStatusSuccess;
}
#endif

/*******************************************************************************
 * toString
 ******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
std::string CobaltSolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toString( size_t indentLevel ) const {
  return "CobaltSolutionTensorContractionCPU";
}


size_t coordsToSerial( CobaltTensor tensor, size_t *coords ) {
  size_t serial = 0;
  for (size_t i = 0; i < tensor.numDimensions; i++) {
    serial += coords[i] * tensor.dimensions[i].stride;
  }
  return serial;
}

/*******************************************************************************
 * cobaltGetSolution
 * need to list all wanted template variants for compiler in this file
 ******************************************************************************/
CobaltStatus cobaltGetSolutionCPU(
    CobaltProblem problem,
    CobaltSolution **solution ) {
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



/*******************************************************************************
 * Explicit Template Instantiation - redundant of cobaltGetSolutionCPU
 ******************************************************************************/
//template class CobaltSolutionTensorContractionCPU<float,float,float,float,float>;
//template class CobaltSolutionTensorContractionCPU<double,double,double,double,double>;
//template class CobaltSolutionTensorContractionCPU<CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat>;
//template class CobaltSolutionTensorContractionCPU<CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble>;
