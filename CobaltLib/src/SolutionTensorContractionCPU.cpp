#include "Problem.h"
#include "SolutionTensorContractionCPU.h"
#include "StructOperations.h"
#include "MathTemplates.h"

#include <assert.h>
#include <algorithm>

namespace Cobalt {

/*******************************************************************************
 * constructor
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
SolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::SolutionTensorContractionCPU( const Problem & inputProblem )
  : SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>(inputProblem) {
}


/*******************************************************************************
 * enqueue
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
CobaltStatus SolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::enqueue(
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl & ctrl ) {
  //bool zzz0zC = std::is_same<TypeC, CobaltComplexDouble>::value;
  //bool zzz0zA = std::is_same<TypeA, CobaltComplexDouble>::value;
  //bool zzz0zB = std::is_same<TypeB, CobaltComplexDouble>::value;
  //bool zzz0zAlpha = problem.getDataTypeAlpha() == cobaltDataTypeNone;
  //bool zzz0zBeta = std::is_same<TypeBeta, CobaltComplexDouble>::value;
  //printf("zzz0z=%s,%s,%s,%s,%s\n",
  //  zzz0zC ? "T" : "F",
  //  zzz0zA ? "T" : "F",
  //  zzz0zB ? "T" : "F",
  //  zzz0zAlpha ? "T" : "F",
  //  zzz0zBeta ? "T" : "F"
  //  );
  //bool zzz0z = zzz0zC && zzz0zA && zzz0zB && zzz0zAlpha && zzz0zBeta;

  // pointers to data
  TypeC *dataC = (TypeC *)tensorDataC.data;
  dataC += tensorDataC.offset;
  TypeA *dataA = (TypeA *)tensorDataA.data;
  dataA += tensorDataA.offset;
  TypeB *dataB = (TypeB *)tensorDataB.data;
  dataB += tensorDataB.offset;

  
  // index sizes
  unsigned int numIndicesFreeC = problem.tensorC.numDims();
  unsigned int numIndicesSummation = static_cast<unsigned int>(problem.indicesSummation.size());
  unsigned int numIndicesFreeAB = problem.tensorA.numDims() - numIndicesSummation;

  // allocate coords and sizes
  std::vector<unsigned int> freeCoord(numIndicesFreeC);
  std::vector<unsigned int> boundCoord( numIndicesSummation );
  std::vector<unsigned int> boundIndexSizes( numIndicesSummation );

  // initialize coords & sizes
  for (size_t i = 0; i < numIndicesFreeC; i++) {
    freeCoord[i] = 0;
  }

  for (size_t i = 0; i < problem.tensorA.numDims(); i++) {
    if ( problem.indicesA[i] >= numIndicesFreeC) {
      boundIndexSizes[problem.indicesA[i]-numIndicesFreeC] = problem.tensorA[i].size;
      //printf("boundIndexSizes[%u] = %u\n", problem.indicesA[i] - numIndicesFreeC, problem.tensorA[i].size);
      // TODO - verify
    }
  }

  // allocate tensor coords
  std::vector<unsigned int> coordsC( problem.tensorC.numDims() );
  std::vector<unsigned int> coordsA( problem.tensorA.numDims() );
  std::vector<unsigned int> coordsB( problem.tensorB.numDims() );

  while (true) { // iterate over entire free index range

    TypeC sumC = getZero<TypeC>();
    // reset summation indices
    for (unsigned int b = 0; b < numIndicesSummation; b++) {
      boundCoord[b] = 0;
    }
    while (true) { // iterate over entire bound index range
      
      // convert free/bound coord into tensorA,B 
      for (unsigned int i = 0; i < problem.tensorA.numDims(); i++) {
        if (problem.indicesA[i] < numIndicesFreeC) {
          coordsA[i] = freeCoord[problem.indicesA[i]];
        } else {
          coordsA[i] = boundCoord[problem.indicesA[i]-numIndicesFreeC];
        }
      }
      for (unsigned int i = 0; i < problem.tensorB.numDims(); i++) {
        if (problem.indicesB[i] < numIndicesFreeC) {
          coordsB[i] = freeCoord[problem.indicesB[i]];
        } else {
          coordsB[i] = boundCoord[problem.indicesB[i]-numIndicesFreeC];
        }
      }
      
      size_t serialIdxA = problem.tensorA.getIndex(coordsA);
      TypeA valueA = dataA[serialIdxA];
      if ( std::is_same<TypeA, CobaltComplexFloat>::value
        || std::is_same<TypeA, CobaltComplexDouble>::value) {
        if ( problem.tensorA.getDataType() == cobaltDataTypeComplexConjugateHalf
          || problem.tensorA.getDataType() == cobaltDataTypeComplexConjugateSingle
          || problem.tensorA.getDataType() == cobaltDataTypeComplexConjugateDouble) {
          complexConjugate<TypeA>( valueA );
        }
      }

      size_t serialIdxB = problem.tensorB.getIndex(coordsB);
      TypeB valueB = dataB[serialIdxB];
      if (std::is_same<TypeB, CobaltComplexFloat>::value
        || std::is_same<TypeB, CobaltComplexDouble>::value) {
        if ( problem.tensorB.getDataType() == cobaltDataTypeComplexConjugateHalf
          || problem.tensorB.getDataType() == cobaltDataTypeComplexConjugateSingle
          || problem.tensorB.getDataType() == cobaltDataTypeComplexConjugateDouble) {
          complexConjugate<TypeB>( valueB );
        }
      }

      TypeC product = multiply<TypeC,TypeA,TypeB>( valueA, valueB);

      sumC = add<TypeC,TypeA,TypeB>(sumC,product);

      // increment bound coord
      boundCoord[numIndicesSummation-1]++;
      for ( size_t b = 0; b < numIndicesSummation - 1; b++) {
        if ( boundCoord[b] >= boundIndexSizes[b]) {
          boundCoord[b] = 0;
          boundCoord[b+1]++;
        }
      }
      if (boundCoord[numIndicesSummation - 1] >= boundIndexSizes[numIndicesSummation - 1]) {
        break; // bound index range exit criteria
      }

    } // bound range


    size_t serialIdxC = problem.tensorC.getIndex(freeCoord);
    if (alpha.data) {
      TypeAlpha *alphaData = static_cast<TypeAlpha*>(alpha.data);
      sumC = multiply<TypeC,TypeAlpha,TypeC>(*alphaData,sumC);
    }
    if (beta.data) {
      TypeBeta *betaData = static_cast<TypeBeta*>(beta.data);
      TypeC tmp = multiply<TypeC,TypeBeta,TypeC>(*betaData, dataC[serialIdxC]);
      sumC = add<TypeC,TypeC,TypeC>(tmp,sumC);
    }

    dataC[serialIdxC] = sumC; // TODO - or += allow split among k

    // increment free coord
    freeCoord[0]++;
    for (size_t f = 0; f < problem.tensorC.numDims()-1; f++) {
      if (freeCoord[f] >= problem.tensorC[f].size) {
        freeCoord[f] = 0;
        freeCoord[f+1]++;
      }
    }
    if (freeCoord[problem.tensorC.numDims() - 1] >= problem.tensorC[problem.tensorC.numDims() - 1].size) {
      break; // free index range exit criteria
    }

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
    unsigned int idxA = *std::find(problem.indicesA, problem.indicesA+3, i);
    unsigned int idxB = *std::find(problem.indicesB, problem.indicesB+3, i);
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
  
  unsigned int tensorAIdxSummation = (unsigned int) (std::find(problem.indicesA, problem.indicesA+problem.tensorA.numDimensions, 2) - problem.indicesA);
  unsigned int tensorBIdxSummation = (unsigned int) (std::find(problem.indicesB, problem.indicesB+problem.tensorB.numDimensions, 2) - problem.indicesB);
  unsigned int tensorAIdxFree;
  unsigned int tensorAStrideFree;
  unsigned int tensorBIdxFree;
  unsigned int tensorBStrideFree;
  for (unsigned int i = 0; i < problem.tensorA.numDimensions; i++) {
    if (problem.indicesA[i] == 0) {
      tensorAIdxFree = i;
      tensorAStrideFree = problem.tensorA.dimensions[0].stride;
      break;
    }
    if (problem.indicesA[i] == 1) {
      tensorAIdxFree = i;
      tensorAStrideFree = problem.tensorA.dimensions[1].stride;
      break;
    }
  }
  for (unsigned int i = 0; i < problem.tensorB.numDimensions; i++) {
    if (problem.indicesB[i] == 0) {
      tensorBIdxFree = i;
      tensorBStrideFree = problem.tensorB.dimensions[0].stride;
      break;
    }
    if (problem.indicesB[i] == 1) {
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
std::string SolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toString( size_t indentLevel ) const {
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
std::tuple<Solution *,CobaltStatus> getSolutionCPU( const Problem & problem) {

  bool problemIsTensorContraction = true;

  if (problemIsTensorContraction) {
    switch(problem.getDataTypeC()) {
    case cobaltDataTypeSingle:
      return std::make_tuple(new Cobalt::SolutionTensorContractionCPU<float,float,float,float,float>( problem ), cobaltStatusSuccess );
    case cobaltDataTypeDouble:
      return std::make_tuple(new Cobalt::SolutionTensorContractionCPU<double,double,double,double,double>( problem ), cobaltStatusSuccess );
    case cobaltDataTypeComplexSingle:
      return std::make_tuple(new Cobalt::SolutionTensorContractionCPU<CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat>( problem ), cobaltStatusSuccess );
    case cobaltDataTypeComplexDouble:
      return std::make_tuple(new Cobalt::SolutionTensorContractionCPU<CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble>( problem ), cobaltStatusSuccess );
    default:
      return std::make_tuple(nullptr, cobaltStatusProblemNotSupported);
    }
  } else {
  // TODO - reorganize to include CPU convolution also
      return std::make_tuple(nullptr, cobaltStatusProblemNotSupported);
  }
}

} // namespace

/*******************************************************************************
 * Explicit Template Instantiation - redundant of cobaltGetSolutionCPU
 ******************************************************************************/
//template class CobaltSolutionTensorContractionCPU<float,float,float,float,float>;
//template class CobaltSolutionTensorContractionCPU<double,double,double,double,double>;
//template class CobaltSolutionTensorContractionCPU<CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat>;
//template class CobaltSolutionTensorContractionCPU<CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble>;
