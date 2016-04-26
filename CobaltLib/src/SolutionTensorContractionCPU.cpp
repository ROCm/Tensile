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

  // pointers to data
  TypeC *dataC = (TypeC *)tensorDataC.data;
  dataC += tensorDataC.offset;
  TypeA *dataA = (TypeA *)tensorDataA.data;
  dataA += tensorDataA.offset;
  TypeB *dataB = (TypeB *)tensorDataB.data;
  dataB += tensorDataB.offset;

  
  // index sizes
  unsigned int numIndicesFreeC = Solution::problem.tensorC.numDims();
  unsigned int numIndicesSummation = static_cast<unsigned int>(Solution::problem.indicesSummation.size());
  unsigned int numIndicesFreeAB = Solution::problem.tensorA.numDims() - numIndicesSummation;

  // allocate coords and sizes
  std::vector<unsigned int> freeCoord(numIndicesFreeC);
  std::vector<unsigned int> boundCoord( numIndicesSummation );
  std::vector<unsigned int> boundIndexSizes( numIndicesSummation );

  // initialize coords & sizes
  for (size_t i = 0; i < numIndicesFreeC; i++) {
    freeCoord[i] = 0;
  }

  for (size_t i = 0; i < Solution::problem.tensorA.numDims(); i++) {
    if ( Solution::problem.indicesA[i] >= numIndicesFreeC) {
      boundIndexSizes[Solution::problem.indicesA[i]-numIndicesFreeC] = Solution::problem.tensorA[i].size;
    }
  }

  // allocate tensor coords
  std::vector<unsigned int> coordsC( Solution::problem.tensorC.numDims() );
  std::vector<unsigned int> coordsA( Solution::problem.tensorA.numDims() );
  std::vector<unsigned int> coordsB( Solution::problem.tensorB.numDims() );

  while (true) { // iterate over entire free index range

    TypeC sumC = getZero<TypeC>();
    // reset summation indices
    for (unsigned int b = 0; b < numIndicesSummation; b++) {
      boundCoord[b] = 0;
    }
    while (true) { // iterate over entire bound index range
      
      // convert free/bound coord into tensorA,B 
      for (unsigned int i = 0; i < Solution::problem.tensorA.numDims(); i++) {
        if (Solution::problem.indicesA[i] < numIndicesFreeC) {
          coordsA[i] = freeCoord[Solution::problem.indicesA[i]];
        } else {
          coordsA[i] = boundCoord[Solution::problem.indicesA[i]-numIndicesFreeC];
        }
      }
      for (unsigned int i = 0; i < Solution::problem.tensorB.numDims(); i++) {
        if (Solution::problem.indicesB[i] < numIndicesFreeC) {
          coordsB[i] = freeCoord[Solution::problem.indicesB[i]];
        } else {
          coordsB[i] = boundCoord[Solution::problem.indicesB[i]-numIndicesFreeC];
        }
      }
      
      size_t serialIdxA = Solution::problem.tensorA.getIndex(coordsA);
      TypeA valueA = dataA[serialIdxA];
      if ( std::is_same<TypeA, CobaltComplexFloat>::value
        || std::is_same<TypeA, CobaltComplexDouble>::value) {
        if ( Solution::problem.tensorA.getDataType() == cobaltDataTypeComplexConjugateHalf
          || Solution::problem.tensorA.getDataType() == cobaltDataTypeComplexConjugateSingle
          || Solution::problem.tensorA.getDataType() == cobaltDataTypeComplexConjugateDouble) {
          complexConjugate<TypeA>( valueA );
        }
      }

      size_t serialIdxB = Solution::problem.tensorB.getIndex(coordsB);
      TypeB valueB = dataB[serialIdxB];
      if (std::is_same<TypeB, CobaltComplexFloat>::value
        || std::is_same<TypeB, CobaltComplexDouble>::value) {
        if ( Solution::problem.tensorB.getDataType() == cobaltDataTypeComplexConjugateHalf
          || Solution::problem.tensorB.getDataType() == cobaltDataTypeComplexConjugateSingle
          || Solution::problem.tensorB.getDataType() == cobaltDataTypeComplexConjugateDouble) {
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


    size_t serialIdxC = Solution::problem.tensorC.getIndex(freeCoord);
    if (alpha.data) {
      TypeAlpha *alphaData = static_cast<TypeAlpha*>(alpha.data);
      sumC = multiply<TypeC,TypeAlpha,TypeC>(*alphaData,sumC);
    }
    if (beta.data) {
      TypeBeta *betaData = static_cast<TypeBeta*>(beta.data);
      TypeC tmp = multiply<TypeC,TypeBeta,TypeC>(*betaData, dataC[serialIdxC]);
      sumC = add<TypeC,TypeC,TypeC>(tmp,sumC);
    }

    dataC[serialIdxC] = sumC;

    // increment free coord
    freeCoord[0]++;
    for (size_t f = 0; f < Solution::problem.tensorC.numDims()-1; f++) {
      if (freeCoord[f] >= Solution::problem.tensorC[f].size) {
        freeCoord[f] = 0;
        freeCoord[f+1]++;
      }
    }
    if (freeCoord[Solution::problem.tensorC.numDims() - 1] >= Solution::problem.tensorC[Solution::problem.tensorC.numDims() - 1].size) {
      break; // free index range exit criteria
    }

  } // free range

  return cobaltStatusSuccess;
} // referenceTensorContraction


/*******************************************************************************
 * toString
 ******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
std::string SolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toString( size_t indentLevel ) const {
  return "CobaltSolutionTensorContractionCPU";
}


/*******************************************************************************
* toString
******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
std::string SolutionTensorContractionCPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toStringDetailXML( size_t indentLevel ) const {
  std::string detail = Cobalt::indent(indentLevel);
  detail += "<ImplementationDetails/>\n";
  return detail;
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
      return std::make_tuple(nullptr, cobaltStatusProblemNotSupported);
  }
}

} // namespace

