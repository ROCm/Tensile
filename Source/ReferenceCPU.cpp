/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#include "ReferenceCPU.h"
#include "MathTemplates.h"

#include <algorithm>


/*******************************************************************************
 * enqueue
 ******************************************************************************/
template< typename Type >
TensileStatus tensileReferenceCPU(
    Type *dataC,
    Type *dataA,
    Type *dataB,
    Type alpha,
    Type beta,
    unsigned int numIndicesC,
    unsigned int numIndicesAB,
    unsigned int *indexAssignmentsA,
    unsigned int *indexAssignmentsB
    ) {

  // index sizes
  unsigned int numIndicesFreeC = Solution::problem.tensorC.numDims();
  unsigned int numIndicesSummation = static_cast<unsigned int>(Solution::problem.indicesSummation.size());
  // unsigned int numIndicesFreeAB = Solution::problem.tensorA.numDims() - numIndicesSummation;

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
      if (
#ifdef Tensile_Enable_FP16_HOST
           std::is_same<TypeA, TensileComplexHalf>() ||
#endif
           std::is_same<TypeA, TensileComplexFloat>()
        || std::is_same<TypeA, TensileComplexDouble>() ) {
        if (
#ifdef Tensile_Enable_FP16_HOST
             Solution::problem.tensorA.getDataType() == tensileDataTypeComplexConjugateHalf ||
#endif
             Solution::problem.tensorA.getDataType() == tensileDataTypeComplexConjugateSingle
          || Solution::problem.tensorA.getDataType() == tensileDataTypeComplexConjugateDouble) {
          complexConjugate<TypeA>( valueA );
        }
      }

      size_t serialIdxB = Solution::problem.tensorB.getIndex(coordsB);
      TypeB valueB = dataB[serialIdxB];
      if (
#ifdef Tensile_Enable_FP16_HOST
           std::is_same<TypeB, TensileComplexHalf>() ||
#endif
           std::is_same<TypeB, TensileComplexFloat>()
        || std::is_same<TypeB, TensileComplexDouble>() ) {
        if (
#ifdef Tensile_Enable_FP16_HOST
             Solution::problem.tensorB.getDataType() == tensileDataTypeComplexConjugateHalf ||
#endif
             Solution::problem.tensorB.getDataType() == tensileDataTypeComplexConjugateSingle
          || Solution::problem.tensorB.getDataType() == tensileDataTypeComplexConjugateDouble) {
          complexConjugate<TypeB>( valueB );
        }
      }

      TypeC product = multiply<TypeC,TypeA,TypeB>( valueA, valueB);

      sumC = add<TypeC,TypeA,TypeB>(sumC,product);

      // increment bound coord
      boundCoord[numIndicesSummation-1]++;
      for ( size_t b = numIndicesSummation - 1; b > 0 ; b--) {
        if ( boundCoord[b] >= boundIndexSizes[b]) {
          boundCoord[b] = 0;
          boundCoord[b-1]++;
        }
      }
      //if (boundCoord[numIndicesSummation - 1] >= boundIndexSizes[numIndicesSummation - 1]) {
      if (boundCoord[0] >= boundIndexSizes[0]) {
        break; // bound index range exit criteria
      }

    } // bound range


    size_t serialIdxC = Solution::problem.tensorC.getIndex(freeCoord);
    if (alpha.data) {
      const TypeAlpha *alphaData = static_cast<const TypeAlpha*>(alpha.data);
      sumC = multiply<TypeC,TypeAlpha,TypeC>(*alphaData,sumC);
    }
    if (beta.data) {
      const TypeBeta *betaData = static_cast<const TypeBeta*>(beta.data);
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

  return tensileStatusSuccess;
} // referenceTensorContraction





/*******************************************************************************
 * tensileGetSolution
 * need to list all wanted template variants for compiler in this file
 ******************************************************************************/
/*
std::tuple<Solution *,TensileStatus> getSolutionCPU( const Problem & problem) {

  bool problemIsTensorContraction = true;

  if (problemIsTensorContraction) {
    switch(problem.getDataTypeC()) {
    case tensileDataTypeSingle:
      return std::make_tuple(new Tensile::SolutionTensorContractionCPU<float,float,float,float,float>( problem ), tensileStatusSuccess );
    case tensileDataTypeDouble:
      return std::make_tuple(new Tensile::SolutionTensorContractionCPU<double,double,double,double,double>( problem ), tensileStatusSuccess );
    case tensileDataTypeComplexSingle:
    case tensileDataTypeComplexConjugateSingle:
      return std::make_tuple(new Tensile::SolutionTensorContractionCPU<TensileComplexFloat,TensileComplexFloat,TensileComplexFloat,TensileComplexFloat,TensileComplexFloat>( problem ), tensileStatusSuccess );
    case tensileDataTypeComplexDouble:
    case tensileDataTypeComplexConjugateDouble:
      return std::make_tuple(new Tensile::SolutionTensorContractionCPU<TensileComplexDouble,TensileComplexDouble,TensileComplexDouble,TensileComplexDouble,TensileComplexDouble>( problem ), tensileStatusSuccess );
#ifdef Tensile_ENABLE_FP16_HOST
    case tensileDataTypeHalf:
      return std::make_tuple(new Tensile::SolutionTensorContractionCPU<TensileHalf,TensileHalf,TensileHalf,TensileHalf,TensileHalf>( problem ), tensileStatusSuccess );
    case tensileDataTypeComplexHalf:
    case tensileDataTypeComplexConjugateHalf:
      return std::make_tuple(new Tensile::SolutionTensorContractionCPU<TensileComplexHalf,TensileComplexHalf,TensileComplexHalf,TensileComplexHalf,TensileComplexHalf>( problem ), tensileStatusSuccess );
#endif
    case tensileNumDataTypes:
    case tensileDataTypeNone:
      return std::make_tuple(nullptr, tensileStatusFailure);
    }
    //default:
    //  return std::make_tuple(nullptr, tensileStatusProblemNotSupported);
    //}
  } else {
      return std::make_tuple(nullptr, tensileStatusFailure);
  }
}
*/

