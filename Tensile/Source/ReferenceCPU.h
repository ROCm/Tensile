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

#ifndef REFERENCE_CPU_H
#define REFERENCE_CPU_H
#include "TensileTypes.h"
#include "MathTemplates.h"
#include <vector>


/*******************************************************************************
 * Reference Tensor Contraction
 ******************************************************************************/
template< typename Type >
TensileStatus tensileReferenceCPU(
    Type *dataC,
    const Type *dataA,
    const Type *dataB,
    Type alpha,
    Type beta,
    unsigned int totalIndices,
    unsigned int *sizes,
    unsigned int numIndicesC,
    unsigned int numIndicesAB,
    const unsigned int *indexAssignmentsA,
    const unsigned int *indexAssignmentsB,
    bool complexConjugateA,
    bool complexConjugateB,
    size_t validationStride // = 1 means do all
  ) {

  // sizes
  unsigned int *sizesA = new unsigned int[numIndicesAB];
  unsigned int *sizesB = new unsigned int[numIndicesAB];
  unsigned int *stridesC = new unsigned int[numIndicesC];
  unsigned int *stridesA = new unsigned int[numIndicesAB];
  unsigned int *stridesB = new unsigned int[numIndicesAB];
  for (unsigned int i = 0; i < numIndicesAB; i++) {
    sizesA[i] = sizes[indexAssignmentsA[i]];
    sizesB[i] = sizes[indexAssignmentsB[i]];
  }
  // strides
  stridesC[0] = 1;
  stridesA[0] = 1;
  stridesB[0] = 1;
  for (unsigned int i = 1; i < numIndicesAB; i++) {
    stridesA[i] = stridesA[i-1] * sizesA[i-1];
    stridesB[i] = stridesB[i-1] * sizesB[i-1];
  }
  for (unsigned int i = 1; i < numIndicesC; i++) {
    stridesC[i] = stridesC[i-1] * sizes[i-1];
  }


  unsigned int numIndicesSummation = totalIndices - numIndicesC;

  // allocate coords and sizes
  std::vector<unsigned int> freeCoord(numIndicesC);
  std::vector<unsigned int> boundCoord( numIndicesSummation );
  std::vector<unsigned int> boundIndexSizes( numIndicesSummation );

  // initialize coords & sizes
  for (unsigned int i = 0; i < numIndicesC; i++) {
    freeCoord[i] = 0;
  }

  for (size_t i = 0; i < numIndicesAB; i++) {
    if ( indexAssignmentsA[i] >= numIndicesC) {
      boundIndexSizes[indexAssignmentsA[i]-numIndicesC] = sizes[indexAssignmentsA[i]];
    }
  }

  // allocate tensor coords
  std::vector<unsigned int> coordsC( numIndicesC );
  std::vector<unsigned int> coordsA( numIndicesAB );
  std::vector<unsigned int> coordsB( numIndicesAB );

  bool moreIndicesC = true;
  while (moreIndicesC) { // iterate over entire free index range

    Type sumC = tensileGetZero<Type>();
    // reset summation indices
    for (unsigned int b = 0; b < numIndicesSummation; b++) {
      boundCoord[b] = 0;
    }
    while (true) { // iterate over entire bound index range
      
      // convert free/bound coord into tensorA,B 
      for (unsigned int i = 0; i < numIndicesAB; i++) {
        if (indexAssignmentsA[i] < numIndicesC) {
          coordsA[i] = freeCoord[indexAssignmentsA[i]];
        } else {
          coordsA[i] = boundCoord[indexAssignmentsA[i]-numIndicesC];
        }
      }
      for (unsigned int i = 0; i < numIndicesAB; i++) {
        if (indexAssignmentsB[i] < numIndicesC) {
          coordsB[i] = freeCoord[indexAssignmentsB[i]];
        } else {
          coordsB[i] = boundCoord[indexAssignmentsB[i]-numIndicesC];
        }
      }
      
      size_t serialIdxA = 0;
      for (unsigned int i = 0; i < numIndicesAB; i++) {
        serialIdxA += coordsA[i]*stridesA[i];
      }
      Type valueA = dataA[serialIdxA];
      if (
#ifdef Tensile_Enable_FP16_HOST
           std::is_same<Type, TensileComplexHalf>() ||
#endif
           std::is_same<Type, TensileComplexFloat>()
        || std::is_same<Type, TensileComplexDouble>() ) {
        if ( complexConjugateA ) {
          tensileComplexConjugate<Type>( valueA );
        }
      }

      size_t serialIdxB = 0;
      for (unsigned int i = 0; i < numIndicesAB; i++) {
        serialIdxB += coordsB[i]*stridesB[i];
      }
      Type valueB = dataB[serialIdxB];
      if (
#ifdef Tensile_Enable_FP16_HOST
           std::is_same<Type, TensileComplexHalf>() ||
#endif
           std::is_same<Type, TensileComplexFloat>()
        || std::is_same<Type, TensileComplexDouble>() ) {
        if ( complexConjugateB ) {
          tensileComplexConjugate<Type>( valueB );
        }
      }

      Type product = tensileMultiply<Type>( valueA, valueB );
      //printf("%f = %f * %f\n", product, valueA, valueB );

      sumC = tensileAdd<Type>(sumC,product);

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


    size_t serialIdxC = 0;
    for (unsigned int i = 0; i < numIndicesC; i++) {
      serialIdxC += freeCoord[i]*stridesC[i];
    }
    sumC = tensileMultiply<Type>(alpha,sumC);
    Type tmp = tensileMultiply<Type>(beta, dataC[serialIdxC]);
    sumC = tensileAdd<Type>(tmp,sumC);

    dataC[serialIdxC] = sumC;

    // increment free coord
    // skip = 1, validate everything
    for (size_t i = 0; i < validationStride; i++) {
      freeCoord[0]++;
      for (size_t f = 0; f < numIndicesC-1; f++) {
        if (freeCoord[f] >= sizes[f]) {
          freeCoord[f] = 0;
          freeCoord[f+1]++;
        }
      }
      if (freeCoord[numIndicesC - 1] >= sizes[numIndicesC - 1]) {
        moreIndicesC = false;
        break; // free index range exit criteria
      }
    }

  } // free range
  delete[] sizesA;
  delete[] sizesB;

  return tensileStatusSuccess;
} // referenceTensorContraction

#endif

