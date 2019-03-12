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
#include <type_traits>


/*******************************************************************************
 * Reference Tensor Contraction
 ******************************************************************************/
typedef union{
  int8_t byte[4];
  uint32_t uval;
  int32_t val;
} int8x4;

void unpack_int8x4(uint32_t in, int32_t &out_0, int32_t &out_1, int32_t &out_2, int32_t &out_3)
{
  int8x4 x;
  x.uval = in;
  out_0 = x.byte[0];
  out_1 = x.byte[1];
  out_2 = x.byte[2];
  out_3 = x.byte[3];
}

template< typename Type, typename DestType >
TensileStatus tensileReferenceCPU(
    DestType *dataD,
    const DestType *dataC,
    const Type *dataA,
    const Type *dataB,
    const unsigned int lda,
    const unsigned int ldb,
    const unsigned int ldc,
    const unsigned int ldd,
    const unsigned int stride_a,
    const unsigned int stride_b,
    const unsigned int stride_c,
    const unsigned int stride_d,
    DestType alpha,
    DestType beta,
    unsigned int totalIndices,
    const unsigned int *sizes,
    const unsigned int *minStrides,
    unsigned int numIndicesC,
    unsigned int numIndicesAB,
    const unsigned int *indexAssignmentsA,
    const unsigned int *indexAssignmentsB,
    bool complexConjugateA,
    bool complexConjugateB,
    size_t validationStride, // = 1 means do all
    bool useHighPrecisionAccumulate
  ) {

  // Only allow high precision accumulate if Type is half
  bool localUseHighPrecisionAccumulate = useHighPrecisionAccumulate && std::is_same<Type, TensileHalf>::value;

  // sizes
  unsigned int *sizesA = new unsigned int[numIndicesAB];
  unsigned int *sizesB = new unsigned int[numIndicesAB];

  // Stride in each index
  std::vector<unsigned int> strides(totalIndices);

  unsigned int *stridesD = new unsigned int[numIndicesC];
  unsigned int *stridesC = new unsigned int[numIndicesC];
  unsigned int *stridesA = new unsigned int[numIndicesAB];
  unsigned int *stridesB = new unsigned int[numIndicesAB];
  for (unsigned int i = 0; i < totalIndices; i++) {
    strides[i] = std::max(minStrides[i], sizes[i]);
  }

  for (unsigned int i = 0; i < numIndicesAB; i++) {
    sizesA[i] = sizes[indexAssignmentsA[i]];
    sizesB[i] = sizes[indexAssignmentsB[i]];
  }
  // strides
  stridesD[0] = 1;
  stridesC[0] = 1;
  stridesA[0] = 1;
  stridesB[0] = 1;
  for (unsigned int i = 1; i < numIndicesAB; i++) {
    stridesA[i] = stridesA[i-1] * strides[indexAssignmentsA[i-1]];
    stridesB[i] = stridesB[i-1] * strides[indexAssignmentsB[i-1]];
  }
  for (unsigned int i = 1; i < numIndicesC; i++) {
    stridesD[i] = stridesD[i-1] * strides[i-1];
    stridesC[i] = stridesC[i-1] * strides[i-1];
  }
  if (lda != std::numeric_limits<unsigned int>::max())  stridesA[1] = lda;
  if (ldb != std::numeric_limits<unsigned int>::max())  stridesB[1] = ldb;
  if (ldc != std::numeric_limits<unsigned int>::max())  stridesC[1] = ldc;
  if (ldd != std::numeric_limits<unsigned int>::max())  stridesD[1] = ldd;
  if (stride_a != std::numeric_limits<unsigned int>::max())  stridesA[2] = stride_a;
  if (stride_b != std::numeric_limits<unsigned int>::max())  stridesB[2] = stride_b;
  if (stride_c != std::numeric_limits<unsigned int>::max())  stridesC[2] = stride_c;
  if (stride_d != std::numeric_limits<unsigned int>::max())  stridesD[2] = stride_d;

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
    float sumCfloat = 0.0f;
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
#ifdef Tensile_ENABLE_HALF
//           std::is_same<Type, TensileComplexHalf>() ||
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
#ifdef Tensile_ENABLE_HALF
//           std::is_same<Type, TensileComplexHalf>() ||
#endif
           std::is_same<Type, TensileComplexFloat>()
        || std::is_same<Type, TensileComplexDouble>() ) {
        if ( complexConjugateB ) {
          tensileComplexConjugate<Type>( valueB );
        }
      }

      if(std::is_same<Type, uint32_t>() && std::is_same<DestType, int32_t>())
      {
         int32_t a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3;
         unpack_int8x4(valueA, a_0, a_1, a_2, a_3);
         unpack_int8x4(valueB, b_0, b_1, b_2, b_3);
         sumC += (a_0 * b_0) + (a_1 * b_1) + (a_2 * b_2) + (a_3 * b_3);
      }
      else
      {
        Type product = tensileMultiply<Type>( valueA, valueB );
        //printf("%f = %f * %f\n", product, valueA, valueB );

        if (localUseHighPrecisionAccumulate)
          sumCfloat = tensileAdd<float>(sumCfloat,(float)product);
        else
          sumC = tensileAdd<Type>(sumC,product);
      }

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


    size_t serialIdxD = 0;
    size_t serialIdxC = 0;
    for (unsigned int i = 0; i < numIndicesC; i++) {
      serialIdxD += freeCoord[i]*stridesD[i];
      serialIdxC += freeCoord[i]*stridesC[i];
    }
    if (localUseHighPrecisionAccumulate)
      sumCfloat = tensileMultiply<float>((float)alpha,sumCfloat);
    else
      sumC = tensileMultiply<Type>(alpha,sumC);
    if (!tensileIsZero(beta)) {
      Type tmp = tensileMultiply<Type>(beta, dataC[serialIdxC]);
      if (localUseHighPrecisionAccumulate)
        sumCfloat = tensileAdd<float>((float)tmp,sumCfloat);
      else
        sumC = tensileAdd<Type>(tmp,sumC);
    }

    if (localUseHighPrecisionAccumulate)
      dataD[serialIdxD] = (Type)sumCfloat;
    else
      dataD[serialIdxD] = sumC;

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

