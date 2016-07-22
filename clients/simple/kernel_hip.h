#include <hip_runtime.h>

#if 0

// unroll can be 4, 8, 16 or 32
// unroll = 8 hangs
#define NUM_UNROLL_ITER     8

/*******************************************************************************
* Kernel
******************************************************************************/

/* CT_SSSSS_Cij_Sk_Aik_Bjk_i16x4f_j16x4f_k16_O2 */

/* tile parameters */
#define WG_0I           16
#define WG_1J           16
#define UT_0I       4
#define UT_1J       4
#define MT_0I       64
#define MT_1J       64


/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* global tile indices being loaded */
/* fast read */
#define globalIdxA0I(LID) (groupIdx0I*MT_0I + (localSerial+(LID)*WG_0I*WG_1J)/NUM_UNROLL_ITER)
#define globalIdxAK(LID) (localSerial%NUM_UNROLL_ITER)
/* fast read */
#define globalIdxBK(LID) ((localSerial+(LID)*WG_0I*WG_1J)/MT_1J)
#define globalIdxB1J(LID) (groupIdx1J*MT_1J + (localSerial+(LID)*WG_0I*WG_1J)%MT_1J)

/* global non-tile indices being loaded */


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT_0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT_1J) )

/* local indices being written */
#define localA0I (localSerial / NUM_UNROLL_ITER)
#define localAK (localSerial % NUM_UNROLL_ITER)
#define localAStride (WG_0I*WG_1J/NUM_UNROLL_ITER)
#define localB1J ( localSerial / MT_1J )
#define localBK ( localSerial % MT_1J )
#define localBStride  (WG_0I*WG_1J)

/* data types */
#define TYPE_A float
#define TYPE_B float
#define TYPE_C float
#define TYPE_ALPHA float
#define TYPE_BETA float
#define FMA(A,B,DST) fmaf(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 4x4 micro-tile */
#define UT \
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  offA += MT_0I; \
  offB += MT_1J; \
  TYPE_MAD(rA[0],rB[0],rC[0][0]); \
  TYPE_MAD(rA[0],rB[1],rC[0][1]); \
  TYPE_MAD(rA[0],rB[2],rC[0][2]); \
  TYPE_MAD(rA[0],rB[3],rC[0][3]); \
  TYPE_MAD(rA[1],rB[0],rC[1][0]); \
  TYPE_MAD(rA[1],rB[1],rC[1][1]); \
  TYPE_MAD(rA[1],rB[2],rC[1][2]); \
  TYPE_MAD(rA[1],rB[3],rC[1][3]); \
  TYPE_MAD(rA[2],rB[0],rC[2][0]); \
  TYPE_MAD(rA[2],rB[1],rC[2][1]); \
  TYPE_MAD(rA[2],rB[2],rC[2][2]); \
  TYPE_MAD(rA[2],rB[3],rC[2][3]); \
  TYPE_MAD(rA[3],rB[0],rC[3][0]); \
  TYPE_MAD(rA[3],rB[1],rC[3][1]); \
  TYPE_MAD(rA[3],rB[2],rC[3][2]); \
  TYPE_MAD(rA[3],rB[3],rC[3][3]); \
  __syncthreads();

/* preprocessor definitions of kernel arguments*/
#define strideC0I 1
#define strideA0I 1
#define strideB1J 1


extern "C"
__global__ void kernel_hip(
  hipLaunchParm lp,
  float       *          C,
  float const * __restrict__ A,
  float const * __restrict__ B,
  float const alpha,
  float const beta,
  unsigned int const strideC1J,
  unsigned int const strideAK,
  unsigned int const strideBK,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK ) {

  /* allocate registers */
  TYPE_C rC[UT_0I][UT_1J] = {{0}};
  TYPE_A rA[UT_0I];
  TYPE_B rB[UT_1J];

  /* allocate local memory */
  __shared__ TYPE_A localA[NUM_UNROLL_ITER*MT_0I];
  __shared__ TYPE_B localB[NUM_UNROLL_ITER*MT_1J];

  /* c indices */
  unsigned int groupIdx0I = hc_get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = hc_get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = hc_get_workitem_id(0); // d0
  unsigned int localIdx1J = hc_get_workitem_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT_1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT_0I + localIdx0I;
  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    TYPE_A *lA = localA + GET_LOCAL_INDEX_A(localA0I, localAK);
    TYPE_B *lB = localB + GET_LOCAL_INDEX_B(localB1J, localBK);
    __syncthreads();

    /* load global -> local */
    lA[ 0*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(0), globalIdxAK(0) ) ];
#if NUM_UNROLL_ITER > 4
    lA[ 1*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(1), globalIdxAK(1) ) ];
#endif
#if NUM_UNROLL_ITER > 8
    lA[ 2*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(2), globalIdxAK(2) ) ];
    lA[ 3*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(3), globalIdxAK(3) ) ];
#endif
#if NUM_UNROLL_ITER > 16
    lA[ 4*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(4), globalIdxAK(4) ) ];
    lA[ 5*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(5), globalIdxAK(5) ) ];
    lA[ 6*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(6), globalIdxAK(6) ) ];
    lA[ 7*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(7), globalIdxAK(7) ) ];
#endif

    lB[ 0*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(0), globalIdxBK(0) ) ];
#if NUM_UNROLL_ITER > 4
    lB[ 1*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(1), globalIdxBK(1) ) ];
#endif
#if NUM_UNROLL_ITER > 8
    lB[ 2*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(2), globalIdxBK(2) ) ];
    lB[ 3*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(3), globalIdxBK(3) ) ];
#endif
#if NUM_UNROLL_ITER > 16
    lB[ 4*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(4), globalIdxBK(4) ) ];
    lB[ 5*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(5), globalIdxBK(5) ) ];
    lB[ 6*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(6), globalIdxBK(6) ) ];
    lB[ 7*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(7), globalIdxBK(7) ) ];
#endif

    __syncthreads();
    unsigned int offA = localIdx0I; // d0
    unsigned int offB = localIdx1J; // d1

                                    /* do mads */
    UT
      UT
      UT
      UT
#if NUM_UNROLL_ITER > 4
      UT
      UT
      UT
      UT
#endif
#if NUM_UNROLL_ITER > 8
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
#endif
#if NUM_UNROLL_ITER > 16
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
      UT
#endif


      A += strideAK*NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
  } while (--sumIterK > 0);


  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)

}

#else

// Cobalt sgemm_NT_128x128x8_prefetch


/* CT_SSSSS_Cij_Sk_Aik_Bjk_i16x8f_j16x8f_nl4x4_k8_O2 */

/* tile parameters */
#define WG_0I  16
#define WG_1J  16
#define UT_0I   8
#define UT_1J   8
#define MT_0I  128
#define MT_1J  128
#define MT_0I_2  128
#define MT_1J_2  128
#define UNROLL  8
#define PAD     0

/* num loads parallel and perpendicular to coalesced dimension */
#define NL_PARA_A 4
#define NL_PARA_B 4
#define NL_PERP_A 1
#define NL_PERP_B 1

/* load size parallel and perpendicular to coalesced dimension */
#define LS_PARA_A 32
#define LS_PERP_A 8
#define LS_PARA_B 32
#define LS_PERP_B 8

/* global memory indices */
#define GLOBAL_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GLOBAL_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GLOBAL_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* global non-tile indices being loaded */

/* data types */
#define TYPE_A     float
#define TYPE_B     float
#define TYPE_C     float
#define TYPE_ALPHA float
#define TYPE_BETA  float
#define MAD(A,B,DST) DST += (A)*(B);

/* MADs */
#define TYPE_MAD(MULA,MULB,DST) DST = MAD(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 8x8 micro-tile */
// load
#define MICRO_TILE_PREFETCH \
  rA_red[0] = localReadPtrA[ offA + 0*WG_0I]; \
  rA_red[1] = localReadPtrA[ offA + 1*WG_0I]; \
  rA_red[2] = localReadPtrA[ offA + 2*WG_0I]; \
  rA_red[3] = localReadPtrA[ offA + 3*WG_0I]; \
  rA_red[4] = localReadPtrA[ offA + 4*WG_0I]; \
  rA_red[5] = localReadPtrA[ offA + 5*WG_0I]; \
  rA_red[6] = localReadPtrA[ offA + 6*WG_0I]; \
  rA_red[7] = localReadPtrA[ offA + 7*WG_0I]; \
  rB_red[0] = localReadPtrB[ offB + 0*WG_1J]; \
  rB_red[1] = localReadPtrB[ offB + 1*WG_1J]; \
  rB_red[2] = localReadPtrB[ offB + 2*WG_1J]; \
  rB_red[3] = localReadPtrB[ offB + 3*WG_1J]; \
  rB_red[4] = localReadPtrB[ offB + 4*WG_1J]; \
  rB_red[5] = localReadPtrB[ offB + 5*WG_1J]; \
  rB_red[6] = localReadPtrB[ offB + 6*WG_1J]; \
  rB_red[7] = localReadPtrB[ offB + 7*WG_1J]; \
  offA += (MT_0I+PAD); \
  offB += (MT_1J+PAD); \
  /* mem_fence(CLK_LOCAL_MEM_FENCE); */
  

#define MICRO_TILE_2 \
  /* prefetch black, compute red */ \
  rA_black[0] = localReadPtrA[offA + 0*WG_0I]; \
  rA_black[1] = localReadPtrA[offA + 1*WG_0I]; \
  rA_black[2] = localReadPtrA[offA + 2*WG_0I]; \
  rA_black[3] = localReadPtrA[offA + 3*WG_0I]; \
  rA_black[4] = localReadPtrA[offA + 4*WG_0I]; \
  rA_black[5] = localReadPtrA[offA + 5*WG_0I]; \
  rA_black[6] = localReadPtrA[offA + 6*WG_0I]; \
  rA_black[7] = localReadPtrA[offA + 7*WG_0I]; \
  \
  rB_black[0] = localReadPtrB[offB + 0*WG_1J]; \
  rB_black[1] = localReadPtrB[offB + 1*WG_1J]; \
  rB_black[2] = localReadPtrB[offB + 2*WG_1J]; \
  rB_black[3] = localReadPtrB[offB + 3*WG_1J]; \
  rB_black[4] = localReadPtrB[offB + 4*WG_1J]; \
  rB_black[5] = localReadPtrB[offB + 5*WG_1J]; \
  rB_black[6] = localReadPtrB[offB + 6*WG_1J]; \
  rB_black[7] = localReadPtrB[offB + 7*WG_1J]; \
  \
  /* mem_fence(CLK_LOCAL_MEM_FENCE); */ \
  \
  offA += (MT_0I+PAD); \
  offB += (MT_1J+PAD); \
  \
  TYPE_MAD( rA_red[0], rB_red[0], rC[0][0]); \
  TYPE_MAD( rA_red[0], rB_red[1], rC[0][1]); \
  TYPE_MAD( rA_red[0], rB_red[2], rC[0][2]); \
  TYPE_MAD( rA_red[0], rB_red[3], rC[0][3]); \
  TYPE_MAD( rA_red[0], rB_red[4], rC[0][4]); \
  TYPE_MAD( rA_red[0], rB_red[5], rC[0][5]); \
  TYPE_MAD( rA_red[0], rB_red[6], rC[0][6]); \
  TYPE_MAD( rA_red[0], rB_red[7], rC[0][7]); \
  \
  TYPE_MAD( rA_red[1], rB_red[0], rC[1][0]); \
  TYPE_MAD( rA_red[1], rB_red[1], rC[1][1]); \
  TYPE_MAD( rA_red[1], rB_red[2], rC[1][2]); \
  TYPE_MAD( rA_red[1], rB_red[3], rC[1][3]); \
  TYPE_MAD( rA_red[1], rB_red[4], rC[1][4]); \
  TYPE_MAD( rA_red[1], rB_red[5], rC[1][5]); \
  TYPE_MAD( rA_red[1], rB_red[6], rC[1][6]); \
  TYPE_MAD( rA_red[1], rB_red[7], rC[1][7]); \
  \
  TYPE_MAD( rA_red[2], rB_red[0], rC[2][0]); \
  TYPE_MAD( rA_red[2], rB_red[1], rC[2][1]); \
  TYPE_MAD( rA_red[2], rB_red[2], rC[2][2]); \
  TYPE_MAD( rA_red[2], rB_red[3], rC[2][3]); \
  TYPE_MAD( rA_red[2], rB_red[4], rC[2][4]); \
  TYPE_MAD( rA_red[2], rB_red[5], rC[2][5]); \
  TYPE_MAD( rA_red[2], rB_red[6], rC[2][6]); \
  TYPE_MAD( rA_red[2], rB_red[7], rC[2][7]); \
  \
  TYPE_MAD( rA_red[3], rB_red[0], rC[3][0]); \
  TYPE_MAD( rA_red[3], rB_red[1], rC[3][1]); \
  TYPE_MAD( rA_red[3], rB_red[2], rC[3][2]); \
  TYPE_MAD( rA_red[3], rB_red[3], rC[3][3]); \
  TYPE_MAD( rA_red[3], rB_red[4], rC[3][4]); \
  TYPE_MAD( rA_red[3], rB_red[5], rC[3][5]); \
  TYPE_MAD( rA_red[3], rB_red[6], rC[3][6]); \
  TYPE_MAD( rA_red[3], rB_red[7], rC[3][7]); \
  \
  TYPE_MAD( rA_red[4], rB_red[0], rC[4][0]); \
  TYPE_MAD( rA_red[4], rB_red[1], rC[4][1]); \
  TYPE_MAD( rA_red[4], rB_red[2], rC[4][2]); \
  TYPE_MAD( rA_red[4], rB_red[3], rC[4][3]); \
  TYPE_MAD( rA_red[4], rB_red[4], rC[4][4]); \
  TYPE_MAD( rA_red[4], rB_red[5], rC[4][5]); \
  TYPE_MAD( rA_red[4], rB_red[6], rC[4][6]); \
  TYPE_MAD( rA_red[4], rB_red[7], rC[4][7]); \
  \
  TYPE_MAD( rA_red[5], rB_red[0], rC[5][0]); \
  TYPE_MAD( rA_red[5], rB_red[1], rC[5][1]); \
  TYPE_MAD( rA_red[5], rB_red[2], rC[5][2]); \
  TYPE_MAD( rA_red[5], rB_red[3], rC[5][3]); \
  TYPE_MAD( rA_red[5], rB_red[4], rC[5][4]); \
  TYPE_MAD( rA_red[5], rB_red[5], rC[5][5]); \
  TYPE_MAD( rA_red[5], rB_red[6], rC[5][6]); \
  TYPE_MAD( rA_red[5], rB_red[7], rC[5][7]); \
  \
  TYPE_MAD( rA_red[6], rB_red[0], rC[6][0]); \
  TYPE_MAD( rA_red[6], rB_red[1], rC[6][1]); \
  TYPE_MAD( rA_red[6], rB_red[2], rC[6][2]); \
  TYPE_MAD( rA_red[6], rB_red[3], rC[6][3]); \
  TYPE_MAD( rA_red[6], rB_red[4], rC[6][4]); \
  TYPE_MAD( rA_red[6], rB_red[5], rC[6][5]); \
  TYPE_MAD( rA_red[6], rB_red[6], rC[6][6]); \
  TYPE_MAD( rA_red[6], rB_red[7], rC[6][7]); \
  \
  TYPE_MAD( rA_red[7], rB_red[0], rC[7][0]); \
  TYPE_MAD( rA_red[7], rB_red[1], rC[7][1]); \
  TYPE_MAD( rA_red[7], rB_red[2], rC[7][2]); \
  TYPE_MAD( rA_red[7], rB_red[3], rC[7][3]); \
  TYPE_MAD( rA_red[7], rB_red[4], rC[7][4]); \
  TYPE_MAD( rA_red[7], rB_red[5], rC[7][5]); \
  TYPE_MAD( rA_red[7], rB_red[6], rC[7][6]); \
  TYPE_MAD( rA_red[7], rB_red[7], rC[7][7]); \
  \
  /* mem_fence(CLK_LOCAL_MEM_FENCE); */ \
  /* prefetch red, compute black */ \
  rA_red[0] = localReadPtrA[offA + 0*WG_0I]; \
  rA_red[1] = localReadPtrA[offA + 1*WG_0I]; \
  rA_red[2] = localReadPtrA[offA + 2*WG_0I]; \
  rA_red[3] = localReadPtrA[offA + 3*WG_0I]; \
  rA_red[4] = localReadPtrA[offA + 4*WG_0I]; \
  rA_red[5] = localReadPtrA[offA + 5*WG_0I]; \
  rA_red[6] = localReadPtrA[offA + 6*WG_0I]; \
  rA_red[7] = localReadPtrA[offA + 7*WG_0I]; \
  \
  rB_red[0] = localReadPtrB[offB + 0*WG_1J]; \
  rB_red[1] = localReadPtrB[offB + 1*WG_1J]; \
  rB_red[2] = localReadPtrB[offB + 2*WG_1J]; \
  rB_red[3] = localReadPtrB[offB + 3*WG_1J]; \
  rB_red[4] = localReadPtrB[offB + 4*WG_1J]; \
  rB_red[5] = localReadPtrB[offB + 5*WG_1J]; \
  rB_red[6] = localReadPtrB[offB + 6*WG_1J]; \
  rB_red[7] = localReadPtrB[offB + 7*WG_1J]; \
  \
  /* mem_fence(CLK_LOCAL_MEM_FENCE); */ \
  \
  offA += (MT_0I+PAD); \
  offB += (MT_1J+PAD); \
  \
  TYPE_MAD( rA_black[0], rB_black[0], rC[0][0]); \
  TYPE_MAD( rA_black[0], rB_black[1], rC[0][1]); \
  TYPE_MAD( rA_black[0], rB_black[2], rC[0][2]); \
  TYPE_MAD( rA_black[0], rB_black[3], rC[0][3]); \
  TYPE_MAD( rA_black[0], rB_black[4], rC[0][4]); \
  TYPE_MAD( rA_black[0], rB_black[5], rC[0][5]); \
  TYPE_MAD( rA_black[0], rB_black[6], rC[0][6]); \
  TYPE_MAD( rA_black[0], rB_black[7], rC[0][7]); \
  \
  TYPE_MAD( rA_black[1], rB_black[0], rC[1][0]); \
  TYPE_MAD( rA_black[1], rB_black[1], rC[1][1]); \
  TYPE_MAD( rA_black[1], rB_black[2], rC[1][2]); \
  TYPE_MAD( rA_black[1], rB_black[3], rC[1][3]); \
  TYPE_MAD( rA_black[1], rB_black[4], rC[1][4]); \
  TYPE_MAD( rA_black[1], rB_black[5], rC[1][5]); \
  TYPE_MAD( rA_black[1], rB_black[6], rC[1][6]); \
  TYPE_MAD( rA_black[1], rB_black[7], rC[1][7]); \
  \
  TYPE_MAD( rA_black[2], rB_black[0], rC[2][0]); \
  TYPE_MAD( rA_black[2], rB_black[1], rC[2][1]); \
  TYPE_MAD( rA_black[2], rB_black[2], rC[2][2]); \
  TYPE_MAD( rA_black[2], rB_black[3], rC[2][3]); \
  TYPE_MAD( rA_black[2], rB_black[4], rC[2][4]); \
  TYPE_MAD( rA_black[2], rB_black[5], rC[2][5]); \
  TYPE_MAD( rA_black[2], rB_black[6], rC[2][6]); \
  TYPE_MAD( rA_black[2], rB_black[7], rC[2][7]); \
  \
  TYPE_MAD( rA_black[3], rB_black[0], rC[3][0]); \
  TYPE_MAD( rA_black[3], rB_black[1], rC[3][1]); \
  TYPE_MAD( rA_black[3], rB_black[2], rC[3][2]); \
  TYPE_MAD( rA_black[3], rB_black[3], rC[3][3]); \
  TYPE_MAD( rA_black[3], rB_black[4], rC[3][4]); \
  TYPE_MAD( rA_black[3], rB_black[5], rC[3][5]); \
  TYPE_MAD( rA_black[3], rB_black[6], rC[3][6]); \
  TYPE_MAD( rA_black[3], rB_black[7], rC[3][7]); \
  \
  TYPE_MAD( rA_black[4], rB_black[0], rC[4][0]); \
  TYPE_MAD( rA_black[4], rB_black[1], rC[4][1]); \
  TYPE_MAD( rA_black[4], rB_black[2], rC[4][2]); \
  TYPE_MAD( rA_black[4], rB_black[3], rC[4][3]); \
  TYPE_MAD( rA_black[4], rB_black[4], rC[4][4]); \
  TYPE_MAD( rA_black[4], rB_black[5], rC[4][5]); \
  TYPE_MAD( rA_black[4], rB_black[6], rC[4][6]); \
  TYPE_MAD( rA_black[4], rB_black[7], rC[4][7]); \
  \
  TYPE_MAD( rA_black[5], rB_black[0], rC[5][0]); \
  TYPE_MAD( rA_black[5], rB_black[1], rC[5][1]); \
  TYPE_MAD( rA_black[5], rB_black[2], rC[5][2]); \
  TYPE_MAD( rA_black[5], rB_black[3], rC[5][3]); \
  TYPE_MAD( rA_black[5], rB_black[4], rC[5][4]); \
  TYPE_MAD( rA_black[5], rB_black[5], rC[5][5]); \
  TYPE_MAD( rA_black[5], rB_black[6], rC[5][6]); \
  TYPE_MAD( rA_black[5], rB_black[7], rC[5][7]); \
  \
  TYPE_MAD( rA_black[6], rB_black[0], rC[6][0]); \
  TYPE_MAD( rA_black[6], rB_black[1], rC[6][1]); \
  TYPE_MAD( rA_black[6], rB_black[2], rC[6][2]); \
  TYPE_MAD( rA_black[6], rB_black[3], rC[6][3]); \
  TYPE_MAD( rA_black[6], rB_black[4], rC[6][4]); \
  TYPE_MAD( rA_black[6], rB_black[5], rC[6][5]); \
  TYPE_MAD( rA_black[6], rB_black[6], rC[6][6]); \
  TYPE_MAD( rA_black[6], rB_black[7], rC[6][7]); \
  \
  TYPE_MAD( rA_black[7], rB_black[0], rC[7][0]); \
  TYPE_MAD( rA_black[7], rB_black[1], rC[7][1]); \
  TYPE_MAD( rA_black[7], rB_black[2], rC[7][2]); \
  TYPE_MAD( rA_black[7], rB_black[3], rC[7][3]); \
  TYPE_MAD( rA_black[7], rB_black[4], rC[7][4]); \
  TYPE_MAD( rA_black[7], rB_black[5], rC[7][5]); \
  TYPE_MAD( rA_black[7], rB_black[6], rC[7][6]); \
  TYPE_MAD( rA_black[7], rB_black[7], rC[7][7]); \
  \
  /* mem_fence(CLK_LOCAL_MEM_FENCE); */

#define MICRO_TILE_2_LAST \
  /* prefetch black, compute red */ \
  rA_black[0] = localReadPtrA[offA + 0*WG_0I]; \
  rA_black[1] = localReadPtrA[offA + 1*WG_0I]; \
  rA_black[2] = localReadPtrA[offA + 2*WG_0I]; \
  rA_black[3] = localReadPtrA[offA + 3*WG_0I]; \
  rA_black[4] = localReadPtrA[offA + 4*WG_0I]; \
  rA_black[5] = localReadPtrA[offA + 5*WG_0I]; \
  rA_black[6] = localReadPtrA[offA + 6*WG_0I]; \
  rA_black[7] = localReadPtrA[offA + 7*WG_0I]; \
  \
  rB_black[0] = localReadPtrB[offB + 0*WG_1J]; \
  rB_black[1] = localReadPtrB[offB + 1*WG_1J]; \
  rB_black[2] = localReadPtrB[offB + 2*WG_1J]; \
  rB_black[3] = localReadPtrB[offB + 3*WG_1J]; \
  rB_black[4] = localReadPtrB[offB + 4*WG_1J]; \
  rB_black[5] = localReadPtrB[offB + 5*WG_1J]; \
  rB_black[6] = localReadPtrB[offB + 6*WG_1J]; \
  rB_black[7] = localReadPtrB[offB + 7*WG_1J]; \
  \
  /* mem_fence(CLK_LOCAL_MEM_FENCE); */ \
  \
  offA += (MT_0I+PAD); \
  offB += (MT_1J+PAD); \
  \
  TYPE_MAD( rA_red[0], rB_red[0], rC[0][0]); \
  TYPE_MAD( rA_red[0], rB_red[1], rC[0][1]); \
  TYPE_MAD( rA_red[0], rB_red[2], rC[0][2]); \
  TYPE_MAD( rA_red[0], rB_red[3], rC[0][3]); \
  TYPE_MAD( rA_red[0], rB_red[4], rC[0][4]); \
  TYPE_MAD( rA_red[0], rB_red[5], rC[0][5]); \
  TYPE_MAD( rA_red[0], rB_red[6], rC[0][6]); \
  TYPE_MAD( rA_red[0], rB_red[7], rC[0][7]); \
  \
  TYPE_MAD( rA_red[1], rB_red[0], rC[1][0]); \
  TYPE_MAD( rA_red[1], rB_red[1], rC[1][1]); \
  TYPE_MAD( rA_red[1], rB_red[2], rC[1][2]); \
  TYPE_MAD( rA_red[1], rB_red[3], rC[1][3]); \
  TYPE_MAD( rA_red[1], rB_red[4], rC[1][4]); \
  TYPE_MAD( rA_red[1], rB_red[5], rC[1][5]); \
  TYPE_MAD( rA_red[1], rB_red[6], rC[1][6]); \
  TYPE_MAD( rA_red[1], rB_red[7], rC[1][7]); \
  \
  TYPE_MAD( rA_red[2], rB_red[0], rC[2][0]); \
  TYPE_MAD( rA_red[2], rB_red[1], rC[2][1]); \
  TYPE_MAD( rA_red[2], rB_red[2], rC[2][2]); \
  TYPE_MAD( rA_red[2], rB_red[3], rC[2][3]); \
  TYPE_MAD( rA_red[2], rB_red[4], rC[2][4]); \
  TYPE_MAD( rA_red[2], rB_red[5], rC[2][5]); \
  TYPE_MAD( rA_red[2], rB_red[6], rC[2][6]); \
  TYPE_MAD( rA_red[2], rB_red[7], rC[2][7]); \
  \
  TYPE_MAD( rA_red[3], rB_red[0], rC[3][0]); \
  TYPE_MAD( rA_red[3], rB_red[1], rC[3][1]); \
  TYPE_MAD( rA_red[3], rB_red[2], rC[3][2]); \
  TYPE_MAD( rA_red[3], rB_red[3], rC[3][3]); \
  TYPE_MAD( rA_red[3], rB_red[4], rC[3][4]); \
  TYPE_MAD( rA_red[3], rB_red[5], rC[3][5]); \
  TYPE_MAD( rA_red[3], rB_red[6], rC[3][6]); \
  TYPE_MAD( rA_red[3], rB_red[7], rC[3][7]); \
  \
  TYPE_MAD( rA_red[4], rB_red[0], rC[4][0]); \
  TYPE_MAD( rA_red[4], rB_red[1], rC[4][1]); \
  TYPE_MAD( rA_red[4], rB_red[2], rC[4][2]); \
  TYPE_MAD( rA_red[4], rB_red[3], rC[4][3]); \
  TYPE_MAD( rA_red[4], rB_red[4], rC[4][4]); \
  TYPE_MAD( rA_red[4], rB_red[5], rC[4][5]); \
  TYPE_MAD( rA_red[4], rB_red[6], rC[4][6]); \
  TYPE_MAD( rA_red[4], rB_red[7], rC[4][7]); \
  \
  TYPE_MAD( rA_red[5], rB_red[0], rC[5][0]); \
  TYPE_MAD( rA_red[5], rB_red[1], rC[5][1]); \
  TYPE_MAD( rA_red[5], rB_red[2], rC[5][2]); \
  TYPE_MAD( rA_red[5], rB_red[3], rC[5][3]); \
  TYPE_MAD( rA_red[5], rB_red[4], rC[5][4]); \
  TYPE_MAD( rA_red[5], rB_red[5], rC[5][5]); \
  TYPE_MAD( rA_red[5], rB_red[6], rC[5][6]); \
  TYPE_MAD( rA_red[5], rB_red[7], rC[5][7]); \
  \
  TYPE_MAD( rA_red[6], rB_red[0], rC[6][0]); \
  TYPE_MAD( rA_red[6], rB_red[1], rC[6][1]); \
  TYPE_MAD( rA_red[6], rB_red[2], rC[6][2]); \
  TYPE_MAD( rA_red[6], rB_red[3], rC[6][3]); \
  TYPE_MAD( rA_red[6], rB_red[4], rC[6][4]); \
  TYPE_MAD( rA_red[6], rB_red[5], rC[6][5]); \
  TYPE_MAD( rA_red[6], rB_red[6], rC[6][6]); \
  TYPE_MAD( rA_red[6], rB_red[7], rC[6][7]); \
  \
  TYPE_MAD( rA_red[7], rB_red[0], rC[7][0]); \
  TYPE_MAD( rA_red[7], rB_red[1], rC[7][1]); \
  TYPE_MAD( rA_red[7], rB_red[2], rC[7][2]); \
  TYPE_MAD( rA_red[7], rB_red[3], rC[7][3]); \
  TYPE_MAD( rA_red[7], rB_red[4], rC[7][4]); \
  TYPE_MAD( rA_red[7], rB_red[5], rC[7][5]); \
  TYPE_MAD( rA_red[7], rB_red[6], rC[7][6]); \
  TYPE_MAD( rA_red[7], rB_red[7], rC[7][7]); \
  \
  /* mem_fence(CLK_LOCAL_MEM_FENCE); */ \
  /* don't prefetch red, compute black */ \
  \
  TYPE_MAD( rA_black[0], rB_black[0], rC[0][0]); \
  TYPE_MAD( rA_black[0], rB_black[1], rC[0][1]); \
  TYPE_MAD( rA_black[0], rB_black[2], rC[0][2]); \
  TYPE_MAD( rA_black[0], rB_black[3], rC[0][3]); \
  TYPE_MAD( rA_black[0], rB_black[4], rC[0][4]); \
  TYPE_MAD( rA_black[0], rB_black[5], rC[0][5]); \
  TYPE_MAD( rA_black[0], rB_black[6], rC[0][6]); \
  TYPE_MAD( rA_black[0], rB_black[7], rC[0][7]); \
  \
  TYPE_MAD( rA_black[1], rB_black[0], rC[1][0]); \
  TYPE_MAD( rA_black[1], rB_black[1], rC[1][1]); \
  TYPE_MAD( rA_black[1], rB_black[2], rC[1][2]); \
  TYPE_MAD( rA_black[1], rB_black[3], rC[1][3]); \
  TYPE_MAD( rA_black[1], rB_black[4], rC[1][4]); \
  TYPE_MAD( rA_black[1], rB_black[5], rC[1][5]); \
  TYPE_MAD( rA_black[1], rB_black[6], rC[1][6]); \
  TYPE_MAD( rA_black[1], rB_black[7], rC[1][7]); \
  \
  TYPE_MAD( rA_black[2], rB_black[0], rC[2][0]); \
  TYPE_MAD( rA_black[2], rB_black[1], rC[2][1]); \
  TYPE_MAD( rA_black[2], rB_black[2], rC[2][2]); \
  TYPE_MAD( rA_black[2], rB_black[3], rC[2][3]); \
  TYPE_MAD( rA_black[2], rB_black[4], rC[2][4]); \
  TYPE_MAD( rA_black[2], rB_black[5], rC[2][5]); \
  TYPE_MAD( rA_black[2], rB_black[6], rC[2][6]); \
  TYPE_MAD( rA_black[2], rB_black[7], rC[2][7]); \
  \
  TYPE_MAD( rA_black[3], rB_black[0], rC[3][0]); \
  TYPE_MAD( rA_black[3], rB_black[1], rC[3][1]); \
  TYPE_MAD( rA_black[3], rB_black[2], rC[3][2]); \
  TYPE_MAD( rA_black[3], rB_black[3], rC[3][3]); \
  TYPE_MAD( rA_black[3], rB_black[4], rC[3][4]); \
  TYPE_MAD( rA_black[3], rB_black[5], rC[3][5]); \
  TYPE_MAD( rA_black[3], rB_black[6], rC[3][6]); \
  TYPE_MAD( rA_black[3], rB_black[7], rC[3][7]); \
  \
  TYPE_MAD( rA_black[4], rB_black[0], rC[4][0]); \
  TYPE_MAD( rA_black[4], rB_black[1], rC[4][1]); \
  TYPE_MAD( rA_black[4], rB_black[2], rC[4][2]); \
  TYPE_MAD( rA_black[4], rB_black[3], rC[4][3]); \
  TYPE_MAD( rA_black[4], rB_black[4], rC[4][4]); \
  TYPE_MAD( rA_black[4], rB_black[5], rC[4][5]); \
  TYPE_MAD( rA_black[4], rB_black[6], rC[4][6]); \
  TYPE_MAD( rA_black[4], rB_black[7], rC[4][7]); \
  \
  TYPE_MAD( rA_black[5], rB_black[0], rC[5][0]); \
  TYPE_MAD( rA_black[5], rB_black[1], rC[5][1]); \
  TYPE_MAD( rA_black[5], rB_black[2], rC[5][2]); \
  TYPE_MAD( rA_black[5], rB_black[3], rC[5][3]); \
  TYPE_MAD( rA_black[5], rB_black[4], rC[5][4]); \
  TYPE_MAD( rA_black[5], rB_black[5], rC[5][5]); \
  TYPE_MAD( rA_black[5], rB_black[6], rC[5][6]); \
  TYPE_MAD( rA_black[5], rB_black[7], rC[5][7]); \
  \
  TYPE_MAD( rA_black[6], rB_black[0], rC[6][0]); \
  TYPE_MAD( rA_black[6], rB_black[1], rC[6][1]); \
  TYPE_MAD( rA_black[6], rB_black[2], rC[6][2]); \
  TYPE_MAD( rA_black[6], rB_black[3], rC[6][3]); \
  TYPE_MAD( rA_black[6], rB_black[4], rC[6][4]); \
  TYPE_MAD( rA_black[6], rB_black[5], rC[6][5]); \
  TYPE_MAD( rA_black[6], rB_black[6], rC[6][6]); \
  TYPE_MAD( rA_black[6], rB_black[7], rC[6][7]); \
  \
  TYPE_MAD( rA_black[7], rB_black[0], rC[7][0]); \
  TYPE_MAD( rA_black[7], rB_black[1], rC[7][1]); \
  TYPE_MAD( rA_black[7], rB_black[2], rC[7][2]); \
  TYPE_MAD( rA_black[7], rB_black[3], rC[7][3]); \
  TYPE_MAD( rA_black[7], rB_black[4], rC[7][4]); \
  TYPE_MAD( rA_black[7], rB_black[5], rC[7][5]); \
  TYPE_MAD( rA_black[7], rB_black[6], rC[7][6]); \
  TYPE_MAD( rA_black[7], rB_black[7], rC[7][7]); \
  \
  /* mem_fence(CLK_LOCAL_MEM_FENCE); */

/* preprocessor definitions of kernel arguments*/
#define strideC0I 1
#define strideA0I 1
#define strideB1J 1


/* kernel */
extern "C"
__global__ void kernel_hip(
  hipLaunchParm lp,
  float       *          C,
  float const * __restrict__ A,
  float const * __restrict__ B,
  float const alpha,
  float const beta,
  unsigned int const strideC1J,
  unsigned int const strideAK,
  unsigned int const strideBK,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK ) {

  /* allocate registers */
  TYPE_C rC[UT_0I][UT_1J] = {{0}};
  TYPE_A rA_red[UT_0I];
  TYPE_B rB_red[UT_1J];
  TYPE_A rA_black[UT_0I];
  TYPE_B rB_black[UT_1J];

  /* allocate local memory */
  __shared__ TYPE_A localBasePtrA[2*UNROLL*MT_0I_2];
  __shared__ TYPE_B localBasePtrB[2*UNROLL*MT_1J_2];


  /* c indices (group) */
  unsigned int g0I = hc_get_group_id(0); // d0, tensorA
  unsigned int g1J = hc_get_group_id(1); // d1, tensorB

  /* c indices (local) */
  unsigned int l0I = hc_get_workitem_id(0); // d0
  unsigned int l1J = hc_get_workitem_id(1); // d1
  unsigned int loadSerial = l0I + l1J*WG_0I;
  unsigned int a0I = loadSerial%LS_PARA_A;
  unsigned int b1J = loadSerial%LS_PARA_B;

  /* unrolled summation index */
  unsigned int aK = loadSerial/LS_PARA_A;
  unsigned int bK = loadSerial/LS_PARA_B;

  /* where will this thread read from global memory */
  A += GLOBAL_A( a0I+g0I*MT_0I, aK );
  B += GLOBAL_B( b1J+g1J*MT_1J, bK );

  /* where will this thread's micro-tile read from local memory */
  unsigned int localReadOffsetA = UNROLL*MT_0I_2;
  unsigned int localReadOffsetB = UNROLL*MT_1J_2;
  TYPE_A *localReadPtrA  = localBasePtrA + localReadOffsetA;
  TYPE_B *localReadPtrB  = localBasePtrB + localReadOffsetB;

  /* where will this thread write to local memory */
  unsigned int localWriteOffsetA = a0I + aK*(MT_0I+PAD);
  unsigned int localWriteOffsetB = b1J + bK*(MT_1J+PAD);
  TYPE_A *localWritePtrA = localBasePtrA + localWriteOffsetA;
  TYPE_B *localWritePtrB = localBasePtrB + localWriteOffsetB;

  /* 0th load A global -> local */
  localWritePtrA[ 0*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = A[ 0*LS_PARA_A + 0*LS_PERP_A*strideAK];
  localWritePtrA[ 1*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = A[ 1*LS_PARA_A + 0*LS_PERP_A*strideAK];
  localWritePtrA[ 2*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = A[ 2*LS_PARA_A + 0*LS_PERP_A*strideAK];
  localWritePtrA[ 3*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = A[ 3*LS_PARA_A + 0*LS_PERP_A*strideAK];

  /* 0th load B global -> local */
  localWritePtrB[ 0*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = B[ 0*LS_PARA_B + 0*LS_PERP_B*strideBK];
  localWritePtrB[ 1*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = B[ 1*LS_PARA_B + 0*LS_PERP_B*strideBK];
  localWritePtrB[ 2*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = B[ 2*LS_PARA_B + 0*LS_PERP_B*strideBK];
  localWritePtrB[ 3*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = B[ 3*LS_PARA_B + 0*LS_PERP_B*strideBK];
  __syncthreads();

  /* iterate over summation indice(s) except last */
  unsigned int sumIterK = sizeK / UNROLL;
  do {

    /* swap local read offset and update pointers */
    localReadOffsetA ^= UNROLL*MT_0I_2;
    localReadOffsetB ^= UNROLL*MT_1J_2;
    localReadPtrA = localBasePtrA + localReadOffsetA;
    localReadPtrB = localBasePtrB + localReadOffsetB;

    /* swap local write offset and update pointers */
    localWriteOffsetA ^= UNROLL*MT_0I_2;
    localWriteOffsetB ^= UNROLL*MT_1J_2;
    localWritePtrA = localBasePtrA + localWriteOffsetA;
    localWritePtrB = localBasePtrB + localWriteOffsetB;

    /* incr global read */
    A += (long) strideAK*UNROLL;
    B += (long) strideBK*UNROLL;

    /* load A global -> local */
    localWritePtrA[ 0*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = A[ 0*LS_PARA_A + 0*LS_PERP_A*strideAK];
    localWritePtrA[ 1*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = A[ 1*LS_PARA_A + 0*LS_PERP_A*strideAK];
    localWritePtrA[ 2*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = A[ 2*LS_PARA_A + 0*LS_PERP_A*strideAK];
    localWritePtrA[ 3*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = A[ 3*LS_PARA_A + 0*LS_PERP_A*strideAK];

    /* load B global -> local */
    localWritePtrB[ 0*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = B[ 0*LS_PARA_B + 0*LS_PERP_B*strideBK];
    localWritePtrB[ 1*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = B[ 1*LS_PARA_B + 0*LS_PERP_B*strideBK];
    localWritePtrB[ 2*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = B[ 2*LS_PARA_B + 0*LS_PERP_B*strideBK];
    localWritePtrB[ 3*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = B[ 3*LS_PARA_B + 0*LS_PERP_B*strideBK];
    __syncthreads();

    /* unroll offsets */
    unsigned int offA = l0I; // d0
    unsigned int offB = l1J; // d1

    /* do fmas */
    MICRO_TILE_PREFETCH
    MICRO_TILE_2
    MICRO_TILE_2
    MICRO_TILE_2
    MICRO_TILE_2_LAST

    __syncthreads();

  } while (--sumIterK > 1);

  /* do last iteration without loading from global */

  /* swap local read offset and update pointers */
  localReadOffsetA ^= UNROLL*MT_0I_2;
  localReadOffsetB ^= UNROLL*MT_1J_2;
  localReadPtrA = localBasePtrA + localReadOffsetA;
  localReadPtrB = localBasePtrB + localReadOffsetB;

  /* do fmas */
  unsigned int offA = l0I; // d0
  unsigned int offB = l1J; // d1
  MICRO_TILE_PREFETCH
  MICRO_TILE_2
  MICRO_TILE_2
  MICRO_TILE_2
  MICRO_TILE_2_LAST

  /* which global Cij index */
  unsigned int globalC1J = g1J*MT_1J + l1J;
  unsigned int globalC0I = g0I*MT_0I + l0I;

  /* write global C */
  //TYPE_MAD_WRITE( C[ GLOBAL_C( 0, 0) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 6*WG_1J) ], alpha, rC[0][6], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 7*WG_1J) ], alpha, rC[0][7], beta)

  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 6*WG_1J) ], alpha, rC[1][6], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 7*WG_1J) ], alpha, rC[1][7], beta)

  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 6*WG_1J) ], alpha, rC[2][6], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 7*WG_1J) ], alpha, rC[2][7], beta)

  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 6*WG_1J) ], alpha, rC[3][6], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 7*WG_1J) ], alpha, rC[3][7], beta)

  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 4*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 4*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 4*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 4*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 4*WG_0I, globalC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 4*WG_0I, globalC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 4*WG_0I, globalC1J + 6*WG_1J) ], alpha, rC[4][6], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 4*WG_0I, globalC1J + 7*WG_1J) ], alpha, rC[4][7], beta)

  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 5*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 5*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 5*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 5*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 5*WG_0I, globalC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 5*WG_0I, globalC1J + 5*WG_1J) ], alpha, rC[5][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 5*WG_0I, globalC1J + 6*WG_1J) ], alpha, rC[5][6], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 5*WG_0I, globalC1J + 7*WG_1J) ], alpha, rC[5][7], beta)

  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 6*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[6][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 6*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[6][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 6*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[6][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 6*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[6][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 6*WG_0I, globalC1J + 4*WG_1J) ], alpha, rC[6][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 6*WG_0I, globalC1J + 5*WG_1J) ], alpha, rC[6][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 6*WG_0I, globalC1J + 6*WG_1J) ], alpha, rC[6][6], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 6*WG_0I, globalC1J + 7*WG_1J) ], alpha, rC[6][7], beta)

  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 7*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[7][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 7*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[7][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 7*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[7][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 7*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[7][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 7*WG_0I, globalC1J + 4*WG_1J) ], alpha, rC[7][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 7*WG_0I, globalC1J + 5*WG_1J) ], alpha, rC[7][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 7*WG_0I, globalC1J + 6*WG_1J) ], alpha, rC[7][6], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 7*WG_0I, globalC1J + 7*WG_1J) ], alpha, rC[7][7], beta)

};


#endif
