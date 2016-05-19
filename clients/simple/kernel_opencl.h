const char * kernelSource = R"(

/* CT_SSSSS_Cij_Sk_Aik_Bjk_i16x6f_j16x6f_k16_O2 */

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   16


/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* global tile indices being loaded */
/* fast read */
#define globalIdxA0I(LID) (groupIdx0I*MACRO_TILE_0I + (localSerial+(LID)*WG_DIM_0I*WG_DIM_1J)/NUM_UNROLL_ITER)
#define globalIdxAK(LID) (localSerial%NUM_UNROLL_ITER)
/* fast read */
#define globalIdxBK(LID) ((localSerial+(LID)*WG_DIM_0I*WG_DIM_1J)/MACRO_TILE_1J)
#define globalIdxB1J(LID) (groupIdx1J*MACRO_TILE_1J + (localSerial+(LID)*WG_DIM_0I*WG_DIM_1J)%MACRO_TILE_1J)

/* global non-tile indices being loaded */


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J) )

/* local indices being written */
#define localA0I (localSerial / NUM_UNROLL_ITER)
#define localAK (localSerial % NUM_UNROLL_ITER)
#define localAStride (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER)
#define localB1J ( localSerial / MACRO_TILE_1J )
#define localBK ( localSerial % MACRO_TILE_1J )
#define localBStride  (WG_DIM_0I*WG_DIM_1J)

/* data types */
#define DATA_TYPE_STR_A float
#define DATA_TYPE_STR_B float
#define DATA_TYPE_STR_C float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA float
#define FMA(A,B,DST) mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 6x6 micro-tile */
#define MICRO_TILE \
  rA[0] = localA[offA + 0*WG_DIM_0I]; \
  rA[1] = localA[offA + 1*WG_DIM_0I]; \
  rA[2] = localA[offA + 2*WG_DIM_0I]; \
  rA[3] = localA[offA + 3*WG_DIM_0I]; \
  rA[4] = localA[offA + 4*WG_DIM_0I]; \
  rA[5] = localA[offA + 5*WG_DIM_0I]; \
  rB[0] = localB[offB + 0*WG_DIM_1J]; \
  rB[1] = localB[offB + 1*WG_DIM_1J]; \
  rB[2] = localB[offB + 2*WG_DIM_1J]; \
  rB[3] = localB[offB + 3*WG_DIM_1J]; \
  rB[4] = localB[offB + 4*WG_DIM_1J]; \
  rB[5] = localB[offB + 5*WG_DIM_1J]; \
  offA += MACRO_TILE_0I; \
  offB += MACRO_TILE_1J; \
  TYPE_MAD(rA[0],rB[0],rC[0][0]); \
  TYPE_MAD(rA[0],rB[1],rC[0][1]); \
  TYPE_MAD(rA[0],rB[2],rC[0][2]); \
  TYPE_MAD(rA[0],rB[3],rC[0][3]); \
  TYPE_MAD(rA[0],rB[4],rC[0][4]); \
  TYPE_MAD(rA[0],rB[5],rC[0][5]); \
  TYPE_MAD(rA[1],rB[0],rC[1][0]); \
  TYPE_MAD(rA[1],rB[1],rC[1][1]); \
  TYPE_MAD(rA[1],rB[2],rC[1][2]); \
  TYPE_MAD(rA[1],rB[3],rC[1][3]); \
  TYPE_MAD(rA[1],rB[4],rC[1][4]); \
  TYPE_MAD(rA[1],rB[5],rC[1][5]); \
  TYPE_MAD(rA[2],rB[0],rC[2][0]); \
  TYPE_MAD(rA[2],rB[1],rC[2][1]); \
  TYPE_MAD(rA[2],rB[2],rC[2][2]); \
  TYPE_MAD(rA[2],rB[3],rC[2][3]); \
  TYPE_MAD(rA[2],rB[4],rC[2][4]); \
  TYPE_MAD(rA[2],rB[5],rC[2][5]); \
  TYPE_MAD(rA[3],rB[0],rC[3][0]); \
  TYPE_MAD(rA[3],rB[1],rC[3][1]); \
  TYPE_MAD(rA[3],rB[2],rC[3][2]); \
  TYPE_MAD(rA[3],rB[3],rC[3][3]); \
  TYPE_MAD(rA[3],rB[4],rC[3][4]); \
  TYPE_MAD(rA[3],rB[5],rC[3][5]); \
  TYPE_MAD(rA[4],rB[0],rC[4][0]); \
  TYPE_MAD(rA[4],rB[1],rC[4][1]); \
  TYPE_MAD(rA[4],rB[2],rC[4][2]); \
  TYPE_MAD(rA[4],rB[3],rC[4][3]); \
  TYPE_MAD(rA[4],rB[4],rC[4][4]); \
  TYPE_MAD(rA[4],rB[5],rC[4][5]); \
  TYPE_MAD(rA[5],rB[0],rC[5][0]); \
  TYPE_MAD(rA[5],rB[1],rC[5][1]); \
  TYPE_MAD(rA[5],rB[2],rC[5][2]); \
  TYPE_MAD(rA[5],rB[3],rC[5][3]); \
  TYPE_MAD(rA[5],rB[4],rC[5][4]); \
  TYPE_MAD(rA[5],rB[5],rC[5][5]); \
  mem_fence(CLK_LOCAL_MEM_FENCE);

/* preprocessor definitions of kernel arguments*/
#define strideC0I 1
#define strideA0I 1
#define strideB1J 1


__attribute__((reqd_work_group_size(WG_DIM_0I,WG_DIM_1J,1)))
__kernel void gemm_kernel(
  __global float       *          C,
  __global float const * restrict A,
  __global float const * restrict B,
  float const alpha,
  float const beta,
  unsigned int const strideC1J,
  unsigned int const strideAK,
  unsigned int const strideBK,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK ) {

  /* allocate registers */
  DATA_TYPE_STR_C rC[MICRO_TILE_0I][MICRO_TILE_1J] = {{0}};
  DATA_TYPE_STR_A rA[MICRO_TILE_0I];
  DATA_TYPE_STR_B rB[MICRO_TILE_1J];

  /* allocate local memory */
  __local DATA_TYPE_STR_A localA[NUM_UNROLL_ITER*MACRO_TILE_0I];
  __local DATA_TYPE_STR_B localB[NUM_UNROLL_ITER*MACRO_TILE_1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MACRO_TILE_1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MACRO_TILE_0I + localIdx0I;
  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    __local DATA_TYPE_STR_A *lA = localA + GET_LOCAL_INDEX_A(localA0I, localAK);
    __local DATA_TYPE_STR_B *lB = localB + GET_LOCAL_INDEX_B(localB1J, localBK);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */
    lA[ 0*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(0), globalIdxAK(0) ) ];
    lA[ 1*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(1), globalIdxAK(1) ) ];
    lA[ 2*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(2), globalIdxAK(2) ) ];
    lA[ 3*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(3), globalIdxAK(3) ) ];
    lA[ 4*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(4), globalIdxAK(4) ) ];
    lA[ 5*localAStride ] = A[ GET_GLOBAL_INDEX_A( globalIdxA0I(5), globalIdxAK(5) ) ];
    lB[ 0*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(0), globalIdxBK(0) ) ];
    lB[ 1*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(1), globalIdxBK(1) ) ];
    lB[ 2*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(2), globalIdxBK(2) ) ];
    lB[ 3*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(3), globalIdxBK(3) ) ];
    lB[ 4*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(4), globalIdxBK(4) ) ];
    lB[ 5*localBStride ] = B[ GET_GLOBAL_INDEX_B( globalIdxB1J(5), globalIdxBK(5) ) ];
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int offA = localIdx0I; // d0
    unsigned int offB = localIdx1J; // d1

    /* do mads */
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    A += strideAK*NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );
  //printf("%u, %u, %f\n", get_global_id(0), get_global_id(1), rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[5][5], beta)

};
)";
