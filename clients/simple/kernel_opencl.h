

// 6x6 micro tile
// float4 - can't get to work
// unroll 16
#if 0
const char * kernelSource = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   16
#define LOAD_WIDTH        4

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
  __global float4 const * restrict A,
  __global float4 const * restrict B,
  float const alpha,
  float const beta,
  unsigned int const strideC1J,
  unsigned int const strideAK,
  unsigned int const strideBK,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK ) {

  /* allocate registers */
  float rC[MICRO_TILE_0I][MICRO_TILE_1J] = {{0}};
  float rA[MICRO_TILE_0I];
  float rB[MICRO_TILE_1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*MACRO_TILE_0I];
  __local float localB[NUM_UNROLL_ITER*MACRO_TILE_1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;
  unsigned int subGroupA = localIdx1J > WG_DIM_1J/2;
  unsigned int subGroupSerial = localIdx0I + (localIdx1J%(WG_DIM_1J/2))*WG_DIM_0I;
  //unsigned int localSerial = localIdx0I*WG_DIM_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = subGroupSerial%(WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER);
  unsigned int aK = subGroupSerial/(WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER);
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;


  A +=  aI+groupIdx0I*MACRO_TILE_0I/LOAD_WIDTH + aK*strideAK/LOAD_WIDTH;
  B +=  bJ+groupIdx1J*MACRO_TILE_1J/LOAD_WIDTH + bK*strideBK/LOAD_WIDTH;
  __global float4 *globalPtr = (__global float4 *) (subGroupA ? A : B);

  __local  float4 *localPtr  = (__local float4 *)  (subGroupA ?  localA : localB);
  localPtr += GET_LOCAL_INDEX_A(aI, aK)/LOAD_WIDTH;

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);


    /* load global -> local */
#if 0
    localPtr[0*8] = globalPtr[0*8];
    localPtr[1*8] = globalPtr[1*8];
    localPtr[2*8] = globalPtr[2*8];
#else
    localPtr[0*8] = (float4){1.f, 1.f, 1.f, 1.f};
    localPtr[1*8] = (float4){1.f, 1.f, 1.f, 1.f};
    localPtr[2*8] = (float4){1.f, 1.f, 1.f, 1.f};
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    printf("local[%u]=%f %f\n", localSerial, localA[localSerial], localB[localSerial]);

    if (localSerial == 0) {
      for (unsigned int i = 0; i < NUM_UNROLL_ITER*MACRO_TILE_0I; i++) {
        printf("localA[%u]=%f\n", i, localA[i]);
      }
      for (unsigned int i = 0; i < NUM_UNROLL_ITER*MACRO_TILE_0I; i++) {
        printf("localB[%u]=%f\n", i, localB[i]);
      }
    }
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

    A += strideAK*NUM_UNROLL_ITER/LOAD_WIDTH;
    B += strideBK*NUM_UNROLL_ITER/LOAD_WIDTH;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );
  //printf("%u, %u, %f\n", get_global_id(0), get_global_id(1), rC[0][0] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MACRO_TILE_1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MACRO_TILE_0I + localIdx0I;

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
#endif




// 4x4 micro tile
// float2
#if 0
const char * kernelSource = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     4
#define MICRO_TILE_1J     4
#define MACRO_TILE_0I     64
#define MACRO_TILE_1J     64
#define NUM_UNROLL_ITER   8


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
  rB[0] = localB[offB + 0*WG_DIM_1J]; \
  rB[1] = localB[offB + 1*WG_DIM_1J]; \
  rB[2] = localB[offB + 2*WG_DIM_1J]; \
  rB[3] = localB[offB + 3*WG_DIM_1J]; \
  offA += MACRO_TILE_0I; \
  offB += MACRO_TILE_1J; \
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
  mem_fence(CLK_LOCAL_MEM_FENCE);

/* preprocessor definitions of kernel arguments*/
#define strideC0I 1
#define strideA0I 1
#define strideB1J 1


__attribute__((reqd_work_group_size(WG_DIM_0I,WG_DIM_1J,1)))
__kernel void gemm_kernel(
  __global float       *          C,
  __global float2 const * restrict A,
  __global float2 const * restrict B,
  float const alpha,
  float const beta,
  unsigned int const strideC1J,
  unsigned int const strideAK,
  unsigned int const strideBK,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK ) {

  /* allocate registers */
  float rC[MICRO_TILE_0I][MICRO_TILE_1J] = {{0}};
  float rA[MICRO_TILE_0I];
  float rB[MICRO_TILE_1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*MACRO_TILE_0I];
  __local float localB[NUM_UNROLL_ITER*MACRO_TILE_1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;
  //unsigned int localSerial = localIdx0I*WG_DIM_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%(WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER);
  unsigned int aK = localSerial/(WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER);
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);

  __local float2 *lA = (__local float2 *) (localA + GET_LOCAL_INDEX_A(aI, aK)/2);
  __local float2 *lB = (__local float2 *) (localB + GET_LOCAL_INDEX_B(bJ, bK)/2);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);


    /* load global -> local */
    lA[ 0] = A[ 0];
    lA[16] = A[16];

    lB[ 0] = B[ 0];
    lB[16] = B[16];

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

    A += strideAK*NUM_UNROLL_ITER/2;
    B += strideBK*NUM_UNROLL_ITER/2;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );
  //printf("%u, %u, %f\n", get_global_id(0), get_global_id(1), rC[0][0] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MACRO_TILE_1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MACRO_TILE_0I + localIdx0I;

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[0][3], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[1][3], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[2][3], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[3][3], beta)

};
)";
#endif


// 6x6 micro tile
// float2
#if 0
const char * kernelSource = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   8


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
  __global float2 const * restrict A,
  __global float2 const * restrict B,
  float const alpha,
  float const beta,
  unsigned int const strideC1J,
  unsigned int const strideAK,
  unsigned int const strideBK,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK ) {

  /* allocate registers */
  float rC[MICRO_TILE_0I][MICRO_TILE_1J] = {{0}};
  float rA[MICRO_TILE_0I];
  float rB[MICRO_TILE_1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*MACRO_TILE_0I];
  __local float localB[NUM_UNROLL_ITER*MACRO_TILE_1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;
  //unsigned int localSerial = localIdx0I*WG_DIM_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%(WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER);
  unsigned int aK = localSerial/(WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER);
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);

  __local float2 *lA = (__local float2 *) (localA + GET_LOCAL_INDEX_A(aI, aK)/2);
  __local float2 *lB = (__local float2 *) (localB + GET_LOCAL_INDEX_B(bJ, bK)/2);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);


    /* load global -> local */
    lA[ 0] = A[ 0];
    lA[16] = A[16];
    lA[32] = A[32];
    //lA[48] = A[48];
    //lA[64] = A[64];
    //lA[80] = A[80];

    lB[ 0] = B[ 0];
    lB[16] = B[16];
    lB[32] = B[32];
    //lB[48] = B[48];
    //lB[64] = B[64];
    //lB[80] = B[80];

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

    A += strideAK*NUM_UNROLL_ITER/2;
    B += strideBK*NUM_UNROLL_ITER/2;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );
  //printf("%u, %u, %f\n", get_global_id(0), get_global_id(1), rC[0][0] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MACRO_TILE_1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MACRO_TILE_0I + localIdx0I;

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
#endif



// 8x8 micro tile
#if 0
const char * kernelSource = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     8
#define MICRO_TILE_1J     8
#define MACRO_TILE_0I     128
#define MACRO_TILE_1J     128
#define NUM_UNROLL_ITER   8


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
  rA[6] = localA[offA + 6*WG_DIM_0I]; \
  rA[7] = localA[offA + 7*WG_DIM_0I]; \
  rB[0] = localB[offB + 0*WG_DIM_1J]; \
  rB[1] = localB[offB + 1*WG_DIM_1J]; \
  rB[2] = localB[offB + 2*WG_DIM_1J]; \
  rB[3] = localB[offB + 3*WG_DIM_1J]; \
  rB[4] = localB[offB + 4*WG_DIM_1J]; \
  rB[5] = localB[offB + 5*WG_DIM_1J]; \
  rB[6] = localB[offB + 6*WG_DIM_1J]; \
  rB[7] = localB[offB + 7*WG_DIM_1J]; \
  offA += MACRO_TILE_0I; \
  offB += MACRO_TILE_1J; \
  TYPE_MAD(rA[0],rB[0],rC[0][0]); \
  TYPE_MAD(rA[0],rB[1],rC[0][1]); \
  TYPE_MAD(rA[0],rB[2],rC[0][2]); \
  TYPE_MAD(rA[0],rB[3],rC[0][3]); \
  TYPE_MAD(rA[0],rB[4],rC[0][4]); \
  TYPE_MAD(rA[0],rB[5],rC[0][5]); \
  TYPE_MAD(rA[0],rB[6],rC[0][6]); \
  TYPE_MAD(rA[0],rB[7],rC[0][7]); \
\
  TYPE_MAD(rA[1],rB[0],rC[1][0]); \
  TYPE_MAD(rA[1],rB[1],rC[1][1]); \
  TYPE_MAD(rA[1],rB[2],rC[1][2]); \
  TYPE_MAD(rA[1],rB[3],rC[1][3]); \
  TYPE_MAD(rA[1],rB[4],rC[1][4]); \
  TYPE_MAD(rA[1],rB[5],rC[1][5]); \
  TYPE_MAD(rA[1],rB[6],rC[1][6]); \
  TYPE_MAD(rA[1],rB[7],rC[1][7]); \
\
  TYPE_MAD(rA[2],rB[0],rC[2][0]); \
  TYPE_MAD(rA[2],rB[1],rC[2][1]); \
  TYPE_MAD(rA[2],rB[2],rC[2][2]); \
  TYPE_MAD(rA[2],rB[3],rC[2][3]); \
  TYPE_MAD(rA[2],rB[4],rC[2][4]); \
  TYPE_MAD(rA[2],rB[5],rC[2][5]); \
  TYPE_MAD(rA[2],rB[6],rC[2][6]); \
  TYPE_MAD(rA[2],rB[7],rC[2][7]); \
\
  TYPE_MAD(rA[3],rB[0],rC[3][0]); \
  TYPE_MAD(rA[3],rB[1],rC[3][1]); \
  TYPE_MAD(rA[3],rB[2],rC[3][2]); \
  TYPE_MAD(rA[3],rB[3],rC[3][3]); \
  TYPE_MAD(rA[3],rB[4],rC[3][4]); \
  TYPE_MAD(rA[3],rB[5],rC[3][5]); \
  TYPE_MAD(rA[3],rB[6],rC[3][6]); \
  TYPE_MAD(rA[3],rB[7],rC[3][7]); \
\
  TYPE_MAD(rA[4],rB[0],rC[4][0]); \
  TYPE_MAD(rA[4],rB[1],rC[4][1]); \
  TYPE_MAD(rA[4],rB[2],rC[4][2]); \
  TYPE_MAD(rA[4],rB[3],rC[4][3]); \
  TYPE_MAD(rA[4],rB[4],rC[4][4]); \
  TYPE_MAD(rA[4],rB[5],rC[4][5]); \
  TYPE_MAD(rA[4],rB[6],rC[4][6]); \
  TYPE_MAD(rA[4],rB[7],rC[4][7]); \
\
  TYPE_MAD(rA[5],rB[0],rC[5][0]); \
  TYPE_MAD(rA[5],rB[1],rC[5][1]); \
  TYPE_MAD(rA[5],rB[2],rC[5][2]); \
  TYPE_MAD(rA[5],rB[3],rC[5][3]); \
  TYPE_MAD(rA[5],rB[4],rC[5][4]); \
  TYPE_MAD(rA[5],rB[5],rC[5][5]); \
  TYPE_MAD(rA[5],rB[6],rC[5][6]); \
  TYPE_MAD(rA[5],rB[7],rC[5][7]); \
\
  TYPE_MAD(rA[6],rB[0],rC[6][0]); \
  TYPE_MAD(rA[6],rB[1],rC[6][1]); \
  TYPE_MAD(rA[6],rB[2],rC[6][2]); \
  TYPE_MAD(rA[6],rB[3],rC[6][3]); \
  TYPE_MAD(rA[6],rB[4],rC[6][4]); \
  TYPE_MAD(rA[6],rB[5],rC[6][5]); \
  TYPE_MAD(rA[6],rB[6],rC[6][6]); \
  TYPE_MAD(rA[6],rB[7],rC[6][7]); \
\
  TYPE_MAD(rA[7],rB[0],rC[7][0]); \
  TYPE_MAD(rA[7],rB[1],rC[7][1]); \
  TYPE_MAD(rA[7],rB[2],rC[7][2]); \
  TYPE_MAD(rA[7],rB[3],rC[7][3]); \
  TYPE_MAD(rA[7],rB[4],rC[7][4]); \
  TYPE_MAD(rA[7],rB[5],rC[7][5]); \
  TYPE_MAD(rA[7],rB[6],rC[7][6]); \
  TYPE_MAD(rA[7],rB[7],rC[7][7]); \
\
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
  float rC[MICRO_TILE_0I][MICRO_TILE_1J] = {{0}};
  float rA[MICRO_TILE_0I];
  float rB[MICRO_TILE_1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*MACRO_TILE_0I];
  __local float localB[NUM_UNROLL_ITER*MACRO_TILE_1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;
  //unsigned int localSerial = localIdx0I*WG_DIM_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%(WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER);
  unsigned int aK = localSerial/(WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER);
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);

    __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
    __local float *lB = localB + GET_LOCAL_INDEX_B(bJ, bK);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);


    /* load global -> local */
    lA[ 0] = A[ 0];
    lA[16] = A[16];
    lA[32] = A[32];
    lA[48] = A[48];
#if NUM_UNROLL_ITER>8
    lA[64] = A[64];
    lA[80] = A[80];
    lA[96] = A[96];
    lA[112] = A[112];
#endif
    lB[ 0] = B[ 0];
    lB[16] = B[16];
    lB[32] = B[32];
    lB[48] = B[48];
#if NUM_UNROLL_ITER>8
    lB[64] = B[64];
    lB[80] = B[80];
    lB[96] = B[96];
    lB[112] = B[112];
#endif

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
#if NUM_UNROLL_ITER>8
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
#endif

    A += strideAK*NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );
  //printf("%u, %u, %f\n", get_global_id(0), get_global_id(1), rC[0][0] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MACRO_TILE_1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MACRO_TILE_0I + localIdx0I;

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 6*WG_DIM_1J) ], alpha, rC[0][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 7*WG_DIM_1J) ], alpha, rC[0][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 6*WG_DIM_1J) ], alpha, rC[1][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 7*WG_DIM_1J) ], alpha, rC[1][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 6*WG_DIM_1J) ], alpha, rC[2][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 7*WG_DIM_1J) ], alpha, rC[2][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 6*WG_DIM_1J) ], alpha, rC[3][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 7*WG_DIM_1J) ], alpha, rC[3][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 6*WG_DIM_1J) ], alpha, rC[4][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 7*WG_DIM_1J) ], alpha, rC[4][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[5][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 6*WG_DIM_1J) ], alpha, rC[5][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 7*WG_DIM_1J) ], alpha, rC[5][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[6][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[6][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[6][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[6][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[6][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[6][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_DIM_0I, globalIdxC1J + 6*WG_DIM_1J) ], alpha, rC[6][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_DIM_0I, globalIdxC1J + 7*WG_DIM_1J) ], alpha, rC[6][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[7][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[7][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[7][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[7][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[7][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[7][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_DIM_0I, globalIdxC1J + 6*WG_DIM_1J) ], alpha, rC[7][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_DIM_0I, globalIdxC1J + 7*WG_DIM_1J) ], alpha, rC[7][7], beta)

};
)";
#endif


// 6x6 micro tile
// unroll 8
#if 0
const char * kernelSource = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   8
#define TPI (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J) )

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
  float rC[MICRO_TILE_0I][MICRO_TILE_1J] = {{0}};
  float rA[MICRO_TILE_0I];
  float rB[MICRO_TILE_1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*MACRO_TILE_0I];
  __local float localB[NUM_UNROLL_ITER*MACRO_TILE_1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;
  //unsigned int localSerial = localIdx0I*WG_DIM_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%TPI;
  unsigned int aK = localSerial/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;


  //A += GET_GLOBAL_INDEX_A( (groupIdx0I*MACRO_TILE_0I + (localSerial+(LID)*WG_DIM_0I*WG_DIM_1J)/NUM_UNROLL_ITER), localSerial%NUM_UNROLL_ITER )


  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + GET_LOCAL_INDEX_B(bK, bJ);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */
    lA[0*TPI] = 1.f; // A[0*TPI+0*strideAK];
    lA[1*TPI] = 1.f; // A[1*TPI+0*strideAK];
    lA[2*TPI] = 1.f; // A[2*TPI+0*strideAK];
#if NUM_UNROLL_ITER>8
    lA[3*TPI] = A[3*TPI+0*strideAK];
    lA[4*TPI] = A[4*TPI+0*strideAK];
    lA[5*TPI] = A[5*TPI+0*strideAK];
#endif

    lB[0*TPI] = 1.f; // B[0*TPI+0*strideBK];
    lB[1*TPI] = 1.f; // B[1*TPI+0*strideBK];
    lB[2*TPI] = 1.f; // B[2*TPI+0*strideBK];
#if NUM_UNROLL_ITER>8
    lB[3*TPI] = B[3*TPI+0*strideBK];
    lB[4*TPI] = B[4*TPI+0*strideBK];
    lB[5*TPI] = B[5*TPI+0*strideBK];
#endif

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
#if NUM_UNROLL_ITER>8
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
#endif

    A += strideAK*NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MACRO_TILE_1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MACRO_TILE_0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

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
#endif


// original 0
// unroll 16
// both Cobalt & original validate
#if 0


const char * kernelSource = R"(

#define COBALT_PATH 0

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

#if COBALT_PATH
#else
  A +=  groupIdx0I*96 + localIdx0I + localIdx1J*strideAK;
  B +=  groupIdx1J*96 + localIdx0I + localIdx1J*strideBK;
#endif

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MACRO_TILE_1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MACRO_TILE_0I + localIdx0I;

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
#if COBALT_PATH
    __local DATA_TYPE_STR_A *lA = localA + GET_LOCAL_INDEX_A(localA0I, localAK);
    __local DATA_TYPE_STR_B *lB = localB + GET_LOCAL_INDEX_B(localB1J, localBK);
#else
    __local DATA_TYPE_STR_A *lA = localA + localIdx1J*96+localIdx0I; // old_fast
    __local DATA_TYPE_STR_B *lB = localB + localIdx1J*96+localIdx0I; // old_fast
#endif
    barrier(CLK_LOCAL_MEM_FENCE);


    /* load global -> local */
#if COBALT_PATH
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
#else
    lB[ 0] = B[0];
    lB[16] = B[16+0*strideBK];
    lB[32] = B[32+0*strideBK];
    lB[48] = B[48+0*strideBK];
    lB[64] = B[64+0*strideBK];
    lB[80] = B[80+0*strideBK];
    lA[ 0] = A[0];
    lA[16] = A[16+0*strideAK];
    lA[32] = A[32+0*strideAK];
    lA[48] = A[48+0*strideAK];
    lA[64] = A[64+0*strideAK];
    lA[80] = A[80+0*strideAK];
#endif



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
  printf("%02u, %02u, c=%f alpha=%f beta=%f\n", get_global_id(0), get_global_id(1), rC[0][0], alpha, beta );

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


#endif