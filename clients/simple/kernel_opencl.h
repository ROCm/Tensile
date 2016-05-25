/*

*/

const char * kernelSource_TN = 0;
const char * kernelSource_TT = 0;


// NN - 6x6 micro tile
// unroll 8
// single source load (w/ PAD to eliminate bank conflict added from ssl)
// this is fastest so far: 60 vgpr, 90% valusage, 84%peak
// read B more registers but coalesced
#if 1
const char * kernelSource_NN = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
// for NN
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK)  ( (IDX0I)*strideA0I + (IDXK) *strideAK  )
#define GET_GLOBAL_INDEX_B(IDXK, IDX1J)  ( (IDXK) *strideBK  + (IDX1J)*strideB1J )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J+PAD) )

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
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
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
  __local float localA[NUM_UNROLL_ITER*(MACRO_TILE_0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MACRO_TILE_1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;

  unsigned int aI = (localSerial%128)%TPI;
  unsigned int aK = (localSerial%128)/TPI;
  unsigned int bJ = (localSerial%128)/NUM_UNROLL_ITER;
  unsigned int bK = (localSerial%128)%NUM_UNROLL_ITER;

  __local  float *localPtr;
  __global float *globalPtr;
  unsigned int globalInc;

  // localSerial [0,127] load A, [128,256] load B
  if (localSerial < 128 ) { // A
    localPtr = localA + GET_LOCAL_INDEX_A(aI, aK);
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    localPtr = localB + bJ+bK*(MACRO_TILE_1J+PAD); // GET_LOCAL_INDEX_B(bK, bJ);
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);
    //printf("t=%03u g=%04u l=%03u\n", localSerial, GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK), bJ*NUM_UNROLL_ITER+bK*(MACRO_TILE_1J+PAD) );
    globalInc = NUM_UNROLL_ITER;
  }


  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */

    if (localSerial < 128) { // load A w/o lda
      localPtr[ 0*TPI ] = globalPtr[ 0*TPI ];
      localPtr[ 1*TPI ] = globalPtr[ 1*TPI ];
      localPtr[ 2*TPI ] = globalPtr[ 2*TPI ];
      localPtr[ 3*TPI ] = globalPtr[ 3*TPI ];
      localPtr[ 4*TPI ] = globalPtr[ 4*TPI ];
      localPtr[ 5*TPI ] = globalPtr[ 5*TPI ];
    } else { // load B w/ ldb
      /*printf("L[%03u] T[%03u] %i: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f \n", sumIterK, localSerial, localSerial < 128,
        globalPtr[ 0*TPI*strideBK ],
        globalPtr[ 1*TPI*strideBK ],
        globalPtr[ 2*TPI*strideBK ],
        globalPtr[ 3*TPI*strideBK ],
        globalPtr[ 4*TPI*strideBK ],
        globalPtr[ 5*TPI*strideBK ] );*/

      localPtr[ 0*TPI ] = globalPtr[ 0*TPI*strideBK ];
      localPtr[ 1*TPI ] = globalPtr[ 1*TPI*strideBK ];
      localPtr[ 2*TPI ] = globalPtr[ 2*TPI*strideBK ];
      localPtr[ 3*TPI ] = globalPtr[ 3*TPI*strideBK ];
      localPtr[ 4*TPI ] = globalPtr[ 4*TPI*strideBK ];
      localPtr[ 5*TPI ] = globalPtr[ 5*TPI*strideBK ];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //printf("L[%03u] T[%03u]: lA=%f; lB=%f\n", sumIterK, localSerial, localA[localSerial], localB[localSerial]);
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

    globalPtr += globalInc;
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

// NN - 6x6 micro tile
// unroll 8
// single source load (w/ PAD to eliminate bank conflict added from ssl)
// this is fastest so far: 63 vgpr, 86% valubusy, 80%peak
// read B simple but not coalesced
#if 0
const char * kernelSource_NN = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
// for NT
//#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
//#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
//#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

// for NN
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK)  ( (IDX0I)*strideA0I + (IDXK) *strideAK  )
#define GET_GLOBAL_INDEX_B(IDXK, IDX1J)  ( (IDXK) *strideBK  + (IDX1J)*strideB1J )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J+PAD) )

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
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
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
  __local float localA[NUM_UNROLL_ITER*(MACRO_TILE_0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MACRO_TILE_1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;

  unsigned int aI = (localSerial%128)%TPI;
  unsigned int aK = (localSerial%128)/TPI;
  unsigned int bJ = (localSerial%128)/TPI;
  unsigned int bK = (localSerial%128)%TPI;

  __local  float *localPtr;
  __global float *globalPtr;
  unsigned int globalInc;

  // localSerial [0,127] load A, [128,256] load B
  if (localSerial < 128 ) { // A
    localPtr = localA + GET_LOCAL_INDEX_A(aI, aK);
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    localPtr = localB + localSerial%128; // GET_LOCAL_INDEX_A(bJ, bK);
    globalPtr = B + (localSerial%128+groupIdx1J*MACRO_TILE_1J)*strideBK;
    globalInc = NUM_UNROLL_ITER;
  }


  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */
    /*printf("L[%03u] T[%03u] %i: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f \n", sumIterK, localSerial, localSerial < 128,
        globalPtr[ 0*TPI],
        globalPtr[ 1*TPI],
        globalPtr[ 2*TPI],
        globalPtr[ 3*TPI],
        globalPtr[ 4*TPI],
        globalPtr[ 5*TPI] );*/

    if (localSerial < 128) { // load A w/o lda
      localPtr[ 0*TPI ] = globalPtr[ 0*TPI ];
      localPtr[ 1*TPI ] = globalPtr[ 1*TPI ];
      localPtr[ 2*TPI ] = globalPtr[ 2*TPI ];
      localPtr[ 3*TPI ] = globalPtr[ 3*TPI ];
      localPtr[ 4*TPI ] = globalPtr[ 4*TPI ];
      localPtr[ 5*TPI ] = globalPtr[ 5*TPI ];
    } else if (localSerial < 128+96) { // load B w/ ldb
      localPtr[ 0*(MACRO_TILE_1J+PAD) ] = globalPtr[ 0 ];
      localPtr[ 1*(MACRO_TILE_1J+PAD) ] = globalPtr[ 1 ];
      localPtr[ 2*(MACRO_TILE_1J+PAD) ] = globalPtr[ 2 ];
      localPtr[ 3*(MACRO_TILE_1J+PAD) ] = globalPtr[ 3 ];
      localPtr[ 4*(MACRO_TILE_1J+PAD) ] = globalPtr[ 4 ];
      localPtr[ 5*(MACRO_TILE_1J+PAD) ] = globalPtr[ 5 ];
      localPtr[ 6*(MACRO_TILE_1J+PAD) ] = globalPtr[ 6 ];
      localPtr[ 7*(MACRO_TILE_1J+PAD) ] = globalPtr[ 7 ];
    } else {
      // nothing; loading b will only use 96 threads
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //printf("L[%03u] T[%03u]: lA=%f; lB=%f\n", sumIterK, localSerial, localA[localSerial], localB[localSerial]);
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

    // A += strideAK*NUM_UNROLL_ITER;
    // B += strideBK*NUM_UNROLL_ITER;
    globalPtr += globalInc;
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




// NT 8x8 micro tile
// unroll 8
// single source load (w/ PAD to eliminate bank conflict added from ssl)
// prefetch global -> local
// prefetch local -> registers
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_DIM_0I           16
#define WG_DIM_1J           16
#define MICRO_TILE_0I       8
#define MICRO_TILE_1J       8
#define MACRO_TILE_0I       128
#define MACRO_TILE_1J       128
#define PAD                 1
#define MACRO_TILE_0I_POW2  256
#define MACRO_TILE_1J_POW2  256
#define NUM_UNROLL_ITER     8
#define TPI (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J+PAD) )

/* data types */
#define DATA_TYPE_STR_A float
#define DATA_TYPE_STR_B float
#define DATA_TYPE_STR_C float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA float
#define FMA(A,B,DST) mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 8x8 micro-tile */
// load
#define MICRO_TILE_PREFETCH \
  rA_red[0] = compA[ offA + 0*WG_DIM_0I]; \
  rA_red[1] = compA[ offA + 1*WG_DIM_0I]; \
  rA_red[2] = compA[ offA + 2*WG_DIM_0I]; \
  rA_red[3] = compA[ offA + 3*WG_DIM_0I]; \
  rA_red[4] = compA[ offA + 4*WG_DIM_0I]; \
  rA_red[5] = compA[ offA + 5*WG_DIM_0I]; \
  rA_red[6] = compA[ offA + 6*WG_DIM_0I]; \
  rA_red[7] = compA[ offA + 7*WG_DIM_0I]; \
  rB_red[0] = compB[ offB + 0*WG_DIM_1J]; \
  rB_red[1] = compB[ offB + 1*WG_DIM_1J]; \
  rB_red[2] = compB[ offB + 2*WG_DIM_1J]; \
  rB_red[3] = compB[ offB + 3*WG_DIM_1J]; \
  rB_red[4] = compB[ offB + 4*WG_DIM_1J]; \
  rB_red[5] = compB[ offB + 5*WG_DIM_1J]; \
  rB_red[6] = compB[ offB + 6*WG_DIM_1J]; \
  rB_red[7] = compB[ offB + 7*WG_DIM_1J]; \
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
  mem_fence(CLK_LOCAL_MEM_FENCE);
  

// load black, compute red
#define MICRO_TILE_2 \
  rA_black[0] = compA[offA + 0*WG_DIM_0I]; \
  rA_black[1] = compA[offA + 1*WG_DIM_0I]; \
  rA_black[2] = compA[offA + 2*WG_DIM_0I]; \
  rA_black[3] = compA[offA + 3*WG_DIM_0I]; \
  rA_black[4] = compA[offA + 4*WG_DIM_0I]; \
  rA_black[5] = compA[offA + 5*WG_DIM_0I]; \
  rA_black[6] = compA[offA + 6*WG_DIM_0I]; \
  rA_black[7] = compA[offA + 7*WG_DIM_0I]; \
  \
  rB_black[0] = compB[offB + 0*WG_DIM_1J]; \
  rB_black[1] = compB[offB + 1*WG_DIM_1J]; \
  rB_black[2] = compB[offB + 2*WG_DIM_1J]; \
  rB_black[3] = compB[offB + 3*WG_DIM_1J]; \
  rB_black[4] = compB[offB + 4*WG_DIM_1J]; \
  rB_black[5] = compB[offB + 5*WG_DIM_1J]; \
  rB_black[6] = compB[offB + 6*WG_DIM_1J]; \
  rB_black[7] = compB[offB + 7*WG_DIM_1J]; \
  \
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
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
  mem_fence(CLK_LOCAL_MEM_FENCE); \
  \
  rA_red[0] = compA[offA + 0*WG_DIM_0I]; \
  rA_red[1] = compA[offA + 1*WG_DIM_0I]; \
  rA_red[2] = compA[offA + 2*WG_DIM_0I]; \
  rA_red[3] = compA[offA + 3*WG_DIM_0I]; \
  rA_red[4] = compA[offA + 4*WG_DIM_0I]; \
  rA_red[5] = compA[offA + 5*WG_DIM_0I]; \
  rA_red[6] = compA[offA + 6*WG_DIM_0I]; \
  rA_red[7] = compA[offA + 7*WG_DIM_0I]; \
  \
  rB_red[0] = compB[offB + 0*WG_DIM_1J]; \
  rB_red[1] = compB[offB + 1*WG_DIM_1J]; \
  rB_red[2] = compB[offB + 2*WG_DIM_1J]; \
  rB_red[3] = compB[offB + 3*WG_DIM_1J]; \
  rB_red[4] = compB[offB + 4*WG_DIM_1J]; \
  rB_red[5] = compB[offB + 5*WG_DIM_1J]; \
  rB_red[6] = compB[offB + 6*WG_DIM_1J]; \
  rB_red[7] = compB[offB + 7*WG_DIM_1J]; \
  \
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
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
  mem_fence(CLK_LOCAL_MEM_FENCE);


/* preprocessor definitions of kernel arguments*/
#define strideC0I 1
#define strideA0I 1
#define strideB1J 1

)"
R"(

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
  float rA_red[MICRO_TILE_0I];
  float rB_red[MICRO_TILE_1J];
  float rA_black[MICRO_TILE_0I];
  float rB_black[MICRO_TILE_1J];

  /* allocate local memory */
  __local float localA_raw[2*NUM_UNROLL_ITER*MACRO_TILE_0I_POW2];
  __local float localB_raw[2*NUM_UNROLL_ITER*MACRO_TILE_1J_POW2];

  // red   starts at 0*NUM_UNROLL_ITER*MACRO_TILE_POW2
  // black starts at 1*NUM_UNROLL_ITER*MACRO_TILE_POW2

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;

  unsigned int aI = (localSerial%128)%TPI;
  unsigned int aK = (localSerial%128)/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  __local  float  *loadPtrBase;
  unsigned int loadPtrOffset;
  unsigned int loadSwapBit;
  const __global float *globalPtr;
  unsigned int globalInc;

  // localSerial [0,127] load A, [128,256] load B
  if (localSerial < 128 ) { // A
    loadPtrOffset = GET_LOCAL_INDEX_A(aI, aK);
    loadPtrBase = localA_raw;
    loadSwapBit = NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    loadPtrOffset = GET_LOCAL_INDEX_A(bJ, bK);
    loadPtrBase = localB_raw;
    loadSwapBit = NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);
    globalInc = strideBK*NUM_UNROLL_ITER;
  }
  // address from which to read for computation
  unsigned int compAOffset = NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
  unsigned int compBOffset = NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;

  __local float *compA  = localA_raw + compAOffset;
  __local float *compB  = localB_raw + compBOffset;


  /* 0th load global -> local */
  __local float *loadPtr = loadPtrBase + loadPtrOffset;
  loadPtr[ 0*TPI] = globalPtr[ 0*TPI];
  loadPtr[ 1*TPI] = globalPtr[ 1*TPI];
#if NUM_UNROLL_ITER>2
  loadPtr[ 2*TPI] = globalPtr[ 2*TPI];
  loadPtr[ 3*TPI] = globalPtr[ 3*TPI];
#endif
#if NUM_UNROLL_ITER>4
  loadPtr[ 4*TPI] = globalPtr[ 4*TPI];
  loadPtr[ 5*TPI] = globalPtr[ 5*TPI];
  loadPtr[ 6*TPI] = globalPtr[ 6*TPI];
  loadPtr[ 7*TPI] = globalPtr[ 7*TPI];
#endif
#if NUM_UNROLL_ITER>8
  loadPtr[ 8*TPI] = globalPtr[ 8*TPI];
  loadPtr[ 9*TPI] = globalPtr[ 9*TPI];
  loadPtr[10*TPI] = globalPtr[10*TPI];
  loadPtr[11*TPI] = globalPtr[11*TPI];
  loadPtr[12*TPI] = globalPtr[12*TPI];
  loadPtr[13*TPI] = globalPtr[13*TPI];
  loadPtr[14*TPI] = globalPtr[14*TPI];
  loadPtr[15*TPI] = globalPtr[15*TPI];
#endif
  barrier(CLK_LOCAL_MEM_FENCE);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {

    /* swap load pointers */
    loadPtrOffset ^= loadSwapBit;
    loadPtr = loadPtrBase + loadPtrOffset;

    /* swap compute pointers */
    compAOffset ^= NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
    compBOffset ^= NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;
    compA = localA_raw + compAOffset;
    compB = localB_raw + compBOffset;

    /* load global -> local */
    globalPtr += globalInc;
    loadPtr[ 0*TPI] = globalPtr[ 0*TPI];
    loadPtr[ 1*TPI] = globalPtr[ 1*TPI];
#if NUM_UNROLL_ITER>2
    loadPtr[ 2*TPI] = globalPtr[ 2*TPI];
    loadPtr[ 3*TPI] = globalPtr[ 3*TPI];
#endif
#if NUM_UNROLL_ITER>4
    loadPtr[ 4*TPI] = globalPtr[ 4*TPI];
    loadPtr[ 5*TPI] = globalPtr[ 5*TPI];
    loadPtr[ 6*TPI] = globalPtr[ 6*TPI];
    loadPtr[ 7*TPI] = globalPtr[ 7*TPI];
#endif
#if NUM_UNROLL_ITER>8
    loadPtr[ 8*TPI] = globalPtr[ 8*TPI];
    loadPtr[ 9*TPI] = globalPtr[ 9*TPI];
    loadPtr[10*TPI] = globalPtr[10*TPI];
    loadPtr[11*TPI] = globalPtr[11*TPI];
    loadPtr[12*TPI] = globalPtr[12*TPI];
    loadPtr[13*TPI] = globalPtr[13*TPI];
    loadPtr[14*TPI] = globalPtr[14*TPI];
    loadPtr[15*TPI] = globalPtr[15*TPI];
#endif

    /* do mads */
    unsigned int offA = localIdx0I; // d0
    unsigned int offB = localIdx1J; // d1
    MICRO_TILE_PREFETCH
    MICRO_TILE_2
#if NUM_UNROLL_ITER>2
    MICRO_TILE_2
#endif
#if NUM_UNROLL_ITER>4
    MICRO_TILE_2
    MICRO_TILE_2
#endif
#if NUM_UNROLL_ITER>8
    MICRO_TILE_2
    MICRO_TILE_2
    MICRO_TILE_2
    MICRO_TILE_2
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

  } while (--sumIterK > 1);

  /* swap compute pointers */
  compAOffset ^= NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
  compBOffset ^= NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;
  compA = localA_raw + compAOffset;
  compB = localB_raw + compBOffset;

  /* do mads */
  unsigned int offA = localIdx0I; // d0
  unsigned int offB = localIdx1J; // d1
  MICRO_TILE_PREFETCH
  MICRO_TILE_2
#if NUM_UNROLL_ITER>4
  MICRO_TILE_2
#endif
#if NUM_UNROLL_ITER>4
  MICRO_TILE_2
  MICRO_TILE_2
#endif
#if NUM_UNROLL_ITER>8
  MICRO_TILE_2
  MICRO_TILE_2
  MICRO_TILE_2
  MICRO_TILE_2
#endif
  
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


// NT 6x6 micro tile
// unroll 8
// single source load (w/ PAD to eliminate bank conflict added from ssl)
// prefetch global -> local
// prefetch local -> registers
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_DIM_0I           16
#define WG_DIM_1J           16
#define MICRO_TILE_0I       6
#define MICRO_TILE_1J       6
#define MACRO_TILE_0I       96
#define MACRO_TILE_1J       96
#define MACRO_TILE_0I_POW2  128
#define MACRO_TILE_1J_POW2  128
#define NUM_UNROLL_ITER     8
#define PAD                 1
#define TPI (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J+PAD) )

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
// load
#define MICRO_TILE_PREFETCH \
  rA_red[0] = compA[ offA + 0*WG_DIM_0I]; \
  rA_red[1] = compA[ offA + 1*WG_DIM_0I]; \
  rA_red[2] = compA[ offA + 2*WG_DIM_0I]; \
  rA_red[3] = compA[ offA + 3*WG_DIM_0I]; \
  rA_red[4] = compA[ offA + 4*WG_DIM_0I]; \
  rA_red[5] = compA[ offA + 5*WG_DIM_0I]; \
  rB_red[0] = compB[ offB + 0*WG_DIM_1J]; \
  rB_red[1] = compB[ offB + 1*WG_DIM_1J]; \
  rB_red[2] = compB[ offB + 2*WG_DIM_1J]; \
  rB_red[3] = compB[ offB + 3*WG_DIM_1J]; \
  rB_red[4] = compB[ offB + 4*WG_DIM_1J]; \
  rB_red[5] = compB[ offB + 5*WG_DIM_1J]; \
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
  mem_fence(CLK_LOCAL_MEM_FENCE);
  

// load black, compute red
#define MICRO_TILE_2 \
  rA_black[0] = compA[offA + 0*WG_DIM_0I]; \
  rA_black[1] = compA[offA + 1*WG_DIM_0I]; \
  rA_black[2] = compA[offA + 2*WG_DIM_0I]; \
  rA_black[3] = compA[offA + 3*WG_DIM_0I]; \
  rA_black[4] = compA[offA + 4*WG_DIM_0I]; \
  rA_black[5] = compA[offA + 5*WG_DIM_0I]; \
  rB_black[0] = compB[offB + 0*WG_DIM_1J]; \
  rB_black[1] = compB[offB + 1*WG_DIM_1J]; \
  rB_black[2] = compB[offB + 2*WG_DIM_1J]; \
  rB_black[3] = compB[offB + 3*WG_DIM_1J]; \
  rB_black[4] = compB[offB + 4*WG_DIM_1J]; \
  rB_black[5] = compB[offB + 5*WG_DIM_1J]; \
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
  TYPE_MAD( rA_red[0], rB_red[0], rC[0][0]); \
  TYPE_MAD( rA_red[0], rB_red[1], rC[0][1]); \
  TYPE_MAD( rA_red[0], rB_red[2], rC[0][2]); \
  TYPE_MAD( rA_red[0], rB_red[3], rC[0][3]); \
  TYPE_MAD( rA_red[0], rB_red[4], rC[0][4]); \
  TYPE_MAD( rA_red[0], rB_red[5], rC[0][5]); \
  TYPE_MAD( rA_red[1], rB_red[0], rC[1][0]); \
  TYPE_MAD( rA_red[1], rB_red[1], rC[1][1]); \
  TYPE_MAD( rA_red[1], rB_red[2], rC[1][2]); \
  TYPE_MAD( rA_red[1], rB_red[3], rC[1][3]); \
  TYPE_MAD( rA_red[1], rB_red[4], rC[1][4]); \
  TYPE_MAD( rA_red[1], rB_red[5], rC[1][5]); \
  TYPE_MAD( rA_red[2], rB_red[0], rC[2][0]); \
  TYPE_MAD( rA_red[2], rB_red[1], rC[2][1]); \
  TYPE_MAD( rA_red[2], rB_red[2], rC[2][2]); \
  TYPE_MAD( rA_red[2], rB_red[3], rC[2][3]); \
  TYPE_MAD( rA_red[2], rB_red[4], rC[2][4]); \
  TYPE_MAD( rA_red[2], rB_red[5], rC[2][5]); \
  TYPE_MAD( rA_red[3], rB_red[0], rC[3][0]); \
  TYPE_MAD( rA_red[3], rB_red[1], rC[3][1]); \
  TYPE_MAD( rA_red[3], rB_red[2], rC[3][2]); \
  TYPE_MAD( rA_red[3], rB_red[3], rC[3][3]); \
  TYPE_MAD( rA_red[3], rB_red[4], rC[3][4]); \
  TYPE_MAD( rA_red[3], rB_red[5], rC[3][5]); \
  TYPE_MAD( rA_red[4], rB_red[0], rC[4][0]); \
  TYPE_MAD( rA_red[4], rB_red[1], rC[4][1]); \
  TYPE_MAD( rA_red[4], rB_red[2], rC[4][2]); \
  TYPE_MAD( rA_red[4], rB_red[3], rC[4][3]); \
  TYPE_MAD( rA_red[4], rB_red[4], rC[4][4]); \
  TYPE_MAD( rA_red[4], rB_red[5], rC[4][5]); \
  TYPE_MAD( rA_red[5], rB_red[0], rC[5][0]); \
  TYPE_MAD( rA_red[5], rB_red[1], rC[5][1]); \
  TYPE_MAD( rA_red[5], rB_red[2], rC[5][2]); \
  TYPE_MAD( rA_red[5], rB_red[3], rC[5][3]); \
  TYPE_MAD( rA_red[5], rB_red[4], rC[5][4]); \
  TYPE_MAD( rA_red[5], rB_red[5], rC[5][5]); \
  /* mem_fence(CLK_LOCAL_MEM_FENCE); */ \
  \
  rA_red[0] = compA[ offA + 0*WG_DIM_0I]; \
  rA_red[1] = compA[ offA + 1*WG_DIM_0I]; \
  rA_red[2] = compA[ offA + 2*WG_DIM_0I]; \
  rA_red[3] = compA[ offA + 3*WG_DIM_0I]; \
  rA_red[4] = compA[ offA + 4*WG_DIM_0I]; \
  rA_red[5] = compA[ offA + 5*WG_DIM_0I]; \
  rB_red[0] = compB[ offB + 0*WG_DIM_1J]; \
  rB_red[1] = compB[ offB + 1*WG_DIM_1J]; \
  rB_red[2] = compB[ offB + 2*WG_DIM_1J]; \
  rB_red[3] = compB[ offB + 3*WG_DIM_1J]; \
  rB_red[4] = compB[ offB + 4*WG_DIM_1J]; \
  rB_red[5] = compB[ offB + 5*WG_DIM_1J]; \
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
  TYPE_MAD( rA_black[0], rB_black[0], rC[0][0]); \
  TYPE_MAD( rA_black[0], rB_black[1], rC[0][1]); \
  TYPE_MAD( rA_black[0], rB_black[2], rC[0][2]); \
  TYPE_MAD( rA_black[0], rB_black[3], rC[0][3]); \
  TYPE_MAD( rA_black[0], rB_black[4], rC[0][4]); \
  TYPE_MAD( rA_black[0], rB_black[5], rC[0][5]); \
  TYPE_MAD( rA_black[1], rB_black[0], rC[1][0]); \
  TYPE_MAD( rA_black[1], rB_black[1], rC[1][1]); \
  TYPE_MAD( rA_black[1], rB_black[2], rC[1][2]); \
  TYPE_MAD( rA_black[1], rB_black[3], rC[1][3]); \
  TYPE_MAD( rA_black[1], rB_black[4], rC[1][4]); \
  TYPE_MAD( rA_black[1], rB_black[5], rC[1][5]); \
  TYPE_MAD( rA_black[2], rB_black[0], rC[2][0]); \
  TYPE_MAD( rA_black[2], rB_black[1], rC[2][1]); \
  TYPE_MAD( rA_black[2], rB_black[2], rC[2][2]); \
  TYPE_MAD( rA_black[2], rB_black[3], rC[2][3]); \
  TYPE_MAD( rA_black[2], rB_black[4], rC[2][4]); \
  TYPE_MAD( rA_black[2], rB_black[5], rC[2][5]); \
  TYPE_MAD( rA_black[3], rB_black[0], rC[3][0]); \
  TYPE_MAD( rA_black[3], rB_black[1], rC[3][1]); \
  TYPE_MAD( rA_black[3], rB_black[2], rC[3][2]); \
  TYPE_MAD( rA_black[3], rB_black[3], rC[3][3]); \
  TYPE_MAD( rA_black[3], rB_black[4], rC[3][4]); \
  TYPE_MAD( rA_black[3], rB_black[5], rC[3][5]); \
  TYPE_MAD( rA_black[4], rB_black[0], rC[4][0]); \
  TYPE_MAD( rA_black[4], rB_black[1], rC[4][1]); \
  TYPE_MAD( rA_black[4], rB_black[2], rC[4][2]); \
  TYPE_MAD( rA_black[4], rB_black[3], rC[4][3]); \
  TYPE_MAD( rA_black[4], rB_black[4], rC[4][4]); \
  TYPE_MAD( rA_black[4], rB_black[5], rC[4][5]); \
  TYPE_MAD( rA_black[5], rB_black[0], rC[5][0]); \
  TYPE_MAD( rA_black[5], rB_black[1], rC[5][1]); \
  TYPE_MAD( rA_black[5], rB_black[2], rC[5][2]); \
  TYPE_MAD( rA_black[5], rB_black[3], rC[5][3]); \
  TYPE_MAD( rA_black[5], rB_black[4], rC[5][4]); \
  TYPE_MAD( rA_black[5], rB_black[5], rC[5][5]); \
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
  float rA_red[MICRO_TILE_0I];
  float rB_red[MICRO_TILE_1J];
  float rA_black[MICRO_TILE_0I];
  float rB_black[MICRO_TILE_1J];

  /* allocate local memory */
  __local float localA_raw[2*NUM_UNROLL_ITER*MACRO_TILE_0I_POW2];
  __local float localB_raw[2*NUM_UNROLL_ITER*MACRO_TILE_1J_POW2];

  // red   starts at 0*NUM_UNROLL_ITER*MACRO_TILE_POW2
  // black starts at 1*NUM_UNROLL_ITER*MACRO_TILE_POW2

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;

  unsigned int aI = (localSerial%128)%TPI;
  unsigned int aK = (localSerial%128)/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  __local  float  *loadPtrBase;
  unsigned int loadPtrOffset;
  unsigned int loadSwapBit;
  const __global float *globalPtr;
  unsigned int globalInc;

  // localSerial [0,127] load A, [128,256] load B
  if (localSerial < 128 ) { // A
    loadPtrOffset = GET_LOCAL_INDEX_A(aI, aK);
    loadPtrBase = localA_raw;
    loadSwapBit = NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    loadPtrOffset = GET_LOCAL_INDEX_A(bJ, bK);
    loadPtrBase = localB_raw;
    loadSwapBit = NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);
    globalInc = strideBK*NUM_UNROLL_ITER;
  }
  // address from which to read for computation
  unsigned int compAOffset = NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
  unsigned int compBOffset = NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;

  __local float *compA  = localA_raw + compAOffset;
  __local float *compB  = localB_raw + compBOffset;


  /* 0th load global -> local */
  __local float *loadPtr = loadPtrBase + loadPtrOffset;
  loadPtr[ 0*TPI] = globalPtr[ 0*TPI];
  loadPtr[ 1*TPI] = globalPtr[ 1*TPI];
  loadPtr[ 2*TPI] = globalPtr[ 2*TPI];
  loadPtr[ 3*TPI] = globalPtr[ 3*TPI];
  loadPtr[ 4*TPI] = globalPtr[ 4*TPI];
  loadPtr[ 5*TPI] = globalPtr[ 5*TPI];
#if NUM_UNROLL_ITER>8
  loadPtr[ 6*TPI] = globalPtr[ 6*TPI];
  loadPtr[ 7*TPI] = globalPtr[ 7*TPI];
  loadPtr[ 8*TPI] = globalPtr[ 8*TPI];
  loadPtr[ 9*TPI] = globalPtr[ 9*TPI];
  loadPtr[10*TPI] = globalPtr[10*TPI];
  loadPtr[11*TPI] = globalPtr[11*TPI];
#endif
  barrier(CLK_LOCAL_MEM_FENCE);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {

    /* swap load pointers */
    loadPtrOffset ^= loadSwapBit;
    loadPtr = loadPtrBase + loadPtrOffset;

    /* swap compute pointers */
    compAOffset ^= NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
    compBOffset ^= NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;
    compA = localA_raw + compAOffset;
    compB = localB_raw + compBOffset;

    /* load global -> local */
    globalPtr += globalInc;
    loadPtr[ 0*TPI] = globalPtr[ 0*TPI];
    loadPtr[ 1*TPI] = globalPtr[ 1*TPI];
    loadPtr[ 2*TPI] = globalPtr[ 2*TPI];
    loadPtr[ 3*TPI] = globalPtr[ 3*TPI];
    loadPtr[ 4*TPI] = globalPtr[ 4*TPI];
    loadPtr[ 5*TPI] = globalPtr[ 5*TPI];
#if NUM_UNROLL_ITER>8
    loadPtr[ 6*TPI] = globalPtr[ 6*TPI];
    loadPtr[ 7*TPI] = globalPtr[ 7*TPI];
    loadPtr[ 8*TPI] = globalPtr[ 8*TPI];
    loadPtr[ 9*TPI] = globalPtr[ 9*TPI];
    loadPtr[10*TPI] = globalPtr[10*TPI];
    loadPtr[11*TPI] = globalPtr[11*TPI];
#endif

    /* do mads */
    unsigned int offA = localIdx0I; // d0
    unsigned int offB = localIdx1J; // d1
    MICRO_TILE_PREFETCH
    MICRO_TILE_2
    MICRO_TILE_2
    MICRO_TILE_2
    MICRO_TILE_2
#if NUM_UNROLL_ITER>8
    MICRO_TILE_2
    MICRO_TILE_2
    MICRO_TILE_2
    MICRO_TILE_2
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

  } while (--sumIterK > 1);

  /* swap compute pointers */
  compAOffset ^= NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
  compBOffset ^= NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;
  compA = localA_raw + compAOffset;
  compB = localB_raw + compBOffset;

  /* do mads */
  unsigned int offA = localIdx0I; // d0
  unsigned int offB = localIdx1J; // d1
  MICRO_TILE_PREFETCH
  MICRO_TILE_2
  MICRO_TILE_2
  MICRO_TILE_2
  MICRO_TILE_2
#if NUM_UNROLL_ITER>8
  MICRO_TILE_2
  MICRO_TILE_2
  MICRO_TILE_2
  MICRO_TILE_2
#endif
  
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

// NT 6x6 micro tile
// unroll 8
// single source load (w/ PAD to eliminate bank conflict added from ssl)
// prefetch global -> local
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_DIM_0I           16
#define WG_DIM_1J           16
#define MICRO_TILE_0I       6
#define MICRO_TILE_1J       6
#define MACRO_TILE_0I       96
#define MACRO_TILE_1J       96
#define MACRO_TILE_0I_POW2  128
#define MACRO_TILE_1J_POW2  128
#define NUM_UNROLL_ITER     8
#define PAD                 1
#define TPI (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J+PAD) )

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
  rA[0] = compA[offA + 0*WG_DIM_0I]; \
  rA[1] = compA[offA + 1*WG_DIM_0I]; \
  rA[2] = compA[offA + 2*WG_DIM_0I]; \
  rA[3] = compA[offA + 3*WG_DIM_0I]; \
  rA[4] = compA[offA + 4*WG_DIM_0I]; \
  rA[5] = compA[offA + 5*WG_DIM_0I]; \
  rB[0] = compB[offB + 0*WG_DIM_1J]; \
  rB[1] = compB[offB + 1*WG_DIM_1J]; \
  rB[2] = compB[offB + 2*WG_DIM_1J]; \
  rB[3] = compB[offB + 3*WG_DIM_1J]; \
  rB[4] = compB[offB + 4*WG_DIM_1J]; \
  rB[5] = compB[offB + 5*WG_DIM_1J]; \
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
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
  __local float localA_raw[2*NUM_UNROLL_ITER*MACRO_TILE_0I_POW2];
  __local float localB_raw[2*NUM_UNROLL_ITER*MACRO_TILE_1J_POW2];

  // red   starts at 0*NUM_UNROLL_ITER*MACRO_TILE_POW2
  // black starts at 1*NUM_UNROLL_ITER*MACRO_TILE_POW2

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;

  unsigned int aI = (localSerial%128)%TPI;
  unsigned int aK = (localSerial%128)/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  __local  float  *loadPtrBase;
  unsigned int loadPtrOffset;
  unsigned int loadSwapBit;
  const __global float *globalPtr;
  unsigned int globalInc;

  // localSerial [0,127] load A, [128,256] load B
  if (localSerial < 128 ) { // A
    loadPtrOffset = GET_LOCAL_INDEX_A(aI, aK);
    loadPtrBase = localA_raw;
    loadSwapBit = NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    loadPtrOffset = GET_LOCAL_INDEX_A(bJ, bK);
    loadPtrBase = localB_raw;
    loadSwapBit = NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);
    globalInc = strideBK*NUM_UNROLL_ITER;
  }
  // address from which to read for computation
  unsigned int compAOffset = NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
  unsigned int compBOffset = NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;

  __local float *compA  = localA_raw + compAOffset;
  __local float *compB  = localB_raw + compBOffset;


  /* 0th load global -> local */
  __local float *loadPtr = loadPtrBase + loadPtrOffset;
  loadPtr[ 0*TPI] = globalPtr[ 0*TPI];
  loadPtr[ 1*TPI] = globalPtr[ 1*TPI];
  loadPtr[ 2*TPI] = globalPtr[ 2*TPI];
  loadPtr[ 3*TPI] = globalPtr[ 3*TPI];
  loadPtr[ 4*TPI] = globalPtr[ 4*TPI];
  loadPtr[ 5*TPI] = globalPtr[ 5*TPI];
#if NUM_UNROLL_ITER>8
  loadPtr[ 6*TPI] = globalPtr[ 6*TPI];
  loadPtr[ 7*TPI] = globalPtr[ 7*TPI];
  loadPtr[ 8*TPI] = globalPtr[ 8*TPI];
  loadPtr[ 9*TPI] = globalPtr[ 9*TPI];
  loadPtr[10*TPI] = globalPtr[10*TPI];
  loadPtr[11*TPI] = globalPtr[11*TPI];
#endif
  barrier(CLK_LOCAL_MEM_FENCE);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {

    /* swap load pointers */
    loadPtrOffset ^= loadSwapBit;
    loadPtr = loadPtrBase + loadPtrOffset;

    /* swap compute pointers */
    compAOffset ^= NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
    compBOffset ^= NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;
    compA = localA_raw + compAOffset;
    compB = localB_raw + compBOffset;

    /* load global -> local */
    globalPtr += globalInc;
    loadPtr[ 0*TPI] = globalPtr[ 0*TPI];
    loadPtr[ 1*TPI] = globalPtr[ 1*TPI];
    loadPtr[ 2*TPI] = globalPtr[ 2*TPI];
    loadPtr[ 3*TPI] = globalPtr[ 3*TPI];
    loadPtr[ 4*TPI] = globalPtr[ 4*TPI];
    loadPtr[ 5*TPI] = globalPtr[ 5*TPI];
#if NUM_UNROLL_ITER>8
    loadPtr[ 6*TPI] = globalPtr[ 6*TPI];
    loadPtr[ 7*TPI] = globalPtr[ 7*TPI];
    loadPtr[ 8*TPI] = globalPtr[ 8*TPI];
    loadPtr[ 9*TPI] = globalPtr[ 9*TPI];
    loadPtr[10*TPI] = globalPtr[10*TPI];
    loadPtr[11*TPI] = globalPtr[11*TPI];
#endif

    /* do mads */
    unsigned int offA = localIdx0I; // d0
    unsigned int offB = localIdx1J; // d1
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
    barrier(CLK_LOCAL_MEM_FENCE);

  } while (--sumIterK > 1);

  /* swap compute pointers */
  compAOffset ^= NUM_UNROLL_ITER*MACRO_TILE_0I_POW2;
  compBOffset ^= NUM_UNROLL_ITER*MACRO_TILE_1J_POW2;
  compA = localA_raw + compAOffset;
  compB = localB_raw + compBOffset;

  /* do mads */
  unsigned int offA = localIdx0I; // d0
  unsigned int offB = localIdx1J; // d1
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

// NT with branches
// 6x6 micro tile
// unroll 8
// single source load (w/ PAD to eliminate bank conflict added from ssl)
// this is fastest so far: 61 vgpr, 88% valusage, 80%peak
#if 1
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J+PAD) )

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
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
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


)"
R"(

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
  __local float localA[NUM_UNROLL_ITER*(MACRO_TILE_0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MACRO_TILE_1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;

  unsigned int aI = (localSerial%128)%TPI; // 0->16-1
  unsigned int aK = (localSerial%128)/TPI; // 0->8-1
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  __local float *localPtr;
  __global float *globalPtr;
  unsigned int globalInc;
  bool doLoad;
  unsigned int maxLoads;

  // localSerial [0,127] load A, [128,256] load B
  if (localSerial < 128 ) { // A
    localPtr = localA + GET_LOCAL_INDEX_A(aI, aK);
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
    //doLoad = aI+groupIdx0I*MACRO_TILE_0I < size0I;
    maxLoads = (size0I - groupIdx0I*MACRO_TILE_0I)/TPI;
    if (aI < (size0I - groupIdx0I*MACRO_TILE_0I)%TPI ) { maxLoads++; }
  } else { // B
    localPtr = localB + GET_LOCAL_INDEX_A(bJ, bK);
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);
    globalInc = strideBK*NUM_UNROLL_ITER;
    //doLoad = bJ+groupIdx1J*MACRO_TILE_1J < size1J;
    maxLoads = (size1J - groupIdx1J*MACRO_TILE_1J)/TPI;
    if (bJ < (size1J - groupIdx1J*MACRO_TILE_1J)%TPI ) { maxLoads++; }
  }
  //printf("%2u,%2u maxLoads=%u\n", get_global_id(0), get_global_id(1), maxLoads);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */
    if (get_group_id(0) < get_num_groups(0)-1 &&
        get_group_id(1) < get_num_groups(1)-1) {
      localPtr[ 0*TPI] = globalPtr[ 0*TPI];
      localPtr[ 1*TPI] = globalPtr[ 1*TPI];
      localPtr[ 2*TPI] = globalPtr[ 2*TPI];
      localPtr[ 3*TPI] = globalPtr[ 3*TPI];
      localPtr[ 4*TPI] = globalPtr[ 4*TPI];
      localPtr[ 5*TPI] = globalPtr[ 5*TPI];
    } else {
      if (maxLoads >= 6) {
        localPtr[ 0*TPI] = globalPtr[ 0*TPI];
        localPtr[ 1*TPI] = globalPtr[ 1*TPI];
        localPtr[ 2*TPI] = globalPtr[ 2*TPI];
        localPtr[ 3*TPI] = globalPtr[ 3*TPI];
        localPtr[ 4*TPI] = globalPtr[ 4*TPI];
        localPtr[ 5*TPI] = globalPtr[ 5*TPI];
      } else if (maxLoads == 5) {
        localPtr[ 0*TPI] = globalPtr[ 0*TPI];
        localPtr[ 1*TPI] = globalPtr[ 1*TPI];
        localPtr[ 2*TPI] = globalPtr[ 2*TPI];
        localPtr[ 3*TPI] = globalPtr[ 3*TPI];
        localPtr[ 4*TPI] = globalPtr[ 4*TPI];
      } else if (maxLoads == 4) {
        localPtr[ 0*TPI] = globalPtr[ 0*TPI];
        localPtr[ 1*TPI] = globalPtr[ 1*TPI];
        localPtr[ 2*TPI] = globalPtr[ 2*TPI];
        localPtr[ 3*TPI] = globalPtr[ 3*TPI];
      } else if (maxLoads == 3) {
        localPtr[ 0*TPI] = globalPtr[ 0*TPI];
        localPtr[ 1*TPI] = globalPtr[ 1*TPI];
        localPtr[ 2*TPI] = globalPtr[ 2*TPI];
      } else if (maxLoads == 2) {
        localPtr[ 0*TPI] = globalPtr[ 0*TPI];
        localPtr[ 1*TPI] = globalPtr[ 1*TPI];
      } else if (maxLoads == 1) {
        localPtr[ 0*TPI] = globalPtr[ 0*TPI];
      }
    }

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

    globalPtr += globalInc;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MACRO_TILE_1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MACRO_TILE_0I + localIdx0I;
  //printf("%04u, %04u, %f\n", globalIdxC0I, globalIdxC1J, rC[0][0] );

  /* write global C */
  if (globalIdxC0I + 0*WG_DIM_0I < size0I && globalIdxC1J + 0*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[0][0], beta) }
  if (globalIdxC0I + 0*WG_DIM_0I < size0I && globalIdxC1J + 1*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[0][1], beta) }
  if (globalIdxC0I + 0*WG_DIM_0I < size0I && globalIdxC1J + 2*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[0][2], beta) }
  if (globalIdxC0I + 0*WG_DIM_0I < size0I && globalIdxC1J + 3*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[0][3], beta) }
  if (globalIdxC0I + 0*WG_DIM_0I < size0I && globalIdxC1J + 4*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[0][4], beta) }
  if (globalIdxC0I + 0*WG_DIM_0I < size0I && globalIdxC1J + 5*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[0][5], beta) }

  if (globalIdxC0I + 1*WG_DIM_0I < size0I && globalIdxC1J + 0*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[1][0], beta) }
  if (globalIdxC0I + 1*WG_DIM_0I < size0I && globalIdxC1J + 1*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[1][1], beta) }
  if (globalIdxC0I + 1*WG_DIM_0I < size0I && globalIdxC1J + 2*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[1][2], beta) }
  if (globalIdxC0I + 1*WG_DIM_0I < size0I && globalIdxC1J + 3*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[1][3], beta) }
  if (globalIdxC0I + 1*WG_DIM_0I < size0I && globalIdxC1J + 4*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[1][4], beta) }
  if (globalIdxC0I + 1*WG_DIM_0I < size0I && globalIdxC1J + 5*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[1][5], beta) }

  if (globalIdxC0I + 2*WG_DIM_0I < size0I && globalIdxC1J + 0*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[2][0], beta) }
  if (globalIdxC0I + 2*WG_DIM_0I < size0I && globalIdxC1J + 1*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[2][1], beta) }
  if (globalIdxC0I + 2*WG_DIM_0I < size0I && globalIdxC1J + 2*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[2][2], beta) }
  if (globalIdxC0I + 2*WG_DIM_0I < size0I && globalIdxC1J + 3*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[2][3], beta) }
  if (globalIdxC0I + 2*WG_DIM_0I < size0I && globalIdxC1J + 4*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[2][4], beta) }
  if (globalIdxC0I + 2*WG_DIM_0I < size0I && globalIdxC1J + 5*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[2][5], beta) }

  if (globalIdxC0I + 3*WG_DIM_0I < size0I && globalIdxC1J + 0*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[3][0], beta) }
  if (globalIdxC0I + 3*WG_DIM_0I < size0I && globalIdxC1J + 1*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[3][1], beta) }
  if (globalIdxC0I + 3*WG_DIM_0I < size0I && globalIdxC1J + 2*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[3][2], beta) }
  if (globalIdxC0I + 3*WG_DIM_0I < size0I && globalIdxC1J + 3*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[3][3], beta) }
  if (globalIdxC0I + 3*WG_DIM_0I < size0I && globalIdxC1J + 4*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[3][4], beta) }
  if (globalIdxC0I + 3*WG_DIM_0I < size0I && globalIdxC1J + 5*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[3][5], beta) }

  if (globalIdxC0I + 4*WG_DIM_0I < size0I && globalIdxC1J + 0*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[4][0], beta) }
  if (globalIdxC0I + 4*WG_DIM_0I < size0I && globalIdxC1J + 1*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[4][1], beta) }
  if (globalIdxC0I + 4*WG_DIM_0I < size0I && globalIdxC1J + 2*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[4][2], beta) }
  if (globalIdxC0I + 4*WG_DIM_0I < size0I && globalIdxC1J + 3*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[4][3], beta) }
  if (globalIdxC0I + 4*WG_DIM_0I < size0I && globalIdxC1J + 4*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[4][4], beta) }
  if (globalIdxC0I + 4*WG_DIM_0I < size0I && globalIdxC1J + 5*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[4][5], beta) }

  if (globalIdxC0I + 5*WG_DIM_0I < size0I && globalIdxC1J + 0*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[5][0], beta) }
  if (globalIdxC0I + 5*WG_DIM_0I < size0I && globalIdxC1J + 1*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[5][1], beta) }
  if (globalIdxC0I + 5*WG_DIM_0I < size0I && globalIdxC1J + 2*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[5][2], beta) }
  if (globalIdxC0I + 5*WG_DIM_0I < size0I && globalIdxC1J + 3*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[5][3], beta) }
  if (globalIdxC0I + 5*WG_DIM_0I < size0I && globalIdxC1J + 4*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[5][4], beta) }
  if (globalIdxC0I + 5*WG_DIM_0I < size0I && globalIdxC1J + 5*WG_DIM_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[5][5], beta) }

};
)";
#endif

// NT 6x6 micro tile
// unroll 8
// single source load (w/ PAD to eliminate bank conflict added from ssl)
// this is fastest so far: 60 vgpr, 90% valusage, 84%peak
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J+PAD) )

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
  offA += (MACRO_TILE_0I+PAD); \
  offB += (MACRO_TILE_1J+PAD); \
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
  __local float localA[NUM_UNROLL_ITER*(MACRO_TILE_0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MACRO_TILE_1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;

  unsigned int aI = (localSerial%128)%TPI;
  unsigned int aK = (localSerial%128)/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  __local float *localPtr;
  __global float *globalPtr;
  unsigned int globalInc;

  // localSerial [0,127] load A, [128,256] load B
  if (localSerial < 128 ) { // A
    localPtr = localA + GET_LOCAL_INDEX_A(aI, aK);
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    localPtr = localB + GET_LOCAL_INDEX_A(bJ, bK);
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);
    globalInc = strideBK*NUM_UNROLL_ITER;
  }


  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */
    /*printf("L[%03u] T[%03u] %i: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f \n", sumIterK, localSerial, localSerial < 128,
        globalPtr[ 0*TPI],
        globalPtr[ 1*TPI],
        globalPtr[ 2*TPI],
        globalPtr[ 3*TPI],
        globalPtr[ 4*TPI],
        globalPtr[ 5*TPI] );*/

    localPtr[ 0*TPI] = globalPtr[ 0*TPI];
    localPtr[ 1*TPI] = globalPtr[ 1*TPI];
    localPtr[ 2*TPI] = globalPtr[ 2*TPI];
    localPtr[ 3*TPI] = globalPtr[ 3*TPI];
    localPtr[ 4*TPI] = globalPtr[ 4*TPI];
    localPtr[ 5*TPI] = globalPtr[ 5*TPI];
#if NUM_UNROLL_ITER>8
    localPtr[ 6*TPI] = globalPtr[ 6*TPI];
    localPtr[ 7*TPI] = globalPtr[ 7*TPI];
    localPtr[ 8*TPI] = globalPtr[ 8*TPI];
    localPtr[ 9*TPI] = globalPtr[ 9*TPI];
    localPtr[10*TPI] = globalPtr[10*TPI];
    localPtr[11*TPI] = globalPtr[11*TPI];
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
    //printf("L[%03u] T[%03u]: lA=%f; lB=%f\n", sumIterK, localSerial, localA[localSerial], localB[localSerial]);
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

    // A += strideAK*NUM_UNROLL_ITER;
    // B += strideBK*NUM_UNROLL_ITER;
    globalPtr += globalInc;
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

// NT 6x6 micro tile
// unroll 8
// prefetch load global->lds
// simpler swap - not complete
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   8
// threads-per-iteration (loading unroll)
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
  rA[0] = local_A_comp[offA + 0*WG_DIM_0I]; \
  rA[1] = local_A_comp[offA + 1*WG_DIM_0I]; \
  rA[2] = local_A_comp[offA + 2*WG_DIM_0I]; \
  rA[3] = local_A_comp[offA + 3*WG_DIM_0I]; \
  rA[4] = local_A_comp[offA + 4*WG_DIM_0I]; \
  rA[5] = local_A_comp[offA + 5*WG_DIM_0I]; \
  rB[0] = local_B_comp[offB + 0*WG_DIM_1J]; \
  rB[1] = local_B_comp[offB + 1*WG_DIM_1J]; \
  rB[2] = local_B_comp[offB + 2*WG_DIM_1J]; \
  rB[3] = local_B_comp[offB + 3*WG_DIM_1J]; \
  rB[4] = local_B_comp[offB + 4*WG_DIM_1J]; \
  rB[5] = local_B_comp[offB + 5*WG_DIM_1J]; \
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
  __local float local_AB[2*NUM_UNROLL_ITER*(MACRO_TILE_0I+MACRO_TILE_1J)];

  __local float *local_A_raw_red   = local_AB+0*NUM_UNROLL_ITER*MACRO_TILE_0I+0*NUM_UNROLL_ITER*MACRO_TILE_1J;
  __local float *local_B_raw_red   = local_AB+0*NUM_UNROLL_ITER*MACRO_TILE_0I+1*NUM_UNROLL_ITER*MACRO_TILE_1J;
  __local float *local_A_raw_black = local_AB+1*NUM_UNROLL_ITER*MACRO_TILE_0I+0*NUM_UNROLL_ITER*MACRO_TILE_1J;
  __local float *local_B_raw_black = local_AB+1*NUM_UNROLL_ITER*MACRO_TILE_0I+1*NUM_UNROLL_ITER*MACRO_TILE_1J;

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

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);

  /* compute pointers (no offset) */
  __local float *local_A_comp_red   = local_A_raw_red;
  __local float *local_B_comp_red   = local_B_raw_red;
  __local float *local_A_comp_black = local_A_raw_black;
  __local float *local_B_comp_black = local_B_raw_black;

  /* load pointers (offset) */
  __local float *local_A_load_red   = local_A_raw_red   + GET_LOCAL_INDEX_A(aI, aK);
  __local float *local_B_load_red   = local_B_raw_red   + GET_LOCAL_INDEX_B(bK, bJ);
  __local float *local_A_load_black = local_A_raw_black + GET_LOCAL_INDEX_A(aI, aK);
  __local float *local_B_load_black = local_B_raw_black + GET_LOCAL_INDEX_B(bK, bJ);

  /* init local load & compute */
  __local float *local_A_load = local_A_load_red;
  __local float *local_B_load = local_B_load_red;
  __local float *local_A_comp = local_A_comp_black;
  __local float *local_B_comp = local_B_comp_black;

  /* load global -> local */
  local_A_load[0*TPI] = A[0*TPI];
  local_A_load[1*TPI] = A[1*TPI];
  local_A_load[2*TPI] = A[2*TPI];
  local_B_load[0*TPI] = B[0*TPI];
  local_B_load[1*TPI] = B[1*TPI];
  local_B_load[2*TPI] = B[2*TPI];
  barrier(CLK_LOCAL_MEM_FENCE);
  

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {

    /* swap local red/black */
    local_A_load = (local_A_load == local_A_load_red) ? local_A_load_black : local_A_load_red;
    local_B_load = (local_B_load == local_B_load_red) ? local_B_load_black : local_B_load_red;

    /* load global -> local */
    A += strideAK*NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
    local_A_load[0*TPI] = A[0*TPI];
    local_A_load[1*TPI] = A[1*TPI];
    local_A_load[2*TPI] = A[2*TPI];
    local_B_load[0*TPI] = B[0*TPI];
    local_B_load[1*TPI] = B[1*TPI];
    local_B_load[2*TPI] = B[2*TPI];

    /* do mads */
    unsigned int offA = localIdx0I; // d0
    unsigned int offB = localIdx1J; // d1
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    barrier(CLK_LOCAL_MEM_FENCE);

  } while (--sumIterK > 1);

  /* last local compute */
  local_A_comp = (local_A_comp == local_A_comp_red) ? local_A_comp_black : local_A_comp_red;
  local_B_comp = (local_A_comp == local_A_comp_red) ? local_A_comp_black : local_A_comp_red;

  /* do mads */
  unsigned int offA = localIdx0I; // d0
  unsigned int offB = localIdx1J; // d1
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  barrier(CLK_LOCAL_MEM_FENCE);

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


// NT 6x6 micro tile
// unroll 8
// double-buffer load global->lds
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     6
#define MICRO_TILE_1J     6
#define MACRO_TILE_0I     96
#define MACRO_TILE_1J     96
#define NUM_UNROLL_ITER   8
// threads-per-iteration (loading unroll)
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
  rA[0] = local_A_comp[offA + 0*WG_DIM_0I]; \
  rA[1] = local_A_comp[offA + 1*WG_DIM_0I]; \
  rA[2] = local_A_comp[offA + 2*WG_DIM_0I]; \
  rA[3] = local_A_comp[offA + 3*WG_DIM_0I]; \
  rA[4] = local_A_comp[offA + 4*WG_DIM_0I]; \
  rA[5] = local_A_comp[offA + 5*WG_DIM_0I]; \
  rB[0] = local_B_comp[offB + 0*WG_DIM_1J]; \
  rB[1] = local_B_comp[offB + 1*WG_DIM_1J]; \
  rB[2] = local_B_comp[offB + 2*WG_DIM_1J]; \
  rB[3] = local_B_comp[offB + 3*WG_DIM_1J]; \
  rB[4] = local_B_comp[offB + 4*WG_DIM_1J]; \
  rB[5] = local_B_comp[offB + 5*WG_DIM_1J]; \
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
  __local float local_A_raw_red[NUM_UNROLL_ITER*MACRO_TILE_0I];
  __local float local_B_raw_red[NUM_UNROLL_ITER*MACRO_TILE_1J];

  __local float local_A_raw_black[NUM_UNROLL_ITER*MACRO_TILE_0I];
  __local float local_B_raw_black[NUM_UNROLL_ITER*MACRO_TILE_1J];

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

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);

  /* compute pointers (no offset) */
  __local float *local_A_comp_red   = local_A_raw_red;
  __local float *local_B_comp_red   = local_B_raw_red;
  __local float *local_A_comp_black = local_A_raw_black;
  __local float *local_B_comp_black = local_B_raw_black;

  /* load pointers (offset) */
  __local float *local_A_load_red   = local_A_raw_red   + GET_LOCAL_INDEX_A(aI, aK);
  __local float *local_B_load_red   = local_B_raw_red   + GET_LOCAL_INDEX_B(bK, bJ);
  __local float *local_A_load_black = local_A_raw_black + GET_LOCAL_INDEX_A(aI, aK);
  __local float *local_B_load_black = local_B_raw_black + GET_LOCAL_INDEX_B(bK, bJ);

  /* init local load & compute */
  __local float *local_A_load = local_A_load_red;
  __local float *local_B_load = local_B_load_red;
  __local float *local_A_comp = local_A_comp_black;
  __local float *local_B_comp = local_B_comp_black;

  /* load global -> local */
  local_A_load[0*TPI] = A[0*TPI];
  local_A_load[1*TPI] = A[1*TPI];
  local_A_load[2*TPI] = A[2*TPI];
  local_B_load[0*TPI] = B[0*TPI];
  local_B_load[1*TPI] = B[1*TPI];
  local_B_load[2*TPI] = B[2*TPI];
  barrier(CLK_LOCAL_MEM_FENCE);
  

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {

    /* swap local red/black */
    local_A_load = (local_A_load == local_A_load_red) ? local_A_load_black : local_A_load_red;
    local_B_load = (local_B_load == local_B_load_red) ? local_B_load_black : local_B_load_red;

    /* load global -> local */
    A += strideAK*NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
    local_A_load[0*TPI] = A[0*TPI];
    local_A_load[1*TPI] = A[1*TPI];
    local_A_load[2*TPI] = A[2*TPI];
    local_B_load[0*TPI] = B[0*TPI];
    local_B_load[1*TPI] = B[1*TPI];
    local_B_load[2*TPI] = B[2*TPI];

    /* do mads */
    unsigned int offA = localIdx0I; // d0
    unsigned int offB = localIdx1J; // d1
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    barrier(CLK_LOCAL_MEM_FENCE);

  } while (--sumIterK > 1);

  /* last local compute */
  local_A_comp = (local_A_comp == local_A_comp_red) ? local_A_comp_black : local_A_comp_red;
  local_B_comp = (local_A_comp == local_A_comp_red) ? local_A_comp_black : local_A_comp_red;

  /* do mads */
  unsigned int offA = localIdx0I; // d0
  unsigned int offB = localIdx1J; // d1
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  MICRO_TILE
  barrier(CLK_LOCAL_MEM_FENCE);

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

// NT 6x6 micro tile
// float4 - can't get to work
// unroll 16
#if 0
const char * kernelSource_NT = R"(

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




// NT 4x4 micro tile
// float2
#if 0
const char * kernelSource_NT = R"(

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


// NT 6x6 micro tile
// float2
#if 0
const char * kernelSource_NT = R"(

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



// NT 8x8 micro tile
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_DIM_0I         16
#define WG_DIM_1J         16
#define MICRO_TILE_0I     8
#define MICRO_TILE_1J     8
#define MACRO_TILE_0I     128
#define MACRO_TILE_1J     128
#define NUM_UNROLL_ITER   8
#define TPI (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER)


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

  unsigned int aI = localSerial%TPI;
  unsigned int aK = localSerial/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MACRO_TILE_0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MACRO_TILE_1J, bK);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + GET_LOCAL_INDEX_B(bK, bJ);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */

    /* load global -> local */
    lA[0*TPI] = A[0*TPI];
    lA[1*TPI] = A[1*TPI];
    lA[2*TPI] = A[2*TPI];
    lA[3*TPI] = A[3*TPI];
#if NUM_UNROLL_ITER>8
    lA[4*TPI] = A[4*TPI];
    lA[5*TPI] = A[5*TPI];
    lA[6*TPI] = A[6*TPI];
    lA[7*TPI] = A[7*TPI];
#endif

    lB[0*TPI] = B[0*TPI];
    lB[1*TPI] = B[1*TPI];
    lB[2*TPI] = B[2*TPI];
    lB[3*TPI] = B[3*TPI];
#if NUM_UNROLL_ITER>8
    lB[4*TPI] = B[4*TPI];
    lB[5*TPI] = B[5*TPI];
    lB[6*TPI] = B[6*TPI];
    lB[7*TPI] = B[7*TPI];
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


// NT 6x6 micro tile
// unroll 8
#if 0
const char * kernelSource_NT = R"(

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
    lA[0*TPI] = A[0*TPI+0*strideAK];
    lA[1*TPI] = A[1*TPI+0*strideAK];
    lA[2*TPI] = A[2*TPI+0*strideAK];
#if NUM_UNROLL_ITER>8
    lA[3*TPI] = A[3*TPI+0*strideAK];
    lA[4*TPI] = A[4*TPI+0*strideAK];
    lA[5*TPI] = A[5*TPI+0*strideAK];
#endif

    lB[0*TPI] = B[0*TPI+0*strideBK];
    lB[1*TPI] = B[1*TPI+0*strideBK];
    lB[2*TPI] = B[2*TPI+0*strideBK];
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


// NT original 0
// unroll 16
// both Cobalt & original validate
#if 0


const char * kernelSource_NT = R"(

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

// NT Cobalt branched kernel
#if 0
const char * kernelSource_NT = "\n"
"/* CT_SSSSS_Cij_Sk_Aik_Bjk_i16b6f_j16b6f_k8_O2 */\n"
"\n"
"/* tile parameters */\n"
"#define WG_DIM_0I          16\n"
"#define WG_DIM_1J          16\n"
"#define MICRO_TILE_0I  6\n"
"#define MICRO_TILE_1J  6\n"
"#define MACRO_TILE_0I  96\n"
"#define MACRO_TILE_1J  96\n"
"#define NUM_UNROLL_ITER  8\n"
"\n"
"\n"
"/* global memory indices */\n"
"#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )\n"
"#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )\n"
"#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )\n"
"\n"
"/* global tile indices being loaded */\n"
"/* fast read */\n"
"#define globalIdxA0I(LID) (groupIdx0I*MACRO_TILE_0I + (localSerial+(LID)*WG_DIM_0I*WG_DIM_1J)/NUM_UNROLL_ITER)\n"
"#define globalIdxAK(LID) (localSerial%NUM_UNROLL_ITER)\n"
"/* fast read */\n"
"#define globalIdxBK(LID) ((localSerial+(LID)*WG_DIM_0I*WG_DIM_1J)/MACRO_TILE_1J)\n"
"#define globalIdxB1J(LID) (groupIdx1J*MACRO_TILE_1J + (localSerial+(LID)*WG_DIM_0I*WG_DIM_1J)%MACRO_TILE_1J)\n"
"\n"
"/* global non-tile indices being loaded */\n"
"\n"
"\n"
"/* local memory indices */\n"
"#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_0I) )\n"
"#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_1J) )\n"
"\n"
"/* local indices being written */\n"
"#define localA0I (localSerial / NUM_UNROLL_ITER)\n"
"#define localAK (localSerial % NUM_UNROLL_ITER)\n"
"#define localAStride (WG_DIM_0I*WG_DIM_1J/NUM_UNROLL_ITER)\n"
"#define localB1J ( localSerial / MACRO_TILE_1J )\n"
"#define localBK ( localSerial % MACRO_TILE_1J )\n"
"#define localBStride  (WG_DIM_0I*WG_DIM_1J)\n"
"\n"
"/* data types */\n"
"#define DATA_TYPE_STR_A float\n"
"#define DATA_TYPE_STR_B float\n"
"#define DATA_TYPE_STR_C float\n"
"#define DATA_TYPE_STR_ALPHA float\n"
"#define DATA_TYPE_STR_BETA float\n"
"#define FMA(A,B,DST) mad(A,B,DST)\n"
"#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);\n"
"#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);\n"
"\n"
"/* 6x6 micro-tile */\n"
"#define MICRO_TILE \\\n"
"  rA[0] = localA[offA + 0*WG_DIM_0I]; \\\n"
"  rA[1] = localA[offA + 1*WG_DIM_0I]; \\\n"
"  rA[2] = localA[offA + 2*WG_DIM_0I]; \\\n"
"  rA[3] = localA[offA + 3*WG_DIM_0I]; \\\n"
"  rA[4] = localA[offA + 4*WG_DIM_0I]; \\\n"
"  rA[5] = localA[offA + 5*WG_DIM_0I]; \\\n"
"  rB[0] = localB[offB + 0*WG_DIM_1J]; \\\n"
"  rB[1] = localB[offB + 1*WG_DIM_1J]; \\\n"
"  rB[2] = localB[offB + 2*WG_DIM_1J]; \\\n"
"  rB[3] = localB[offB + 3*WG_DIM_1J]; \\\n"
"  rB[4] = localB[offB + 4*WG_DIM_1J]; \\\n"
"  rB[5] = localB[offB + 5*WG_DIM_1J]; \\\n"
"  offA += MACRO_TILE_0I; \\\n"
"  offB += MACRO_TILE_1J; \\\n"
"  TYPE_MAD(rA[0],rB[0],rC[0][0]); \\\n"
"  TYPE_MAD(rA[0],rB[1],rC[0][1]); \\\n"
"  TYPE_MAD(rA[0],rB[2],rC[0][2]); \\\n"
"  TYPE_MAD(rA[0],rB[3],rC[0][3]); \\\n"
"  TYPE_MAD(rA[0],rB[4],rC[0][4]); \\\n"
"  TYPE_MAD(rA[0],rB[5],rC[0][5]); \\\n"
"  TYPE_MAD(rA[1],rB[0],rC[1][0]); \\\n"
"  TYPE_MAD(rA[1],rB[1],rC[1][1]); \\\n"
"  TYPE_MAD(rA[1],rB[2],rC[1][2]); \\\n"
"  TYPE_MAD(rA[1],rB[3],rC[1][3]); \\\n"
"  TYPE_MAD(rA[1],rB[4],rC[1][4]); \\\n"
"  TYPE_MAD(rA[1],rB[5],rC[1][5]); \\\n"
"  TYPE_MAD(rA[2],rB[0],rC[2][0]); \\\n"
"  TYPE_MAD(rA[2],rB[1],rC[2][1]); \\\n"
"  TYPE_MAD(rA[2],rB[2],rC[2][2]); \\\n"
"  TYPE_MAD(rA[2],rB[3],rC[2][3]); \\\n"
"  TYPE_MAD(rA[2],rB[4],rC[2][4]); \\\n"
"  TYPE_MAD(rA[2],rB[5],rC[2][5]); \\\n"
"  TYPE_MAD(rA[3],rB[0],rC[3][0]); \\\n"
"  TYPE_MAD(rA[3],rB[1],rC[3][1]); \\\n"
"  TYPE_MAD(rA[3],rB[2],rC[3][2]); \\\n"
"  TYPE_MAD(rA[3],rB[3],rC[3][3]); \\\n"
"  TYPE_MAD(rA[3],rB[4],rC[3][4]); \\\n"
"  TYPE_MAD(rA[3],rB[5],rC[3][5]); \\\n"
"  TYPE_MAD(rA[4],rB[0],rC[4][0]); \\\n"
"  TYPE_MAD(rA[4],rB[1],rC[4][1]); \\\n"
"  TYPE_MAD(rA[4],rB[2],rC[4][2]); \\\n"
"  TYPE_MAD(rA[4],rB[3],rC[4][3]); \\\n"
"  TYPE_MAD(rA[4],rB[4],rC[4][4]); \\\n"
"  TYPE_MAD(rA[4],rB[5],rC[4][5]); \\\n"
"  TYPE_MAD(rA[5],rB[0],rC[5][0]); \\\n"
"  TYPE_MAD(rA[5],rB[1],rC[5][1]); \\\n"
"  TYPE_MAD(rA[5],rB[2],rC[5][2]); \\\n"
"  TYPE_MAD(rA[5],rB[3],rC[5][3]); \\\n"
"  TYPE_MAD(rA[5],rB[4],rC[5][4]); \\\n"
"  TYPE_MAD(rA[5],rB[5],rC[5][5]); \\\n"
"  mem_fence(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"/* preprocessor definitions of kernel arguments*/\n"
"#define strideC0I 1\n"
"#define strideA0I 1\n"
"#define strideB1J 1\n"
"\n"
"\n"
"__attribute__((reqd_work_group_size(WG_DIM_0I,WG_DIM_1J,1)))\n"
"__kernel void CT_SSSSS_Cij_Sk_Aik_Bjk_i16b6f_j16b6f_k8_O2(\n"
"  __global float       *          C,\n"
"  __global float const * restrict A,\n"
"  __global float const * restrict B,\n"
"  float const alpha,\n"
"  float const beta,\n"
"  unsigned int const strideC1J,\n"
"  unsigned int const strideAK,\n"
"  unsigned int const strideBK,\n"
"  unsigned int const size0I,\n"
"  unsigned int const size1J,\n"
"  unsigned int const sizeK ) {\n"
"\n"
"  /* allocate registers */\n"
"  DATA_TYPE_STR_C rC[MICRO_TILE_0I][MICRO_TILE_1J] = {{0}};\n"
"  DATA_TYPE_STR_A rA[MICRO_TILE_0I];\n"
"  DATA_TYPE_STR_B rB[MICRO_TILE_1J];\n"
"\n"
"  /* allocate local memory */\n"
"  __local DATA_TYPE_STR_A localA[NUM_UNROLL_ITER*MACRO_TILE_0I];\n"
"  __local DATA_TYPE_STR_B localB[NUM_UNROLL_ITER*MACRO_TILE_1J];\n"
"\n"
"  /* c indices */\n"
"  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA\n"
"  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB\n"
"  unsigned int localIdx0I = get_local_id(0); // d0\n"
"  unsigned int localIdx1J = get_local_id(1); // d1\n"
"  unsigned int localSerial = localIdx0I + localIdx1J*WG_DIM_0I;\n"
"\n"
"  /* which global Cij index */\n"
"  unsigned int globalIdxC1J = groupIdx1J*MACRO_TILE_1J + localIdx1J;\n"
"  unsigned int globalIdxC0I = groupIdx0I*MACRO_TILE_0I + localIdx0I;\n"
"  /* iterate over all summation indices */\n"
"  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;\n"
"  do {\n"
"    __local DATA_TYPE_STR_A *lA = localA + GET_LOCAL_INDEX_A(localA0I, localAK);\n"
"    __local DATA_TYPE_STR_B *lB = localB + GET_LOCAL_INDEX_B(localB1J, localBK);\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    /* load global -> local */\n"
"    lA[ 0*localAStride ] = ( globalIdxA0I(0) >= size0I) ? (float)(0.0) : A[ GET_GLOBAL_INDEX_A( globalIdxA0I(0), globalIdxAK(0) ) ];\n"
"    lA[ 1*localAStride ] = ( globalIdxA0I(1) >= size0I) ? (float)(0.0) : A[ GET_GLOBAL_INDEX_A( globalIdxA0I(1), globalIdxAK(1) ) ];\n"
"    lA[ 2*localAStride ] = ( globalIdxA0I(2) >= size0I) ? (float)(0.0) : A[ GET_GLOBAL_INDEX_A( globalIdxA0I(2), globalIdxAK(2) ) ];\n"
"    lB[ 0*localBStride ] = ( globalIdxB1J(0) >= size1J) ? (float)(0.0) : B[ GET_GLOBAL_INDEX_B( globalIdxB1J(0), globalIdxBK(0) ) ];\n"
"    lB[ 1*localBStride ] = ( globalIdxB1J(1) >= size1J) ? (float)(0.0) : B[ GET_GLOBAL_INDEX_B( globalIdxB1J(1), globalIdxBK(1) ) ];\n"
"    lB[ 2*localBStride ] = ( globalIdxB1J(2) >= size1J) ? (float)(0.0) : B[ GET_GLOBAL_INDEX_B( globalIdxB1J(2), globalIdxBK(2) ) ];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    unsigned int offA = localIdx0I; // d0\n"
"    unsigned int offB = localIdx1J; // d1\n"
"\n"
"    /* do mads */\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    A += strideAK*NUM_UNROLL_ITER;\n"
"    B += strideBK*NUM_UNROLL_ITER;\n"
"  } while (--sumIterK > 0);\n"
"\n"
"\n"
"  /* write global C */\n"
"  if (globalIdxC0I + 0*WG_DIM_0I < size0I) {  if (globalIdxC1J + 0*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[0][0], beta) } }\n"
"  if (globalIdxC0I + 0*WG_DIM_0I < size0I) {  if (globalIdxC1J + 1*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[0][1], beta) } }\n"
"  if (globalIdxC0I + 0*WG_DIM_0I < size0I) {  if (globalIdxC1J + 2*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[0][2], beta) } }\n"
"  if (globalIdxC0I + 0*WG_DIM_0I < size0I) {  if (globalIdxC1J + 3*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[0][3], beta) } }\n"
"  if (globalIdxC0I + 0*WG_DIM_0I < size0I) {  if (globalIdxC1J + 4*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[0][4], beta) } }\n"
"  if (globalIdxC0I + 0*WG_DIM_0I < size0I) {  if (globalIdxC1J + 5*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[0][5], beta) } }\n"
"  if (globalIdxC0I + 1*WG_DIM_0I < size0I) {  if (globalIdxC1J + 0*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[1][0], beta) } }\n"
"  if (globalIdxC0I + 1*WG_DIM_0I < size0I) {  if (globalIdxC1J + 1*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[1][1], beta) } }\n"
"  if (globalIdxC0I + 1*WG_DIM_0I < size0I) {  if (globalIdxC1J + 2*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[1][2], beta) } }\n"
"  if (globalIdxC0I + 1*WG_DIM_0I < size0I) {  if (globalIdxC1J + 3*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[1][3], beta) } }\n"
"  if (globalIdxC0I + 1*WG_DIM_0I < size0I) {  if (globalIdxC1J + 4*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[1][4], beta) } }\n"
"  if (globalIdxC0I + 1*WG_DIM_0I < size0I) {  if (globalIdxC1J + 5*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[1][5], beta) } }\n"
"  if (globalIdxC0I + 2*WG_DIM_0I < size0I) {  if (globalIdxC1J + 0*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[2][0], beta) } }\n"
"  if (globalIdxC0I + 2*WG_DIM_0I < size0I) {  if (globalIdxC1J + 1*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[2][1], beta) } }\n"
"  if (globalIdxC0I + 2*WG_DIM_0I < size0I) {  if (globalIdxC1J + 2*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[2][2], beta) } }\n"
"  if (globalIdxC0I + 2*WG_DIM_0I < size0I) {  if (globalIdxC1J + 3*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[2][3], beta) } }\n"
"  if (globalIdxC0I + 2*WG_DIM_0I < size0I) {  if (globalIdxC1J + 4*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[2][4], beta) } }\n"
"  if (globalIdxC0I + 2*WG_DIM_0I < size0I) {  if (globalIdxC1J + 5*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[2][5], beta) } }\n"
"  if (globalIdxC0I + 3*WG_DIM_0I < size0I) {  if (globalIdxC1J + 0*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[3][0], beta) } }\n"
"  if (globalIdxC0I + 3*WG_DIM_0I < size0I) {  if (globalIdxC1J + 1*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[3][1], beta) } }\n"
"  if (globalIdxC0I + 3*WG_DIM_0I < size0I) {  if (globalIdxC1J + 2*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[3][2], beta) } }\n"
"  if (globalIdxC0I + 3*WG_DIM_0I < size0I) {  if (globalIdxC1J + 3*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[3][3], beta) } }\n"
"  if (globalIdxC0I + 3*WG_DIM_0I < size0I) {  if (globalIdxC1J + 4*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[3][4], beta) } }\n"
"  if (globalIdxC0I + 3*WG_DIM_0I < size0I) {  if (globalIdxC1J + 5*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[3][5], beta) } }\n"
"  if (globalIdxC0I + 4*WG_DIM_0I < size0I) {  if (globalIdxC1J + 0*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[4][0], beta) } }\n"
"  if (globalIdxC0I + 4*WG_DIM_0I < size0I) {  if (globalIdxC1J + 1*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[4][1], beta) } }\n"
"  if (globalIdxC0I + 4*WG_DIM_0I < size0I) {  if (globalIdxC1J + 2*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[4][2], beta) } }\n"
"  if (globalIdxC0I + 4*WG_DIM_0I < size0I) {  if (globalIdxC1J + 3*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[4][3], beta) } }\n"
"  if (globalIdxC0I + 4*WG_DIM_0I < size0I) {  if (globalIdxC1J + 4*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[4][4], beta) } }\n"
"  if (globalIdxC0I + 4*WG_DIM_0I < size0I) {  if (globalIdxC1J + 5*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[4][5], beta) } }\n"
"  if (globalIdxC0I + 5*WG_DIM_0I < size0I) {  if (globalIdxC1J + 0*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 0*WG_DIM_1J) ], alpha, rC[5][0], beta) } }\n"
"  if (globalIdxC0I + 5*WG_DIM_0I < size0I) {  if (globalIdxC1J + 1*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 1*WG_DIM_1J) ], alpha, rC[5][1], beta) } }\n"
"  if (globalIdxC0I + 5*WG_DIM_0I < size0I) {  if (globalIdxC1J + 2*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 2*WG_DIM_1J) ], alpha, rC[5][2], beta) } }\n"
"  if (globalIdxC0I + 5*WG_DIM_0I < size0I) {  if (globalIdxC1J + 3*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 3*WG_DIM_1J) ], alpha, rC[5][3], beta) } }\n"
"  if (globalIdxC0I + 5*WG_DIM_0I < size0I) {  if (globalIdxC1J + 4*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 4*WG_DIM_1J) ], alpha, rC[5][4], beta) } }\n"
"  if (globalIdxC0I + 5*WG_DIM_0I < size0I) {  if (globalIdxC1J + 5*WG_DIM_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_DIM_0I, globalIdxC1J + 5*WG_DIM_1J) ], alpha, rC[5][5], beta) } }\n"
"\n"
"}\n"
"";


#endif