/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

// split-k
#if 1

const char * kernelSource_NT = R"(
 /* tile parameters */
#define NUM_THREADS   256
#define NUM_SUBGROUPS  4
#define SG0            8
#define SG1            8
#define UT0            4
#define UT1            4
#define UNROLL         8
#define PAD            4
#define VECTOR_WIDTH   4

#define MT0I   (SG0*UT0)
#define MT1J   (SG1*UT1)

 /* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*1 + (IDX1J)*strideC1J/4 )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK)  ( (IDX0I)*1 + (IDXK) *strideAK /4 )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK)  ( (IDX1J)*1 + (IDXK) *strideBK /4 )

 /* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

 /* data types */
#define DATA_TYPE_STR_A float
#define DATA_TYPE_STR_B float
#define DATA_TYPE_STR_C float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA float
#define FMA(A,B,DST) mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

#define rA0 rA.x
#define rA1 rA.y
#define rA2 rA.z
#define rA3 rA.w

#define rB0 rB.x
#define rB1 rB.y
#define rB2 rB.z
#define rB3 rB.w

#define rC00 rC[0].x
#define rC01 rC[1].x
#define rC02 rC[2].x
#define rC03 rC[3].x
#define rC10 rC[0].y
#define rC11 rC[1].y
#define rC12 rC[2].y
#define rC13 rC[3].y
#define rC20 rC[0].z
#define rC21 rC[1].z
#define rC22 rC[2].z
#define rC23 rC[3].z
#define rC30 rC[0].w
#define rC31 rC[1].w
#define rC32 rC[2].w
#define rC33 rC[3].w

 /* 6x6 micro-tile */
#define MICRO_TILE \
  rA = localReadA[offA]; \
  rB = localReadB[offB]; \
  offA += NUM_SUBGROUPS*(MT0I+PAD); \
  offB += NUM_SUBGROUPS*(MT1J+PAD); \
  TYPE_MAD(rA0,rB0,rC00); \
  TYPE_MAD(rA0,rB1,rC01); \
  TYPE_MAD(rA0,rB2,rC02); \
  TYPE_MAD(rA0,rB3,rC03); \
  TYPE_MAD(rA1,rB0,rC10); \
  TYPE_MAD(rA1,rB1,rC11); \
  TYPE_MAD(rA1,rB2,rC12); \
  TYPE_MAD(rA1,rB3,rC13); \
  TYPE_MAD(rA2,rB0,rC20); \
  TYPE_MAD(rA2,rB1,rC21); \
  TYPE_MAD(rA2,rB2,rC22); \
  TYPE_MAD(rA2,rB3,rC23); \
  TYPE_MAD(rA3,rB0,rC30); \
  TYPE_MAD(rA3,rB1,rC31); \
  TYPE_MAD(rA3,rB2,rC32); \
  TYPE_MAD(rA3,rB3,rC33); \
  mem_fence(CLK_LOCAL_MEM_FENCE);

 /* preprocessor definitions of kernel arguments*/
#define strideC0I 1
#define strideA0I 1
#define strideB1J 1

__attribute__((reqd_work_group_size(NUM_THREADS, 1, 1)))
__kernel void gemm_kernel(
  __global float4       *          C,
  __global float4 const * restrict A,
  __global float4 const * restrict B,
  float const alpha,
  float const beta,
  unsigned int const strideC1J,
  unsigned int const strideAK,
  unsigned int const strideBK,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK) {

  /* allocate registers */
  float4 rC[UT1] = { 0 };
  float4 rA;
  float4 rB;

  /* allocate local memory */
  __local float4 localA[UNROLL*NUM_SUBGROUPS*(MT0I + PAD)/VECTOR_WIDTH];
  __local float4 localB[UNROLL*NUM_SUBGROUPS*(MT1J + PAD)/VECTOR_WIDTH];

  /* indices */
  unsigned int serial = get_local_id(0);
  unsigned int tidS = serial % NUM_SUBGROUPS;
  unsigned int tid0 = (serial / NUM_SUBGROUPS) % SG0;
  unsigned int tid1 = (serial / NUM_SUBGROUPS) / SG0;
  unsigned int groupIdx0I = get_group_id(0);
  unsigned int groupIdx1J = get_group_id(1);

  /* read lds */
  unsigned int localReadOffsetA = tidS + tid0*UT0;
  unsigned int localReadOffsetB = tidS + tid1*UT1;
  __local float4 *localReadA = localA + localReadOffsetA;
  __local float4 *localReadB = localB + localReadOffsetB;

  /* write lds ? */
  unsigned int localWriteOffsetA = serial;
  unsigned int localWriteOffsetB = serial;
  __local float4 *localWriteA = localA + localWriteOffsetA;
  __local float4 *localWriteB = localB + localWriteOffsetB;

  /* read global ? */
  unsigned int readGlobalOffsetA = groupIdx0I*MT0I + tid0 + strideAK*(tidS + tid1*NUM_SUBGROUPS);
  unsigned int readGlobalOffsetB = groupIdx1J*MT1J + tid0 + strideAK*(tidS + tid1*NUM_SUBGROUPS);
  __global float4 const * globalReadA = A + readGlobalOffsetA;
  __global float4 const * globalReadB = B + readGlobalOffsetB;

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / (NUM_SUBGROUPS*UNROLL);
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */
    localWriteA[0] = globalReadA[0];
    localWriteB[0] = globalReadB[0];

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int offA = 0;
    unsigned int offB = 0;

    /* do mads */
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE

    globalReadA += strideAK*(NUM_SUBGROUPS*UNROLL);
    globalReadB += strideBK*(NUM_SUBGROUPS*UNROLL);
  } while (--sumIterK > 0);

  /* reduction in LDS */
  __local float4 localC[MT0I / VECTOR_WIDTH][MT1J];

  if (tidS == 0) {
    localC[tid0][tid1*4+0] = rC[0];
    localC[tid0][tid1*4+1] = rC[1];
    localC[tid0][tid1*4+2] = rC[2];
    localC[tid0][tid1*4+3] = rC[3];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int s = 1; s < NUM_SUBGROUPS; s++) {
    if (tidS == s) {
      localC[tid0][tid1*4+0] += rC[0];
      localC[tid0][tid1*4+1] += rC[1];
      localC[tid0][tid1*4+2] += rC[2];
      localC[tid0][tid1*4+3] += rC[3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // all subgroups have same
  rC[0] = localC[tid0][tid1*4+tidS];

  /* which global Cij index */
  unsigned int writeGlobalC0I = groupIdx0I*MT0I + serial%8;
  unsigned int writeGlobalC1J = groupIdx1J*MT1J + serial/8;

 
  long globalIdx = GET_GLOBAL_INDEX_C(writeGlobalC0I, writeGlobalC1J);
  rA = C[globalIdx];
  rC[0] = alpha*rA + beta*rC[0];
  C[globalIdx] = rC[0];
  // todo

  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 0 * WG_0I, globalIdxC1J + 0 * WG_1J)], alpha, rC00, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 0 * WG_0I, globalIdxC1J + 1 * WG_1J)], alpha, rC01, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 0 * WG_0I, globalIdxC1J + 2 * WG_1J)], alpha, rC02, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 0 * WG_0I, globalIdxC1J + 3 * WG_1J)], alpha, rC03, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 1 * WG_0I, globalIdxC1J + 0 * WG_1J)], alpha, rC10, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 1 * WG_0I, globalIdxC1J + 1 * WG_1J)], alpha, rC11, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 1 * WG_0I, globalIdxC1J + 2 * WG_1J)], alpha, rC12, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 1 * WG_0I, globalIdxC1J + 3 * WG_1J)], alpha, rC13, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 2 * WG_0I, globalIdxC1J + 0 * WG_1J)], alpha, rC20, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 2 * WG_0I, globalIdxC1J + 1 * WG_1J)], alpha, rC21, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 2 * WG_0I, globalIdxC1J + 2 * WG_1J)], alpha, rC22, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 2 * WG_0I, globalIdxC1J + 3 * WG_1J)], alpha, rC23, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 3 * WG_0I, globalIdxC1J + 0 * WG_1J)], alpha, rC30, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 3 * WG_0I, globalIdxC1J + 1 * WG_1J)], alpha, rC31, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 3 * WG_0I, globalIdxC1J + 2 * WG_1J)], alpha, rC32, beta)
  // TYPE_MAD_WRITE(C[GET_GLOBAL_INDEX_C(globalIdxC0I + 3 * WG_0I, globalIdxC1J + 3 * WG_1J)], alpha, rC33, beta)

};

)";

#endif



 /*
 NT - no branches
 */
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
#if 1
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
#else
  unsigned int groupSerial = get_group_id(0)*get_num_groups(1) + get_group_id(1);
  unsigned int groupIdx0I = groupSerial % get_num_groups(0);
  unsigned int groupIdx1J = groupSerial / get_num_groups(0);
#endif



  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

  unsigned int aI = localSerial%TPI;
  unsigned int aK = localSerial/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

  unsigned int nonMultipleSize0 = size0I % MT0I;
  unsigned int nonMultipleSize1 = size1J % MT1J;
  //if (localSerial==0) printf("wg[%u][%u] nm=%u,%u\n", groupIdx0I, groupIdx1J, nonMultipleSize0, nonMultipleSize1);
  bool lastGroup0 = groupIdx0I == get_num_groups(0)-1;
  bool lastGroup1 = groupIdx1J == get_num_groups(1)-1;
  //if (localSerial==0) printf("wg[%u][%u] last=%u,%u\n", groupIdx0I, groupIdx1J, lastGroup0, lastGroup1);
  unsigned int groupShift0 = lastGroup0 && nonMultipleSize0 ? MT0I-nonMultipleSize0 : 0;
  unsigned int groupShift1 = lastGroup1 && nonMultipleSize1 ? MT1J-nonMultipleSize1 : 0;
  //if (localSerial==0) printf("wg[%u][%u] last=%u,%u\n", groupIdx0I, groupIdx1J, groupShift0, lastGroup1);
  unsigned int shiftA = GET_GLOBAL_INDEX_A(groupShift0, 0);
  unsigned int shiftB = GET_GLOBAL_INDEX_B(groupShift1, 0);
  //if (localSerial==0) printf("wg[%u][%u] shift=%u,%u\n", groupIdx0I, groupIdx1J, shiftA, shiftB);
  A -= shiftA;
  B -= shiftB;

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

    lB[0*TPI] = B[0*TPI+0*strideBK];
    lB[1*TPI] = B[1*TPI+0*strideBK];
    lB[2*TPI] = B[2*TPI+0*strideBK];

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

    A += strideAK*NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

#if 0
  /* write global C */
  unsigned int iStart = 0;
  unsigned int jStart = 0;
  if (lastGroup0 && nonMultipleSize0) {
    globalIdxC0I -= MT0I-nonMultipleSize0;
    iStart = 6-1-nonMultipleSize0/16;
    if ( localIdx0I < 6 - nonMultipleSize0%6 ) {
      iStart++;
    }
  }
  if (lastGroup1 && nonMultipleSize1) {
    globalIdxC1J -= MT1J - nonMultipleSize1;
    jStart = 6-1-nonMultipleSize1/16;
    if ( localIdx1J < 6 - nonMultipleSize1%6 ) {
      jStart++;
    }
  }
  //if (groupIdx0I==1 && groupIdx1J==0) printf("t[%u][%u] starts %u, %u @ %u, %u\n", localIdx0I, localIdx1J, iStart, jStart, globalIdxC0I, globalIdxC1J);
  //if (groupIdx0I==1 && groupIdx1J==1) printf("t[%u][%u] starts %u, %u\n", localIdx0I, localIdx1J, iStart, jStart);


  /* write global C */
  unsigned int i = iStart;
  do {
    unsigned int j = jStart;
    do {
      TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + i*WG_0I, globalIdxC1J + j*WG_1J) ], alpha, rC[i][j], beta)
    } while (++j < 6);
  } while (++i < 6);


  //for (unsigned int i = iStart; i < 6; i++) {
  //  for (unsigned int j = jStart; j < 6; j++) {
  //    //unsigned int idx = GET_GLOBAL_INDEX_C( globalIdxC0I + i*WG_0I, globalIdxC1J + j*WG_1J);
  //    //if (groupIdx0I==1 && groupIdx1J==0 && i==0 && j==0) printf("t[%u][%u] i%u,j%u writing %f to %u\n", localIdx0I, localIdx1J, i, j, rC[i][j], idx);
  //    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + i*WG_0I, globalIdxC1J + j*WG_1J) ], alpha, rC[i][j], beta)
  //  }
  //}
#else
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)
#endif
};
)";
#endif


 
 
 
 /*

NT w/ fast branches
*/
/* CT_SSSSS_Cij_Sk_Aik_Bjk_i16b4f_j16b4f_nl1x1_k16_O2 */
#if 0
const char * kernelSource_NT = "\n"
"\n"
"/* tile parameters */\n"
"#define WG_0I  16\n"
"#define WG_1J  16\n"
"#define UT_0I   4\n"
"#define UT_1J   4\n"
"#define MT_0I  64\n"
"#define MT_1J  64\n"
"#define UNROLL 16\n"
"#define PAD     0\n"
"\n"
"/* num loads parallel and perpendicular to coalesced dimension */\n"
"#define NL_PARA_A 1\n"
"#define NL_PARA_B 1\n"
"#define NL_PERP_A 4\n"
"#define NL_PERP_B 4\n"
"\n"
"/* load size parallel and perpendicular to coalesced dimension */\n"
"#define LS_PARA_A 64\n"
"#define LS_PERP_A 4\n"
"#define LS_PARA_B 64\n"
"#define LS_PERP_B 4\n"
"\n"
"/* global memory indices */\n"
"#define GLOBAL_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )\n"
"#define GLOBAL_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )\n"
"#define GLOBAL_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )\n"
"\n"
"\n"
"/* data types */\n"
"#define TYPE_A     float\n"
"#define TYPE_B     float\n"
"#define TYPE_C     float\n"
"#define TYPE_ALPHA float\n"
"#define TYPE_BETA  float\n"
"#define MAD(A,B,DST) mad(A,B,DST)\n"
"\n"
"/* MADs */\n"
"#define TYPE_MAD(MULA,MULB,DST) DST = MAD(MULA,MULB,DST);\n"
"#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);\n"
"\n"
"/* 4x4 micro-tile */\n"
"#define MICRO_TILE \\\n"
"  rA[0] = localA[offA + 0*WG_0I]; \\\n"
"  rA[1] = localA[offA + 1*WG_0I]; \\\n"
"  rA[2] = localA[offA + 2*WG_0I]; \\\n"
"  rA[3] = localA[offA + 3*WG_0I]; \\\n"
"  rB[0] = localB[offB + 0*WG_1J]; \\\n"
"  rB[1] = localB[offB + 1*WG_1J]; \\\n"
"  rB[2] = localB[offB + 2*WG_1J]; \\\n"
"  rB[3] = localB[offB + 3*WG_1J]; \\\n"
"  offA += (MT_0I+PAD); \\\n"
"  offB += (MT_1J+PAD); \\\n"
"  TYPE_MAD(rA[0],rB[0],rC[0][0]); \\\n"
"  TYPE_MAD(rA[0],rB[1],rC[0][1]); \\\n"
"  TYPE_MAD(rA[0],rB[2],rC[0][2]); \\\n"
"  TYPE_MAD(rA[0],rB[3],rC[0][3]); \\\n"
"  TYPE_MAD(rA[1],rB[0],rC[1][0]); \\\n"
"  TYPE_MAD(rA[1],rB[1],rC[1][1]); \\\n"
"  TYPE_MAD(rA[1],rB[2],rC[1][2]); \\\n"
"  TYPE_MAD(rA[1],rB[3],rC[1][3]); \\\n"
"  TYPE_MAD(rA[2],rB[0],rC[2][0]); \\\n"
"  TYPE_MAD(rA[2],rB[1],rC[2][1]); \\\n"
"  TYPE_MAD(rA[2],rB[2],rC[2][2]); \\\n"
"  TYPE_MAD(rA[2],rB[3],rC[2][3]); \\\n"
"  TYPE_MAD(rA[3],rB[0],rC[3][0]); \\\n"
"  TYPE_MAD(rA[3],rB[1],rC[3][1]); \\\n"
"  TYPE_MAD(rA[3],rB[2],rC[3][2]); \\\n"
"  TYPE_MAD(rA[3],rB[3],rC[3][3]); \\\n"
"  mem_fence(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"/* preprocessor definitions of kernel arguments*/\n"
"#define strideC0I 1\n"
"#define strideA0I 1\n"
"#define strideB1J 1\n"
"\n"
"\n"
"/* kernel */\n"
"__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))\n"
"__kernel void CT_SSSSS_Cij_Sk_Aik_Bjk_i16b4f_j16b4f_nl1x1_k16_O2(\n"
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
"  TYPE_C rC[UT_0I][UT_1J] = {{0}};\n"
"  TYPE_A rA[UT_0I];\n"
"  TYPE_B rB[UT_1J];\n"
"\n"
"  /* allocate local memory */\n"
"  __local TYPE_A localA[UNROLL*(MT_0I+PAD)];\n"
"  __local TYPE_B localB[UNROLL*(MT_1J+PAD)];\n"
"\n"
"  /* c indices (group) */\n"
"  unsigned int g0I = get_group_id(0); // d0, tensorA\n"
"  unsigned int g1J = get_group_id(1); // d1, tensorB\n"
"\n"
"  /* c indices (local) */\n"
"  unsigned int l0I = get_local_id(0); // d0\n"
"  unsigned int l1J = get_local_id(1); // d1\n"
"  unsigned int loadSerial = l0I + l1J*WG_0I;\n"
"  unsigned int a0I = loadSerial%LS_PARA_A;\n"
"  unsigned int b1J = loadSerial%LS_PARA_B;\n"
"\n"
"  /* unrolled summation index */\n"
"  unsigned int aK = loadSerial/LS_PARA_A;\n"
"  unsigned int bK = loadSerial/LS_PARA_B;\n"
"\n"
"  /* other non-unrolled summation indices (all start at zero) */\n"
"\n"
"  /* where will this thread read from global memory */\n"
"  A += GLOBAL_A( a0I+g0I*MT_0I, aK );\n"
"  B += GLOBAL_B( b1J+g1J*MT_1J, bK );\n"
"\n"
"  /* where will this thread write to local memory */\n"
"  __local TYPE_A *lA = localA + a0I + aK*(MT_0I+PAD);\n"
"  __local TYPE_B *lB = localB + b1J + bK*(MT_1J+PAD);\n"
"\n"
"  /* iterate over summation indice(s) */\n"
"  unsigned int sumIterK = sizeK / UNROLL;\n"
"  do {\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    /* load A global -> local */\n"
"    float a0, a1, a2, a3, b0, b1, b2, b3;\n"
"    if ( a0I+g0I*MT_0I+0*LS_PARA_A < size0I) {\n"
"     a0 = A[ 0*LS_PARA_A + 0*LS_PERP_A*strideAK];\n"
"     a1 = A[ 0*LS_PARA_A + 1*LS_PERP_A*strideAK];\n"
"     a2 = A[ 0*LS_PARA_A + 2*LS_PERP_A*strideAK];\n"
"     a3 = A[ 0*LS_PARA_A + 3*LS_PERP_A*strideAK];\n"
"    } else {"
"      a0 = 0;\n"
"      a1 = 0;\n"
"      a2 = 0;\n"
"      a3 = 0;\n"
"    }"
"   lA[ 0*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = a0;\n"
"   lA[ 0*LS_PARA_A + 1*LS_PERP_A*(MT_0I+PAD) ] = a1;\n"
"   lA[ 0*LS_PARA_A + 2*LS_PERP_A*(MT_0I+PAD) ] = a2;\n"
"   lA[ 0*LS_PARA_A + 3*LS_PERP_A*(MT_0I+PAD) ] = a3;\n"
"\n"
"    /* load B global -> local */\n"
"    if ( b1J+g1J*MT_1J+0*LS_PARA_B < size1J) {\n"
"      b0 = B[ 0*LS_PARA_B + 0*LS_PERP_B*strideBK];\n"
"      b1 = B[ 0*LS_PARA_B + 1*LS_PERP_B*strideBK];\n"
"      b2 = B[ 0*LS_PARA_B + 2*LS_PERP_B*strideBK];\n"
"      b3 = B[ 0*LS_PARA_B + 3*LS_PERP_B*strideBK];\n"
"    } else {"
"      b0 = 0;\n"
"      b1 = 0;\n"
"      b2 = 0;\n"
"      b3 = 0;\n"
"    }"
"    lB[ 0*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = b0;\n"
"    lB[ 0*LS_PARA_B + 1*LS_PERP_B*(MT_1J+PAD) ] = b1;\n"
"    lB[ 0*LS_PARA_B + 2*LS_PERP_B*(MT_1J+PAD) ] = b2;\n"
"    lB[ 0*LS_PARA_B + 3*LS_PERP_B*(MT_1J+PAD) ] = b3;\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    unsigned int offA = l0I; // d0\n"
"    unsigned int offB = l1J; // d1\n"
"\n"
"    /* do fmas */\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"\n"
"    A += (long) strideAK*UNROLL;\n"
"    B += (long) strideBK*UNROLL;\n"
"  } while (--sumIterK > 0);\n"
"\n"
"  /* which global Cij index */\n"
"  unsigned int globalC1J = g1J*MT_1J + l1J;\n"
"  unsigned int globalC0I = g0I*MT_0I + l0I;\n"
"\n"
"  /* write global C */\n"
"  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[0][0], beta) } }\n"
"  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[0][1], beta) } }\n"
"  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[0][2], beta) } }\n"
"  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[0][3], beta) } }\n"
"  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[1][0], beta) } }\n"
"  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[1][1], beta) } }\n"
"  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[1][2], beta) } }\n"
"  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[1][3], beta) } }\n"
"  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[2][0], beta) } }\n"
"  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[2][1], beta) } }\n"
"  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[2][2], beta) } }\n"
"  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[2][3], beta) } }\n"
"  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[3][0], beta) } }\n"
"  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[3][1], beta) } }\n"
"  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[3][2], beta) } }\n"
"  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[3][3], beta) } }\n"
"\n"
"}\n";
#endif

/*

NT w/ fast branches
*/
/* CT_SSSSS_Cij_Sk_Aik_Bjk_i16b4f_j16b4f_nl1x1_k16_O2 */
#if 0
const char * kernelSource_NT = "\n"
"\n"
"/* tile parameters */\n"
"#define WG_0I  16\n"
"#define WG_1J  16\n"
"#define UT_0I   4\n"
"#define UT_1J   4\n"
"#define MT_0I  64\n"
"#define MT_1J  64\n"
"#define UNROLL 16\n"
"#define PAD     0\n"
"\n"
"/* num loads parallel and perpendicular to coalesced dimension */\n"
"#define NL_PARA_A 1\n"
"#define NL_PARA_B 1\n"
"#define NL_PERP_A 4\n"
"#define NL_PERP_B 4\n"
"\n"
"/* load size parallel and perpendicular to coalesced dimension */\n"
"#define LS_PARA_A 64\n"
"#define LS_PERP_A 4\n"
"#define LS_PARA_B 64\n"
"#define LS_PERP_B 4\n"
"\n"
"/* global memory indices */\n"
"#define GLOBAL_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )\n"
"#define GLOBAL_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )\n"
"#define GLOBAL_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )\n"
"\n"
"\n"
"/* data types */\n"
"#define TYPE_A     float\n"
"#define TYPE_B     float\n"
"#define TYPE_C     float\n"
"#define TYPE_ALPHA float\n"
"#define TYPE_BETA  float\n"
"#define MAD(A,B,DST) mad(A,B,DST)\n"
"\n"
"/* MADs */\n"
"#define TYPE_MAD(MULA,MULB,DST) DST = MAD(MULA,MULB,DST);\n"
"#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);\n"
"\n"
"/* 4x4 micro-tile */\n"
"#define MICRO_TILE \\\n"
"  rA[0] = localA[offA + 0*WG_0I]; \\\n"
"  rA[1] = localA[offA + 1*WG_0I]; \\\n"
"  rA[2] = localA[offA + 2*WG_0I]; \\\n"
"  rA[3] = localA[offA + 3*WG_0I]; \\\n"
"  rB[0] = localB[offB + 0*WG_1J]; \\\n"
"  rB[1] = localB[offB + 1*WG_1J]; \\\n"
"  rB[2] = localB[offB + 2*WG_1J]; \\\n"
"  rB[3] = localB[offB + 3*WG_1J]; \\\n"
"  offA += (MT_0I+PAD); \\\n"
"  offB += (MT_1J+PAD); \\\n"
"  TYPE_MAD(rA[0],rB[0],rC[0][0]); \\\n"
"  TYPE_MAD(rA[0],rB[1],rC[0][1]); \\\n"
"  TYPE_MAD(rA[0],rB[2],rC[0][2]); \\\n"
"  TYPE_MAD(rA[0],rB[3],rC[0][3]); \\\n"
"  TYPE_MAD(rA[1],rB[0],rC[1][0]); \\\n"
"  TYPE_MAD(rA[1],rB[1],rC[1][1]); \\\n"
"  TYPE_MAD(rA[1],rB[2],rC[1][2]); \\\n"
"  TYPE_MAD(rA[1],rB[3],rC[1][3]); \\\n"
"  TYPE_MAD(rA[2],rB[0],rC[2][0]); \\\n"
"  TYPE_MAD(rA[2],rB[1],rC[2][1]); \\\n"
"  TYPE_MAD(rA[2],rB[2],rC[2][2]); \\\n"
"  TYPE_MAD(rA[2],rB[3],rC[2][3]); \\\n"
"  TYPE_MAD(rA[3],rB[0],rC[3][0]); \\\n"
"  TYPE_MAD(rA[3],rB[1],rC[3][1]); \\\n"
"  TYPE_MAD(rA[3],rB[2],rC[3][2]); \\\n"
"  TYPE_MAD(rA[3],rB[3],rC[3][3]); \\\n"
"  mem_fence(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"/* preprocessor definitions of kernel arguments*/\n"
"#define strideC0I 1\n"
"#define strideA0I 1\n"
"#define strideB1J 1\n"
"\n"
"\n"
"/* kernel */\n"
"__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))\n"
"__kernel void CT_SSSSS_Cij_Sk_Aik_Bjk_i16b4f_j16b4f_nl1x1_k16_O2(\n"
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
"  TYPE_C rC[UT_0I][UT_1J] = {{0}};\n"
"  TYPE_A rA[UT_0I];\n"
"  TYPE_B rB[UT_1J];\n"
"\n"
"  /* allocate local memory */\n"
"  __local TYPE_A localA[UNROLL*(MT_0I+PAD)];\n"
"  __local TYPE_B localB[UNROLL*(MT_1J+PAD)];\n"
"\n"
"  /* c indices (group) */\n"
"  unsigned int g0I = get_group_id(0); // d0, tensorA\n"
"  unsigned int g1J = get_group_id(1); // d1, tensorB\n"
"\n"
"  /* c indices (local) */\n"
"  unsigned int l0I = get_local_id(0); // d0\n"
"  unsigned int l1J = get_local_id(1); // d1\n"
"  unsigned int loadSerial = l0I + l1J*WG_0I;\n"
"  unsigned int a0I = loadSerial%LS_PARA_A;\n"
"  unsigned int b1J = loadSerial%LS_PARA_B;\n"
"\n"
"  /* unrolled summation index */\n"
"  unsigned int aK = loadSerial/LS_PARA_A;\n"
"  unsigned int bK = loadSerial/LS_PARA_B;\n"
"\n"
"  /* other non-unrolled summation indices (all start at zero) */\n"
"\n"
"  /* where will this thread read from global memory */\n"
"  A += GLOBAL_A( a0I+g0I*MT_0I, aK );\n"
"  B += GLOBAL_B( b1J+g1J*MT_1J, bK );\n"
"\n"
"  /* where will this thread write to local memory */\n"
"  __local TYPE_A *lA = localA + a0I + aK*(MT_0I+PAD);\n"
"  __local TYPE_B *lB = localB + b1J + bK*(MT_1J+PAD);\n"
"\n"
"  /* iterate over summation indice(s) */\n"
"  unsigned int sumIterK = sizeK / UNROLL;\n"
"  do {\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    /* load A global -> local */\n"
"    float a0, a1, a2, a3, b0, b1, b2, b3;\n"
"    if ( a0I+g0I*MT_0I+0*LS_PARA_A < size0I) {\n"
"     a0 = A[ 0*LS_PARA_A + 0*LS_PERP_A*strideAK];\n"
"     a1 = A[ 0*LS_PARA_A + 1*LS_PERP_A*strideAK];\n"
"     a2 = A[ 0*LS_PARA_A + 2*LS_PERP_A*strideAK];\n"
"     a3 = A[ 0*LS_PARA_A + 3*LS_PERP_A*strideAK];\n"
"    } else {"
"      a0 = 0;\n"
"      a1 = 0;\n"
"      a2 = 0;\n"
"      a3 = 0;\n"
"    }"
"   lA[ 0*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = a0;\n"
"   lA[ 0*LS_PARA_A + 1*LS_PERP_A*(MT_0I+PAD) ] = a1;\n"
"   lA[ 0*LS_PARA_A + 2*LS_PERP_A*(MT_0I+PAD) ] = a2;\n"
"   lA[ 0*LS_PARA_A + 3*LS_PERP_A*(MT_0I+PAD) ] = a3;\n"
"\n"
"    /* load B global -> local */\n"
"    if ( b1J+g1J*MT_1J+0*LS_PARA_B < size1J) {\n"
"      b0 = B[ 0*LS_PARA_B + 0*LS_PERP_B*strideBK];\n"
"      b1 = B[ 0*LS_PARA_B + 1*LS_PERP_B*strideBK];\n"
"      b2 = B[ 0*LS_PARA_B + 2*LS_PERP_B*strideBK];\n"
"      b3 = B[ 0*LS_PARA_B + 3*LS_PERP_B*strideBK];\n"
"    } else {"
"      b0 = 0;\n"
"      b1 = 0;\n"
"      b2 = 0;\n"
"      b3 = 0;\n"
"    }"
"    lB[ 0*LS_PARA_B + 0*LS_PERP_B*(MT_1J+PAD) ] = b0;\n"
"    lB[ 0*LS_PARA_B + 1*LS_PERP_B*(MT_1J+PAD) ] = b1;\n"
"    lB[ 0*LS_PARA_B + 2*LS_PERP_B*(MT_1J+PAD) ] = b2;\n"
"    lB[ 0*LS_PARA_B + 3*LS_PERP_B*(MT_1J+PAD) ] = b3;\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    unsigned int offA = l0I; // d0\n"
"    unsigned int offB = l1J; // d1\n"
"\n"
"    /* do fmas */\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"    MICRO_TILE\n"
"\n"
"    A += (long) strideAK*UNROLL;\n"
"    B += (long) strideBK*UNROLL;\n"
"  } while (--sumIterK > 0);\n"
"\n"
"  /* which global Cij index */\n"
"  unsigned int globalC1J = g1J*MT_1J + l1J;\n"
"  unsigned int globalC0I = g0I*MT_0I + l0I;\n"
"\n"
"  /* write global C */\n"
"  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[0][0], beta) } }\n"
"  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[0][1], beta) } }\n"
"  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[0][2], beta) } }\n"
"  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 0*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[0][3], beta) } }\n"
"  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[1][0], beta) } }\n"
"  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[1][1], beta) } }\n"
"  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[1][2], beta) } }\n"
"  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 1*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[1][3], beta) } }\n"
"  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[2][0], beta) } }\n"
"  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[2][1], beta) } }\n"
"  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[2][2], beta) } }\n"
"  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 2*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[2][3], beta) } }\n"
"  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 0*WG_1J) ], alpha, rC[3][0], beta) } }\n"
"  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 1*WG_1J) ], alpha, rC[3][1], beta) } }\n"
"  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 2*WG_1J) ], alpha, rC[3][2], beta) } }\n"
"  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( globalC0I + 3*WG_0I, globalC1J + 3*WG_1J) ], alpha, rC[3][3], beta) } }\n"
"\n"
"}\n";
#endif

/******************************************************************************
  Configurable Load Kernels
******************************************************************************/

/*
TN - different load patterns

*/
#if 1
const char * kernelSource_TN = R"(

/* tile parameters */
#define PAD               1
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   32

// num load instructions
#define NUM_THREADS       (WG_0I*WG_1J)
#define NUM_LOADS_A       ((MT0I*NUM_UNROLL_ITER)/NUM_THREADS)
#define NUM_LOADS_B       ((MT1J*NUM_UNROLL_ITER)/NUM_THREADS)

/* load pattern
  restrictions:
 - perp*para = num loads
 - num_threads%(MACRO_TILE/NUM_LOADS_PERT) == 0
 for 6x6 micro-tile: perp=12, 6, 3
*/

#define NUM_LOADS_PARA_COAL_A 4
#define NUM_LOADS_PARA_COAL_B 4
#define NUM_LOADS_PERP_COAL_A (NUM_LOADS_A/NUM_LOADS_PARA_COAL_A)
#define NUM_LOADS_PERP_COAL_B (NUM_LOADS_B/NUM_LOADS_PARA_COAL_B)

#define LOAD_SIZE_PARA_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_A)
#define LOAD_SIZE_PARA_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_B)
#define LOAD_SIZE_PERP_COAL_A (MT0I/NUM_LOADS_PERP_COAL_A)
#define LOAD_SIZE_PERP_COAL_B (MT1J/NUM_LOADS_PERP_COAL_B)



/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A( IDXK, IDX0I) ( (IDXK) *strideAK + (IDX0I)*strideA0I  )
#define GET_GLOBAL_INDEX_B( IDXK, IDX1J) ( (IDXK) *strideBK + (IDX1J)*strideB1J )


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

/* data types */
#define DATA_TYPE_STR_A     float
#define DATA_TYPE_STR_B     float
#define DATA_TYPE_STR_C     float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA  float
#define FMA(A,B,DST)        mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 6x6 micro-tile */
#define MICRO_TILE \
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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
#define strideAK 1
#define strideBK 1

)" R"(
__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
__kernel void gemm_kernel(
  __global float       *          C,
  __global float const * restrict A,
  __global float const * restrict B,
  float const alpha,
  float const beta,
  unsigned int const strideC1J,
  unsigned int const strideA0I,
  unsigned int const strideB1J,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK ) {

  /* allocate registers */
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0);
  unsigned int groupIdx1J = get_group_id(1);

  unsigned int localIdx0I = get_local_id(0);
  unsigned int localIdx1J = get_local_id(1);
  unsigned int loadSerial = localIdx0I + localIdx1J*WG_0I; // orig

/*
#define NUM_LOADS_PERP_COAL_A 3
#define NUM_LOADS_PERP_COAL_B 3
#define NUM_LOADS_PARA_COAL_A (NUM_LOADS_A/NUM_LOADS_PERP_COAL_A)
#define NUM_LOADS_PARA_COAL_B (NUM_LOADS_B/NUM_LOADS_PERP_COAL_B)
*/

  unsigned int aI = loadSerial/LOAD_SIZE_PARA_COAL_A;
  unsigned int aK = loadSerial%LOAD_SIZE_PARA_COAL_A;
  unsigned int bJ = loadSerial/LOAD_SIZE_PARA_COAL_B;
  unsigned int bK = loadSerial%LOAD_SIZE_PARA_COAL_B;

  A +=  GET_GLOBAL_INDEX_A(aK, aI+groupIdx0I*MT0I);
  B +=  GET_GLOBAL_INDEX_B(bK, bJ+groupIdx1J*MT1J);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + GET_LOCAL_INDEX_B(bK, bJ);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);
#if 1
    // 4x3=12 for u32
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideA0I];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideA0I];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideA0I];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideA0I];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideB1J];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideB1J];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideB1J];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideB1J];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideA0I];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideA0I];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideA0I];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideA0I];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideB1J];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideB1J];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideB1J];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideB1J];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideA0I];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideA0I];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideA0I];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideA0I];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideB1J];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideB1J];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideB1J];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideB1J];
#endif

)" R"(
#if 0
    // 2x6
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 3
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 4
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 5
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];
#endif

#if 0
    // 1x12
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 3
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 4
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 5
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 6
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 6*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 6*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 6*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 6*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 7
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 7*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 7*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 7*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 7*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 8
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 8*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 8*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 8*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 8*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 9
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 9*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 9*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 9*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 9*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 10
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 10*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 10*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 10*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 10*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 11
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 11*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 11*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 11*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 11*LOAD_SIZE_PERP_COAL_B*strideBK];
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
#if NUM_UNROLL_ITER>16
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
#endif

    A += NUM_UNROLL_ITER;
    B += NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

)" R"(
  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif


/*
NT - different load patterns

*/
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define PAD               1
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   32

// num load instructions
#define NUM_THREADS       (WG_0I*WG_1J)
#define NUM_LOADS_A       ((MT0I*NUM_UNROLL_ITER)/NUM_THREADS)
#define NUM_LOADS_B       ((MT1J*NUM_UNROLL_ITER)/NUM_THREADS)
// restriction: NUM_LOADS must be whole number

/* load pattern
  restrictions:
 - perp*para = num loads
 - num_threads%(MACRO_TILE/NUM_LOADS_PERT) == 0
 for 6x6 micro-tile: para=12, 6, 3
*/
#define NUM_LOADS_PARA_COAL_A 3
#define NUM_LOADS_PARA_COAL_B 3
#define NUM_LOADS_PERP_COAL_A (NUM_LOADS_A/NUM_LOADS_PARA_COAL_A)
#define NUM_LOADS_PERP_COAL_B (NUM_LOADS_B/NUM_LOADS_PARA_COAL_B)

//#define LOAD_SIZE_PERP_COAL_A (MT0I/NUM_LOADS_PERP_COAL_A)
//#define LOAD_SIZE_PERP_COAL_B (MT1J/NUM_LOADS_PERP_COAL_B)
//#define LOAD_SIZE_PARA_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_A)
//#define LOAD_SIZE_PARA_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_B)

#define LOAD_SIZE_PARA_COAL_A (MT0I/NUM_LOADS_PARA_COAL_A)
#define LOAD_SIZE_PARA_COAL_B (MT1J/NUM_LOADS_PARA_COAL_B)
#define LOAD_SIZE_PERP_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PERP_COAL_A)
#define LOAD_SIZE_PERP_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PERP_COAL_B)
#define PERP_A
#define PARA_B

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK)  ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK)  ( (IDX1J)*strideB1J + (IDXK)*strideBK )


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

#define L_IDX_A
#define G_IDX_A

/* data types */
#define DATA_TYPE_STR_A     float
#define DATA_TYPE_STR_B     float
#define DATA_TYPE_STR_C     float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA  float
#define FMA(A,B,DST)        mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 6x6 micro-tile */
#define MICRO_TILE \
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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

)" R"(
__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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

#define NUM_LOADS_PARA_COAL_A 3
#define NUM_LOADS_PARA_COAL_B 3
#define NUM_LOADS_PERP_COAL_A (NUM_LOADS_A/NUM_LOADS_PARA_COAL_A)
#define NUM_LOADS_PERP_COAL_B (NUM_LOADS_B/NUM_LOADS_PARA_COAL_B)

//#define LOAD_SIZE_PERP_COAL_A (MT0I/NUM_LOADS_PERP_COAL_A)
//#define LOAD_SIZE_PERP_COAL_B (MT1J/NUM_LOADS_PERP_COAL_B)
//#define LOAD_SIZE_PARA_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_A)
//#define LOAD_SIZE_PARA_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_B)

#define LOAD_SIZE_PARA_COAL_A (MT0I/NUM_LOADS_PARA_COAL_A)
#define LOAD_SIZE_PARA_COAL_B (MT1J/NUM_LOADS_PARA_COAL_B)
#define LOAD_SIZE_PERP_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PERP_COAL_A)
#define LOAD_SIZE_PERP_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PERP_COAL_B)
  //if (get_global_id(0)==0 && get_global_id(1)==0) {
  //  printf("N_PARA=%u, N_PERP=%u, S_PARA=%u, S_PERP=%u\n",NUM_LOADS_PARA_COAL_A, NUM_LOADS_PERP_COAL_A, LOAD_SIZE_PARA_COAL_A, LOAD_SIZE_PERP_COAL_A);
  //}

  /* allocate registers */
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0);
  unsigned int groupIdx1J = get_group_id(1);

  unsigned int localIdx0I = get_local_id(0);
  unsigned int localIdx1J = get_local_id(1);
  unsigned int loadSerial = localIdx0I + localIdx1J*WG_0I; // orig

/*
#define NUM_LOADS_PERP_COAL_A 3
#define NUM_LOADS_PERP_COAL_B 3
#define NUM_LOADS_PARA_COAL_A (NUM_LOADS_A/NUM_LOADS_PERP_COAL_A)
#define NUM_LOADS_PARA_COAL_B (NUM_LOADS_B/NUM_LOADS_PERP_COAL_B)
*/

  //unsigned int aI = loadSerial/LOAD_SIZE_PARA_COAL_A;
  //unsigned int aK = loadSerial%LOAD_SIZE_PARA_COAL_A;
  //unsigned int bJ = loadSerial/LOAD_SIZE_PARA_COAL_B;
  //unsigned int bK = loadSerial%LOAD_SIZE_PARA_COAL_B;

  unsigned int aI = loadSerial%LOAD_SIZE_PARA_COAL_A;
  unsigned int aK = loadSerial/LOAD_SIZE_PARA_COAL_A;
  unsigned int bJ = loadSerial%LOAD_SIZE_PARA_COAL_B;
  unsigned int bK = loadSerial/LOAD_SIZE_PARA_COAL_B;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + GET_LOCAL_INDEX_B(bK, bJ);
/*

#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )
  __local float *lA = localA + ((aI) + (aK)*(MT0I+PAD) )
  __local float *lB = localB + ((bJ) + (bK)*(MT1J+PAD) )     GET_LOCAL_INDEX_B(bK, bJ);

*/

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);
#if 0
    // 4x3=12 for TN-u32
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B+0*strideBK];

    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B+0*strideBK];

    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B+0*strideBK];
#endif
#if 1
    // 3x4=12 for NT-u32
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    lA[0*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    lA[0*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B*strideBK];

    lA[0*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A+0*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B+0*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    //for (unsigned int i = loadSerial; i<NUM_UNROLL_ITER*(MT0I); i += 256) {
    //printf("G[%02u][%02u] K[%05u] LID[%04u] lA=%3.0f; lB=%3.0f\n", groupIdx0I, groupIdx1J, sumIterK, i, lA[i], lB[i]);
    //}
#endif

)" R"(
#if 0
    // 2x6
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 3
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 4
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 5
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];
#endif

#if 0
    // 1x12
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 3
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 4
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 5
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 6
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 6*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 6*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 6*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 6*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 7
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 7*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 7*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 7*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 7*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 8
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 8*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 8*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 8*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 8*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 9
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 9*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 9*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 9*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 9*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 10
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 10*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 10*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 10*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 10*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 11
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 11*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 11*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 11*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 11*LOAD_SIZE_PERP_COAL_B*strideBK];
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
#if NUM_UNROLL_ITER>16
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
#endif

    A += NUM_UNROLL_ITER*strideAK;
    B += NUM_UNROLL_ITER*strideBK;
  } while (--sumIterK > 0);

)" R"(
  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif

/*
  NT - branched; different load patterns
*/
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define PAD               1
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   32

// num load instructions
#define NUM_THREADS       (WG_0I*WG_1J)
#define NUM_LOADS_A       ((MT0I*NUM_UNROLL_ITER)/NUM_THREADS)
#define NUM_LOADS_B       ((MT1J*NUM_UNROLL_ITER)/NUM_THREADS)
// restriction: NUM_LOADS must be whole number

/* load pattern
  restrictions:
 - perp*para = num loads
 - num_threads%(MACRO_TILE/NUM_LOADS_PERT) == 0
 for 6x6 micro-tile: para=12, 6, 3
*/
#define NUM_LOADS_PARA_COAL_A 3
#define NUM_LOADS_PARA_COAL_B 3
#define NUM_LOADS_PERP_COAL_A (NUM_LOADS_A/NUM_LOADS_PARA_COAL_A)
#define NUM_LOADS_PERP_COAL_B (NUM_LOADS_B/NUM_LOADS_PARA_COAL_B)

//#define LOAD_SIZE_PERP_COAL_A (MT0I/NUM_LOADS_PERP_COAL_A)
//#define LOAD_SIZE_PERP_COAL_B (MT1J/NUM_LOADS_PERP_COAL_B)
//#define LOAD_SIZE_PARA_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_A)
//#define LOAD_SIZE_PARA_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_B)

#define LOAD_SIZE_PARA_COAL_A (MT0I/NUM_LOADS_PARA_COAL_A)
#define LOAD_SIZE_PARA_COAL_B (MT1J/NUM_LOADS_PARA_COAL_B)
#define LOAD_SIZE_PERP_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PERP_COAL_A)
#define LOAD_SIZE_PERP_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PERP_COAL_B)
#define PERP_A
#define PARA_B

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK)  ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK)  ( (IDX1J)*strideB1J + (IDXK)*strideBK )


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

#define L_IDX_A
#define G_IDX_A

/* data types */
#define DATA_TYPE_STR_A     float
#define DATA_TYPE_STR_B     float
#define DATA_TYPE_STR_C     float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA  float
#define FMA(A,B,DST)        mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 6x6 micro-tile */
#define MICRO_TILE \
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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

)" R"(
__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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

#define NUM_LOADS_PARA_COAL_A 3
#define NUM_LOADS_PARA_COAL_B 3
#define NUM_LOADS_PERP_COAL_A (NUM_LOADS_A/NUM_LOADS_PARA_COAL_A)
#define NUM_LOADS_PERP_COAL_B (NUM_LOADS_B/NUM_LOADS_PARA_COAL_B)

//#define LOAD_SIZE_PERP_COAL_A (MT0I/NUM_LOADS_PERP_COAL_A)
//#define LOAD_SIZE_PERP_COAL_B (MT1J/NUM_LOADS_PERP_COAL_B)
//#define LOAD_SIZE_PARA_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_A)
//#define LOAD_SIZE_PARA_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_B)

#define LOAD_SIZE_PARA_COAL_A (MT0I/NUM_LOADS_PARA_COAL_A)
#define LOAD_SIZE_PARA_COAL_B (MT1J/NUM_LOADS_PARA_COAL_B)
#define LOAD_SIZE_PERP_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PERP_COAL_A)
#define LOAD_SIZE_PERP_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PERP_COAL_B)
  //if (get_global_id(0)==0 && get_global_id(1)==0) {
  //  printf("N_PARA=%u, N_PERP=%u, S_PARA=%u, S_PERP=%u\n",NUM_LOADS_PARA_COAL_A, NUM_LOADS_PERP_COAL_A, LOAD_SIZE_PARA_COAL_A, LOAD_SIZE_PERP_COAL_A);
  //}

  /* allocate registers */
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0);
  unsigned int groupIdx1J = get_group_id(1);

  unsigned int localIdx0I = get_local_id(0);
  unsigned int localIdx1J = get_local_id(1);
  unsigned int loadSerial = localIdx0I + localIdx1J*WG_0I; // orig

/*
#define NUM_LOADS_PERP_COAL_A 3
#define NUM_LOADS_PERP_COAL_B 3
#define NUM_LOADS_PARA_COAL_A (NUM_LOADS_A/NUM_LOADS_PERP_COAL_A)
#define NUM_LOADS_PARA_COAL_B (NUM_LOADS_B/NUM_LOADS_PERP_COAL_B)
*/

  //unsigned int aI = loadSerial/LOAD_SIZE_PARA_COAL_A;
  //unsigned int aK = loadSerial%LOAD_SIZE_PARA_COAL_A;
  //unsigned int bJ = loadSerial/LOAD_SIZE_PARA_COAL_B;
  //unsigned int bK = loadSerial%LOAD_SIZE_PARA_COAL_B;

  unsigned int aI = loadSerial%LOAD_SIZE_PARA_COAL_A;
  unsigned int aK = loadSerial/LOAD_SIZE_PARA_COAL_A;
  unsigned int bJ = loadSerial%LOAD_SIZE_PARA_COAL_B;
  unsigned int bK = loadSerial/LOAD_SIZE_PARA_COAL_B;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + GET_LOCAL_INDEX_B(bK, bJ);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);
#if 0
    // 4x3=12 for u32
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A+0*strideAK + 0*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B+0*strideBK + 0*LOAD_SIZE_PERP_COAL_B+0*strideBK];

    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A+0*strideAK + 1*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B+0*strideBK + 1*LOAD_SIZE_PERP_COAL_B+0*strideBK];

    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A+0*strideAK + 2*LOAD_SIZE_PERP_COAL_A+0*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B+0*strideBK];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B+0*strideBK + 2*LOAD_SIZE_PERP_COAL_B+0*strideBK];
#endif
#if 1
    // 3x4=12 for u32
    lA[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+0*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+1*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[1*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+2*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[2*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+0*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+1*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[1*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+2*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[2*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+0*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+1*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[1*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+2*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[2*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+0*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+1*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[1*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*(MT0I+PAD)] = (aI+groupIdx0I*MT0I+2*LOAD_SIZE_PARA_COAL_A >= size0I) ? 0.f : A[2*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+0*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+1*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[1*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+2*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[2*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+0*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+1*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[1*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+2*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[2*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+0*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+1*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[1*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+2*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[2*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+0*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+1*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[1*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*(MT1J+PAD)] = (bJ+groupIdx1J*MT1J+2*LOAD_SIZE_PARA_COAL_B >= size1J) ? 0.f : B[2*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    //for (unsigned int i = loadSerial; i<NUM_UNROLL_ITER*(MT0I); i += 256) {
    //printf("G[%02u][%02u] K[%05u] LID[%04u] lA=%3.0f; lB=%3.0f\n", groupIdx0I, groupIdx1J, sumIterK, i, lA[i], lB[i]);
    //}
#endif

)" R"(
#if 0
    // 2x6
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 3
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 4
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 5
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];
#endif

#if 0
    // 1x12
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 3
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 4
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 5
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 6
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 6*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 6*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 6*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 6*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 7
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 7*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 7*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 7*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 7*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 8
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 8*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 8*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 8*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 8*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 9
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 9*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 9*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 9*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 9*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 10
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 10*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 10*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 10*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 10*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 11
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 11*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 11*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 11*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 11*LOAD_SIZE_PERP_COAL_B*strideBK];
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
#if NUM_UNROLL_ITER>16
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
#endif

    A += NUM_UNROLL_ITER*strideAK;
    B += NUM_UNROLL_ITER*strideBK;
  } while (--sumIterK > 0);

)" R"(
  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  if (globalIdxC0I+0*WG_0I < size0I && globalIdxC1J+0*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)}
  if (globalIdxC0I+0*WG_0I < size0I && globalIdxC1J+1*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)}
  if (globalIdxC0I+0*WG_0I < size0I && globalIdxC1J+2*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)}
  if (globalIdxC0I+0*WG_0I < size0I && globalIdxC1J+3*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)}
  if (globalIdxC0I+0*WG_0I < size0I && globalIdxC1J+4*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)}
  if (globalIdxC0I+0*WG_0I < size0I && globalIdxC1J+5*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)}
  if (globalIdxC0I+1*WG_0I < size0I && globalIdxC1J+0*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)}
  if (globalIdxC0I+1*WG_0I < size0I && globalIdxC1J+1*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)}
  if (globalIdxC0I+1*WG_0I < size0I && globalIdxC1J+2*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)}
  if (globalIdxC0I+1*WG_0I < size0I && globalIdxC1J+3*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)}
  if (globalIdxC0I+1*WG_0I < size0I && globalIdxC1J+4*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)}
  if (globalIdxC0I+1*WG_0I < size0I && globalIdxC1J+5*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)}
  if (globalIdxC0I+2*WG_0I < size0I && globalIdxC1J+0*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)}
  if (globalIdxC0I+2*WG_0I < size0I && globalIdxC1J+1*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)}
  if (globalIdxC0I+2*WG_0I < size0I && globalIdxC1J+2*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)}
  if (globalIdxC0I+2*WG_0I < size0I && globalIdxC1J+3*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)}
  if (globalIdxC0I+2*WG_0I < size0I && globalIdxC1J+4*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)}
  if (globalIdxC0I+2*WG_0I < size0I && globalIdxC1J+5*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)}
  if (globalIdxC0I+3*WG_0I < size0I && globalIdxC1J+0*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)}
  if (globalIdxC0I+3*WG_0I < size0I && globalIdxC1J+1*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)}
  if (globalIdxC0I+3*WG_0I < size0I && globalIdxC1J+2*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)}
  if (globalIdxC0I+3*WG_0I < size0I && globalIdxC1J+3*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)}
  if (globalIdxC0I+3*WG_0I < size0I && globalIdxC1J+4*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)}
  if (globalIdxC0I+3*WG_0I < size0I && globalIdxC1J+5*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)}
  if (globalIdxC0I+4*WG_0I < size0I && globalIdxC1J+0*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)}
  if (globalIdxC0I+4*WG_0I < size0I && globalIdxC1J+1*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)}
  if (globalIdxC0I+4*WG_0I < size0I && globalIdxC1J+2*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)}
  if (globalIdxC0I+4*WG_0I < size0I && globalIdxC1J+3*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)}
  if (globalIdxC0I+4*WG_0I < size0I && globalIdxC1J+4*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)}
  if (globalIdxC0I+4*WG_0I < size0I && globalIdxC1J+5*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)}
  if (globalIdxC0I+5*WG_0I < size0I && globalIdxC1J+0*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)}
  if (globalIdxC0I+5*WG_0I < size0I && globalIdxC1J+1*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)}
  if (globalIdxC0I+5*WG_0I < size0I && globalIdxC1J+2*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)}
  if (globalIdxC0I+5*WG_0I < size0I && globalIdxC1J+3*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)}
  if (globalIdxC0I+5*WG_0I < size0I && globalIdxC1J+4*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)}
  if (globalIdxC0I+5*WG_0I < size0I && globalIdxC1J+5*WG_1J < size1J ) {TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)}

};
)";
#endif



/******************************************************************************
  Final single-read-type Kernels
******************************************************************************/

/*
  NN - u8
*/
#if 1
const char * kernelSource_NN = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*1 + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK)  ( (IDX0I)*1 + (IDXK) *strideAK  )
#define GET_GLOBAL_INDEX_B(IDXK, IDX1J)  ( (IDXK) *1 + (IDX1J)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;
  //unsigned int localSerial = localIdx0I*WG_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%TPI;
  unsigned int aK = localSerial/TPI;
  unsigned int bJ = (localSerial)/NUM_UNROLL_ITER;
  unsigned int bK = (localSerial)%NUM_UNROLL_ITER;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bK, bJ+groupIdx1J*MT1J);

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

    lB[0*TPI] = B[0*TPI*strideBK];
    lB[1*TPI] = B[1*TPI*strideBK];
    lB[2*TPI] = B[2*TPI*strideBK];

#if NUM_UNROLL_ITER>8
    lA[3*TPI] = A[3*TPI+0*strideAK];
    lA[4*TPI] = A[4*TPI+0*strideAK];
    lA[5*TPI] = A[5*TPI+0*strideAK];

    lB[3*TPI] = B[3*TPI*strideBK];
    lB[4*TPI] = B[4*TPI*strideBK];
    lB[5*TPI] = B[5*TPI*strideBK];
#endif
#if NUM_UNROLL_ITER>16
    lA[6*TPI] = A[6*TPI+0*strideAK];
    lA[7*TPI] = A[7*TPI+0*strideAK];
    lA[8*TPI] = A[8*TPI+0*strideAK];
    lA[9*TPI] = A[9*TPI+0*strideAK];
    lA[10*TPI] = A[10*TPI+0*strideAK];
    lA[11*TPI] = A[11*TPI+0*strideAK];

    lB[6*TPI] = B[6*TPI*strideBK];
    lB[7*TPI] = B[7*TPI*strideBK];
    lB[8*TPI] = B[8*TPI*strideBK];
    lB[9*TPI] = B[9*TPI*strideBK];
    lB[10*TPI] = B[10*TPI*strideBK];
    lB[11*TPI] = B[11*TPI*strideBK];
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
#if NUM_UNROLL_ITER>16
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
#endif

    A += strideAK*NUM_UNROLL_ITER;
    B += NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif


/*
  NT - u8
*/
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
#if 1
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
#else
  unsigned int groupSerial = get_group_id(0)*get_num_groups(1) + get_group_id(1);
  unsigned int groupIdx0I = groupSerial % get_num_groups(0);
  unsigned int groupIdx1J = groupSerial / get_num_groups(0);
#endif



  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

  unsigned int aI = localSerial%TPI;
  unsigned int aK = localSerial/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

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

    lB[0*TPI] = B[0*TPI+0*strideBK];
    lB[1*TPI] = B[1*TPI+0*strideBK];
    lB[2*TPI] = B[2*TPI+0*strideBK];

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

    A += strideAK*NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif


/*
  TN - u32
  12% slower than NT
*/
#if 0
const char * kernelSource_TN = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   32
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A( IDXK, IDX0I) ( (IDXK) *strideA0I + (IDX0I)*strideAK  )
#define GET_GLOBAL_INDEX_B( IDXK, IDX1J) ( (IDXK) *strideB1J + (IDX1J)*strideBK )


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

/* data types */
#define DATA_TYPE_STR_A     float
#define DATA_TYPE_STR_B     float
#define DATA_TYPE_STR_C     float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA  float
#define FMA(A,B,DST)        mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 6x6 micro-tile */
#define MICRO_TILE \
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
#if 1
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
#else
  // convert work-group order to z-order
  unsigned int groupRow;
  unsigned int groupCol;
  unsigned int morton = get_group_id(1) * get_num_groups(0) + get_group_id(0);
  groupRow = morton;
  groupCol = ( groupRow >> 1 );
  groupRow &= 0x55555555;
  groupCol &= 0x55555555;
  groupRow |= ( groupRow >> 1 );
  groupCol |= ( groupCol >> 1 );
  groupRow &= 0x33333333;
  groupCol &= 0x33333333;
  groupRow |= ( groupRow >> 2 );
  groupCol |= ( groupCol >> 2 );
  groupRow &= 0x0f0f0f0f;
  groupCol &= 0x0f0f0f0f;
  groupRow |= ( groupRow >> 4 );
  groupCol |= ( groupCol >> 4 );
  groupRow &= 0x00ff00ff;
  groupCol &= 0x00ff00ff;
  groupRow |= ( groupRow >> 8 );
  groupCol |= ( groupCol >> 8 );
  groupRow &= 0x0000ffff;
  groupCol &= 0x0000ffff;
  unsigned int groupIdx0I = groupRow;
  unsigned int groupIdx1J = groupCol;
  //if (get_local_id(0)==0 && get_local_id(1)==0 ) {
  //  printf("S[%03u] G[%03u][%03u] M[%03u][%03u]\n", morton, get_group_id(0), get_group_id(1), groupIdx0I, groupIdx1J );
  //}
#endif

  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I; // orig
  //unsigned int localSerial = localIdx0I*WG_1J + localIdx1J; // new

  unsigned int aI = localSerial/NUM_UNROLL_ITER;
  unsigned int aK = localSerial%NUM_UNROLL_ITER;
  unsigned int bJ = localSerial/NUM_UNROLL_ITER;
  unsigned int bK = localSerial%NUM_UNROLL_ITER;

  A +=  GET_GLOBAL_INDEX_A(aK, aI+groupIdx0I*MT0I);
  B +=  GET_GLOBAL_INDEX_B(bK, bJ+groupIdx1J*MT1J);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + GET_LOCAL_INDEX_B(bK, bJ);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */
    lA[0*TPI] = A[0*TPI*strideAK];
    lA[1*TPI] = A[1*TPI*strideAK];
    lA[2*TPI] = A[2*TPI*strideAK];

    lB[0*TPI] = B[0*TPI*strideBK];
    lB[1*TPI] = B[1*TPI*strideBK];
    lB[2*TPI] = B[2*TPI*strideBK];

#if NUM_UNROLL_ITER>8
    lA[3*TPI] = A[3*TPI*strideAK];
    lA[4*TPI] = A[4*TPI*strideAK];
    lA[5*TPI] = A[5*TPI*strideAK];

    lB[3*TPI] = B[3*TPI*strideBK];
    lB[4*TPI] = B[4*TPI*strideBK];
    lB[5*TPI] = B[5*TPI*strideBK];
#endif

#if NUM_UNROLL_ITER>16
    lA[ 6*TPI] = A[ 6*TPI*strideAK];
    lA[ 7*TPI] = A[ 7*TPI*strideAK];
    lA[ 8*TPI] = A[ 8*TPI*strideAK];
    lA[ 9*TPI] = A[ 9*TPI*strideAK];
    lA[10*TPI] = A[10*TPI*strideAK];
    lA[11*TPI] = A[11*TPI*strideAK];

    lB[ 6*TPI] = B[ 6*TPI*strideBK];
    lB[ 7*TPI] = B[ 7*TPI*strideBK];
    lB[ 8*TPI] = B[ 8*TPI*strideBK];
    lB[ 9*TPI] = B[ 9*TPI*strideBK];
    lB[10*TPI] = B[10*TPI*strideBK];
    lB[11*TPI] = B[11*TPI*strideBK];
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
#if NUM_UNROLL_ITER>16
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
#endif

    A += NUM_UNROLL_ITER;
    B += NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif


/*
  TT - u32
  this one is slow
*/
#if 0
const char * kernelSource_TT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   32
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A( IDXK, IDX0I) ( (IDXK) *strideA0I + (IDX0I)*strideAK  )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK)  ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

/* data types */
#define DATA_TYPE_STR_A     float
#define DATA_TYPE_STR_B     float
#define DATA_TYPE_STR_C     float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA  float
#define FMA(A,B,DST)        mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 6x6 micro-tile */
#define MICRO_TILE \
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

  unsigned int aI = localSerial/NUM_UNROLL_ITER;
  unsigned int aK = localSerial%NUM_UNROLL_ITER;
  unsigned int bJ = localSerial%TPI;
  unsigned int bK = localSerial/TPI;


  A +=  GET_GLOBAL_INDEX_A(aK, aI+groupIdx0I*MT0I);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + GET_LOCAL_INDEX_B(bK, bJ);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */
    lA[0*TPI] = A[0*TPI*strideAK];
    lA[1*TPI] = A[1*TPI*strideAK];
    lA[2*TPI] = A[2*TPI*strideAK];

    lB[0*TPI] = B[0*TPI];
    lB[1*TPI] = B[1*TPI];
    lB[2*TPI] = B[2*TPI];

#if NUM_UNROLL_ITER>8
    lA[3*TPI] = A[3*TPI*strideAK];
    lA[4*TPI] = A[4*TPI*strideAK];
    lA[5*TPI] = A[5*TPI*strideAK];

    lB[3*TPI] = B[3*TPI];
    lB[4*TPI] = B[4*TPI];
    lB[5*TPI] = B[5*TPI];
#endif

#if NUM_UNROLL_ITER>16
    lA[ 6*TPI] = A[ 6*TPI*strideAK];
    lA[ 7*TPI] = A[ 7*TPI*strideAK];
    lA[ 8*TPI] = A[ 8*TPI*strideAK];
    lA[ 9*TPI] = A[ 9*TPI*strideAK];
    lA[10*TPI] = A[10*TPI*strideAK];
    lA[11*TPI] = A[11*TPI*strideAK];

    lB[ 6*TPI] = B[ 6*TPI];
    lB[ 7*TPI] = B[ 7*TPI];
    lB[ 8*TPI] = B[ 8*TPI];
    lB[ 9*TPI] = B[ 9*TPI];
    lB[10*TPI] = B[10*TPI];
    lB[11*TPI] = B[11*TPI];
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
#if NUM_UNROLL_ITER>16
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
#endif

    A += NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif


/*
  TT_switched - u8
  switch the order in which wg are executed on device
  this one is fast
*/
#if 1
const char * kernelSource_TT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A( IDXK, IDX0I) ( (IDXK) *strideA0I + (IDX0I)*strideAK  )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK)  ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

/* data types */
#define DATA_TYPE_STR_A     float
#define DATA_TYPE_STR_B     float
#define DATA_TYPE_STR_C     float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA  float
#define FMA(A,B,DST)        mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 6x6 micro-tile */
#define MICRO_TILE \
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  //unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  //unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  //unsigned int groupSerial = get_group_id(0) + get_group_id(1)*get_num_groups(0);
  unsigned int groupSerial = get_group_id(0)*get_num_groups(1) + get_group_id(1);

  // re-order work-groups
  unsigned int groupIdx0I = groupSerial % get_num_groups(0); // get_group_id(1);
  unsigned int groupIdx1J = groupSerial / get_num_groups(0); // get_group_id(0);


  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

  unsigned int aI = localSerial/NUM_UNROLL_ITER;
  unsigned int aK = localSerial%NUM_UNROLL_ITER;
  unsigned int bJ = localSerial%TPI;
  unsigned int bK = localSerial/TPI;


  A +=  GET_GLOBAL_INDEX_A(aK, aI+groupIdx0I*MT0I);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + GET_LOCAL_INDEX_B(bK, bJ);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* load global -> local */
    lA[0*TPI] = A[0*TPI*strideAK];
    lA[1*TPI] = A[1*TPI*strideAK];
    lA[2*TPI] = A[2*TPI*strideAK];

    lB[0*TPI] = B[0*TPI];
    lB[1*TPI] = B[1*TPI];
    lB[2*TPI] = B[2*TPI];

#if NUM_UNROLL_ITER>8
    lA[3*TPI] = A[3*TPI*strideAK];
    lA[4*TPI] = A[4*TPI*strideAK];
    lA[5*TPI] = A[5*TPI*strideAK];

    lB[3*TPI] = B[3*TPI];
    lB[4*TPI] = B[4*TPI];
    lB[5*TPI] = B[5*TPI];
#endif

#if NUM_UNROLL_ITER>16
    lA[ 6*TPI] = A[ 6*TPI*strideAK];
    lA[ 7*TPI] = A[ 7*TPI*strideAK];
    lA[ 8*TPI] = A[ 8*TPI*strideAK];
    lA[ 9*TPI] = A[ 9*TPI*strideAK];
    lA[10*TPI] = A[10*TPI*strideAK];
    lA[11*TPI] = A[11*TPI*strideAK];

    lB[ 6*TPI] = B[ 6*TPI];
    lB[ 7*TPI] = B[ 7*TPI];
    lB[ 8*TPI] = B[ 8*TPI];
    lB[ 9*TPI] = B[ 9*TPI];
    lB[10*TPI] = B[10*TPI];
    lB[11*TPI] = B[11*TPI];
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
#if NUM_UNROLL_ITER>16
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
#endif

    A += NUM_UNROLL_ITER;
    B += strideBK*NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif


/******************************************************************************
  Graveyard of failed experiments
******************************************************************************/

/*
  TN - different load patterns

*/
#if 0
const char * kernelSource_TN = R"(

/* tile parameters */
#define PAD               1
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   32

// num load instructions
#define NUM_THREADS       (WG_0I*WG_1J)
#define NUM_LOADS_A       ((MT0I*NUM_UNROLL_ITER)/NUM_THREADS)
#define NUM_LOADS_B       ((MT1J*NUM_UNROLL_ITER)/NUM_THREADS)

/* load pattern
  restrictions:
 - perp*para = num loads
 - num_threads%(MACRO_TILE/NUM_LOADS_PERT) == 0
 for 6x6 micro-tile: perp=12, 6, 3
*/
#define NUM_LOADS_PERP_COAL_A 12
#define NUM_LOADS_PERP_COAL_B 12
#define NUM_LOADS_PARA_COAL_A (NUM_LOADS_A/NUM_LOADS_PERP_COAL_A)
#define NUM_LOADS_PARA_COAL_B (NUM_LOADS_B/NUM_LOADS_PERP_COAL_B)

#define LOAD_SIZE_PERP_COAL_A (MT0I/NUM_LOADS_PERP_COAL_A)
#define LOAD_SIZE_PERP_COAL_B (MT1J/NUM_LOADS_PERP_COAL_B)
#define LOAD_SIZE_PARA_COAL_A (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_A)
#define LOAD_SIZE_PARA_COAL_B (NUM_UNROLL_ITER/NUM_LOADS_PARA_COAL_B)


#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A( IDXK, IDX0I) ( (IDXK) *strideA0I + (IDX0I)*strideAK  )
#define GET_GLOBAL_INDEX_B( IDXK, IDX1J) ( (IDXK) *strideB1J + (IDX1J)*strideBK )


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

/* data types */
#define DATA_TYPE_STR_A     float
#define DATA_TYPE_STR_B     float
#define DATA_TYPE_STR_C     float
#define DATA_TYPE_STR_ALPHA float
#define DATA_TYPE_STR_BETA  float
#define FMA(A,B,DST)        mad(A,B,DST)
#define TYPE_MAD(MULA,MULB,DST) DST = FMA(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 6x6 micro-tile */
#define MICRO_TILE \
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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

)" R"(
__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0);
  unsigned int groupIdx1J = get_group_id(1);

  unsigned int localIdx0I = get_local_id(0);
  unsigned int localIdx1J = get_local_id(1);
  unsigned int loadSerial = localIdx0I + localIdx1J*WG_0I; // orig

/*
#define NUM_LOADS_PERP_COAL_A 3
#define NUM_LOADS_PERP_COAL_B 3
#define NUM_LOADS_PARA_COAL_A (NUM_LOADS_A/NUM_LOADS_PERP_COAL_A)
#define NUM_LOADS_PARA_COAL_B (NUM_LOADS_B/NUM_LOADS_PERP_COAL_B)
*/

  unsigned int aI = loadSerial/LOAD_SIZE_PARA_COAL_A;
  unsigned int aK = loadSerial%LOAD_SIZE_PARA_COAL_A;
  unsigned int bJ = loadSerial/LOAD_SIZE_PARA_COAL_B;
  unsigned int bK = loadSerial%LOAD_SIZE_PARA_COAL_B;

  A +=  GET_GLOBAL_INDEX_A(aK, aI+groupIdx0I*MT0I);
  B +=  GET_GLOBAL_INDEX_B(bK, bJ+groupIdx1J*MT1J);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + GET_LOCAL_INDEX_B(bK, bJ);

  /* iterate over all summation indices */
  unsigned int sumIterK = sizeK / NUM_UNROLL_ITER;
  do {
    barrier(CLK_LOCAL_MEM_FENCE);
#if 0
    // 4x3=12 for u32
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[2*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[2*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[3*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[3*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[2*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[2*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[3*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[3*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
#endif

)" R"(
#if 0
    // 2x6
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 3
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 4
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 5
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];
    lA[1*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[1*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];

    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];
    lB[1*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[1*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];
#endif

#if 1
    // 1x12
    // col 0
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 0*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 0*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 0*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 0*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 1
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 1*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 1*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 1*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 1*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 2
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 2*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 2*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 2*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 2*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 3
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 3*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 3*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 3*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 3*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 4
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 4*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 4*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 4*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 4*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 5
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 5*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 5*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 5*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 5*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 6
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 6*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 6*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 6*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 6*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 7
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 7*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 7*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 7*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 7*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 8
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 8*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 8*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 8*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 8*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 9
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 9*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 9*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 9*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 9*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 10
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 10*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 10*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 10*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 10*LOAD_SIZE_PERP_COAL_B*strideBK];

    // col 11
    lA[0*LOAD_SIZE_PARA_COAL_A*(MT0I+PAD) + 11*LOAD_SIZE_PERP_COAL_A] = A[0*LOAD_SIZE_PARA_COAL_A + 11*LOAD_SIZE_PERP_COAL_A*strideAK];
    lB[0*LOAD_SIZE_PARA_COAL_B*(MT1J+PAD) + 11*LOAD_SIZE_PERP_COAL_B] = B[0*LOAD_SIZE_PARA_COAL_B + 11*LOAD_SIZE_PERP_COAL_B*strideBK];
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
#if NUM_UNROLL_ITER>16
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
#endif

    A += NUM_UNROLL_ITER;
    B += NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

)" R"(
  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif



// TN - original
#if 0
const char * kernelSource_TN = R"(

#define  M6x6 \
            rA[0][0] = lA[offA + 0];  \
            rA[0][1] = lA[offA + 16]; \
            rA[0][2] = lA[offA + 32]; \
            rA[0][3] = lA[offA + 48]; \
            rA[0][4] = lA[offA + 64]; \
            rA[0][5] = lA[offA + 80]; \
            rB[0][0] = lB[offB + 0];  \
            rB[0][1] = lB[offB + 16]; \
            rB[0][2] = lB[offB + 32]; \
            rB[0][3] = lB[offB + 48]; \
            rB[0][4] = lB[offB + 64]; \
            rB[0][5] = lB[offB + 80]; \
            offA += 97; \
            offB += 97; \
            rC[0][0]=mad(rA[0][0],rB[0][0],rC[0][0]); \
            rC[1][0]=mad(rA[0][1],rB[0][0],rC[1][0]); \
            rC[2][0]=mad(rA[0][2],rB[0][0],rC[2][0]); \
            rC[3][0]=mad(rA[0][3],rB[0][0],rC[3][0]); \
            rC[4][0]=mad(rA[0][4],rB[0][0],rC[4][0]); \
            rC[5][0]=mad(rA[0][5],rB[0][0],rC[5][0]); \
            rC[0][1]=mad(rA[0][0],rB[0][1],rC[0][1]); \
            rC[1][1]=mad(rA[0][1],rB[0][1],rC[1][1]); \
            rC[2][1]=mad(rA[0][2],rB[0][1],rC[2][1]); \
            rC[3][1]=mad(rA[0][3],rB[0][1],rC[3][1]); \
            rC[4][1]=mad(rA[0][4],rB[0][1],rC[4][1]); \
            rC[5][1]=mad(rA[0][5],rB[0][1],rC[5][1]); \
            rC[0][2]=mad(rA[0][0],rB[0][2],rC[0][2]); \
            rC[1][2]=mad(rA[0][1],rB[0][2],rC[1][2]); \
            rC[2][2]=mad(rA[0][2],rB[0][2],rC[2][2]); \
            rC[3][2]=mad(rA[0][3],rB[0][2],rC[3][2]); \
            rC[4][2]=mad(rA[0][4],rB[0][2],rC[4][2]); \
            rC[5][2]=mad(rA[0][5],rB[0][2],rC[5][2]); \
            rC[0][3]=mad(rA[0][0],rB[0][3],rC[0][3]); \
            rC[1][3]=mad(rA[0][1],rB[0][3],rC[1][3]); \
            rC[2][3]=mad(rA[0][2],rB[0][3],rC[2][3]); \
            rC[3][3]=mad(rA[0][3],rB[0][3],rC[3][3]); \
            rC[4][3]=mad(rA[0][4],rB[0][3],rC[4][3]); \
            rC[5][3]=mad(rA[0][5],rB[0][3],rC[5][3]); \
            rC[0][4]=mad(rA[0][0],rB[0][4],rC[0][4]); \
            rC[1][4]=mad(rA[0][1],rB[0][4],rC[1][4]); \
            rC[2][4]=mad(rA[0][2],rB[0][4],rC[2][4]); \
            rC[3][4]=mad(rA[0][3],rB[0][4],rC[3][4]); \
            rC[4][4]=mad(rA[0][4],rB[0][4],rC[4][4]); \
            rC[5][4]=mad(rA[0][5],rB[0][4],rC[5][4]); \
            rC[0][5]=mad(rA[0][0],rB[0][5],rC[0][5]); \
            rC[1][5]=mad(rA[0][1],rB[0][5],rC[1][5]); \
            rC[2][5]=mad(rA[0][2],rB[0][5],rC[2][5]); \
            rC[3][5]=mad(rA[0][3],rB[0][5],rC[3][5]); \
            rC[4][5]=mad(rA[0][4],rB[0][5],rC[4][5]); \
            rC[5][5]=mad(rA[0][5],rB[0][5],rC[5][5]); \
            mem_fence(CLK_LOCAL_MEM_FENCE);

  __attribute__((reqd_work_group_size(16,16,1)))
  __kernel void sgemm_Col_TN_B1_MX096_NX096_KX16 (
    __global float * C,
    __global float const * restrict A,
    __global float const * restrict B,
    float const alpha,
    float const beta,
    uint ldc,
    uint lda,
    uint ldb,
    uint const M,
    uint const N,
    uint const K )
{
  float rC[6][6]  = { {(float)0} };
  float rA[1][6];
  float rB[1][6];

  __local float lA[1552];
  __local float lB[1552];

  uint gidx = get_group_id(0);
  uint gidy = get_group_id(1);
  uint idx = get_local_id(0);
  uint idy = get_local_id(1);

  A +=  (gidx*96+idy)*lda + idx;
  B +=  (gidy*96+idy)*ldb + idx;


  uint block_k = K >> 4;
  do
  {
    __local float* plA = lA + idx*97+idy;
    __local float* plB = lB + idx*97+idy;
    barrier(CLK_LOCAL_MEM_FENCE);

    plB[0] = B[0];
    plB[16] = B[16*ldb];
    plB[32] = B[32*ldb];
    plB[48] = B[48*ldb];
    plB[64] = B[64*ldb];
    plB[80] = B[80*ldb];

    plA[0] = A[0];
    plA[16] = A[16*lda];
    plA[32] = A[32*lda];
    plA[48] = A[48*lda];
    plA[64] = A[64*lda];
    plA[80] = A[80*lda];


    barrier(CLK_LOCAL_MEM_FENCE);
    uint offA = idx;
    uint offB = idy;

    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6
    M6x6

    A += 16;
    B += 16;
  } while (--block_k > 0);

  C+= gidx*96+idx;
  C+= gidy*96*ldc;
  C+= idy*ldc;

  C[ 0*ldc] = alpha*rC[0][0] + beta*C[ 0*ldc];
  C[16*ldc] = alpha*rC[0][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[0][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[0][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[0][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[0][5] + beta*C[80*ldc];
  C+=16;
  C[ 0*ldc] = alpha*rC[1][0] + beta*C[ 0*ldc];
  C[16*ldc] = alpha*rC[1][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[1][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[1][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[1][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[1][5] + beta*C[80*ldc];
  C+=16;
  C[ 0*ldc] = alpha*rC[2][0] + beta*C[ 0*ldc];
  C[16*ldc] = alpha*rC[2][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[2][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[2][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[2][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[2][5] + beta*C[80*ldc];
  C+=16;
  C[ 0*ldc] = alpha*rC[3][0] + beta*C[ 0*ldc];
  C[16*ldc] = alpha*rC[3][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[3][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[3][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[3][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[3][5] + beta*C[80*ldc];
  C+=16;
  C[ 0*ldc] = alpha*rC[4][0] + beta*C[ 0*ldc];
  C[16*ldc] = alpha*rC[4][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[4][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[4][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[4][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[4][5] + beta*C[80*ldc];
  C+=16;
  C[ 0*ldc] = alpha*rC[5][0] + beta*C [0*ldc];
  C[16*ldc] = alpha*rC[5][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[5][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[5][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[5][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[5][5] + beta*C[80*ldc];

}
)";
#endif

/* 
NT with branches
6x6 micro tile
unroll 8
single source load (w/ PAD to eliminate bank conflict added from ssl)
this is fastest so far: 61 vgpr, 88% valusage, 80%peak
*/
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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

__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

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
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
    //doLoad = aI+groupIdx0I*MT0I < size0I;
    maxLoads = (size0I - groupIdx0I*MT0I)/TPI;
    if (aI < (size0I - groupIdx0I*MT0I)%TPI ) { maxLoads++; }
  } else { // B
    localPtr = localB + GET_LOCAL_INDEX_A(bJ, bK);
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);
    globalInc = strideBK*NUM_UNROLL_ITER;
    //doLoad = bJ+groupIdx1J*MT1J < size1J;
    maxLoads = (size1J - groupIdx1J*MT1J)/TPI;
    if (bJ < (size1J - groupIdx1J*MT1J)%TPI ) { maxLoads++; }
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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%04u, %04u, %f\n", globalIdxC0I, globalIdxC1J, rC[0][0] );

  /* write global C */
  if (globalIdxC0I + 0*WG_0I < size0I && globalIdxC1J + 0*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta) }
  if (globalIdxC0I + 0*WG_0I < size0I && globalIdxC1J + 1*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta) }
  if (globalIdxC0I + 0*WG_0I < size0I && globalIdxC1J + 2*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta) }
  if (globalIdxC0I + 0*WG_0I < size0I && globalIdxC1J + 3*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta) }
  if (globalIdxC0I + 0*WG_0I < size0I && globalIdxC1J + 4*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta) }
  if (globalIdxC0I + 0*WG_0I < size0I && globalIdxC1J + 5*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta) }

  if (globalIdxC0I + 1*WG_0I < size0I && globalIdxC1J + 0*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta) }
  if (globalIdxC0I + 1*WG_0I < size0I && globalIdxC1J + 1*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta) }
  if (globalIdxC0I + 1*WG_0I < size0I && globalIdxC1J + 2*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta) }
  if (globalIdxC0I + 1*WG_0I < size0I && globalIdxC1J + 3*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta) }
  if (globalIdxC0I + 1*WG_0I < size0I && globalIdxC1J + 4*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta) }
  if (globalIdxC0I + 1*WG_0I < size0I && globalIdxC1J + 5*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta) }

  if (globalIdxC0I + 2*WG_0I < size0I && globalIdxC1J + 0*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta) }
  if (globalIdxC0I + 2*WG_0I < size0I && globalIdxC1J + 1*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta) }
  if (globalIdxC0I + 2*WG_0I < size0I && globalIdxC1J + 2*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta) }
  if (globalIdxC0I + 2*WG_0I < size0I && globalIdxC1J + 3*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta) }
  if (globalIdxC0I + 2*WG_0I < size0I && globalIdxC1J + 4*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta) }
  if (globalIdxC0I + 2*WG_0I < size0I && globalIdxC1J + 5*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta) }

  if (globalIdxC0I + 3*WG_0I < size0I && globalIdxC1J + 0*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta) }
  if (globalIdxC0I + 3*WG_0I < size0I && globalIdxC1J + 1*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta) }
  if (globalIdxC0I + 3*WG_0I < size0I && globalIdxC1J + 2*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta) }
  if (globalIdxC0I + 3*WG_0I < size0I && globalIdxC1J + 3*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta) }
  if (globalIdxC0I + 3*WG_0I < size0I && globalIdxC1J + 4*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta) }
  if (globalIdxC0I + 3*WG_0I < size0I && globalIdxC1J + 5*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta) }

  if (globalIdxC0I + 4*WG_0I < size0I && globalIdxC1J + 0*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta) }
  if (globalIdxC0I + 4*WG_0I < size0I && globalIdxC1J + 1*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta) }
  if (globalIdxC0I + 4*WG_0I < size0I && globalIdxC1J + 2*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta) }
  if (globalIdxC0I + 4*WG_0I < size0I && globalIdxC1J + 3*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta) }
  if (globalIdxC0I + 4*WG_0I < size0I && globalIdxC1J + 4*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta) }
  if (globalIdxC0I + 4*WG_0I < size0I && globalIdxC1J + 5*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta) }

  if (globalIdxC0I + 5*WG_0I < size0I && globalIdxC1J + 0*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta) }
  if (globalIdxC0I + 5*WG_0I < size0I && globalIdxC1J + 1*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta) }
  if (globalIdxC0I + 5*WG_0I < size0I && globalIdxC1J + 2*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta) }
  if (globalIdxC0I + 5*WG_0I < size0I && globalIdxC1J + 3*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta) }
  if (globalIdxC0I + 5*WG_0I < size0I && globalIdxC1J + 4*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta) }
  if (globalIdxC0I + 5*WG_0I < size0I && globalIdxC1J + 5*WG_1J < size1J ) { TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta) }

};
)";
#endif



/*
NT 6x6 micro tile
unroll 8
single source load (w/ PAD to eliminate bank conflict added from ssl)
this is fastest so far: 60 vgpr, 90% valusage, 84%peak
*/
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

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
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    localPtr = localB + GET_LOCAL_INDEX_A(bJ, bK);
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);
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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif


// NN 6x6 micro tile
// non-ssl
// non-coalesced very slow
// 
#if 0
const char * kernelSource_NN = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*1 + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK)  ( (IDX0I)*1 + (IDXK) *strideAK  )
#define GET_GLOBAL_INDEX_B(IDXK, IDX1J)  ( (IDXK) *1 + (IDX1J)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;
  //unsigned int localSerial = localIdx0I*WG_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%TPI;
  unsigned int aK = localSerial/TPI;
  unsigned int bJ = (localSerial)/NUM_UNROLL_ITER;
  unsigned int bK = (localSerial)%NUM_UNROLL_ITER;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  (localSerial+groupIdx1J*MT1J)*strideBK; // GET_GLOBAL_INDEX_B(bK, bJ+groupIdx1J*MT1J);

  __local float *lA = localA + GET_LOCAL_INDEX_A(aI, aK);
  __local float *lB = localB + localSerial; // GET_LOCAL_INDEX_B(bK, bJ);


//   unsigned int aI = (localSerial%128)%TPI;
//   unsigned int aK = (localSerial%128)/TPI;
//   unsigned int bJ = (localSerial%128)/TPI;
//   unsigned int bK = (localSerial%128)%TPI;
// 
//   __local  float *localPtr;
//   __global float *globalPtr;
//   unsigned int globalInc;
// 
//   // localSerial [0,127] load A, [128,256] load B
//   if (localSerial < 128 ) { // A
//     localPtr = localA + GET_LOCAL_INDEX_A(aI, aK);
//     globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
//     globalInc = strideAK*NUM_UNROLL_ITER;
//   } else { // B
//     localPtr = localB + localSerial%128; // GET_LOCAL_INDEX_A(bJ, bK);
//     globalPtr = B + (localSerial%128+groupIdx1J*MT1J)*strideBK;
//     globalInc = NUM_UNROLL_ITER;
//   }



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

    //if (localSerial < 96) {
      lB[ 0*2*(MT1J+PAD) ] = B[ 0*2 ];
      lB[ 1*2*(MT1J+PAD) ] = B[ 1*2 ];
      lB[ 2*2*(MT1J+PAD) ] = B[ 2*2 ];
      lB[ 3*2*(MT1J+PAD) ] = B[ 3*2 ];
    //}
    // lB[0*TPI] = B[0*TPI*strideBK];
    // lB[1*TPI] = B[1*TPI*strideBK];
    // lB[2*TPI] = B[2*TPI*strideBK];

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
    B += NUM_UNROLL_ITER;
  } while (--sumIterK > 0);

  //printf("%f, %f, %f, %f, %f, %f\n", rC[0][0], rC[1][1], rC[2][2], rC[3][3], rC[4][4], rC[5][5] );

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif



// original NN
#if 0
const char * kernelSource_NN = R"(

#define  M6x6 \
            rA[0][0] = lA[offA + 0];          \
            rA[0][1] = lA[offA + 16];          \
            rA[0][2] = lA[offA + 32];          \
            rA[0][3] = lA[offA + 48];          \
            rA[0][4] = lA[offA + 64];          \
            rA[0][5] = lA[offA + 80];          \
            rB[0][0] = lB[offB + 0];          \
            rB[0][1] = lB[offB + 16];          \
            rB[0][2] = lB[offB + 32];          \
            rB[0][3] = lB[offB + 48];          \
            rB[0][4] = lB[offB + 64];          \
            rB[0][5] = lB[offB + 80];          \
            offA += 97;                  \
            offB += 97;                  \
            rC[0][0]=mad(rA[0][0],rB[0][0],rC[0][0]); \
            rC[1][0]=mad(rA[0][1],rB[0][0],rC[1][0]); \
            rC[2][0]=mad(rA[0][2],rB[0][0],rC[2][0]); \
            rC[3][0]=mad(rA[0][3],rB[0][0],rC[3][0]); \
            rC[4][0]=mad(rA[0][4],rB[0][0],rC[4][0]); \
            rC[5][0]=mad(rA[0][5],rB[0][0],rC[5][0]); \
            rC[0][1]=mad(rA[0][0],rB[0][1],rC[0][1]); \
            rC[1][1]=mad(rA[0][1],rB[0][1],rC[1][1]); \
            rC[2][1]=mad(rA[0][2],rB[0][1],rC[2][1]); \
            rC[3][1]=mad(rA[0][3],rB[0][1],rC[3][1]); \
            rC[4][1]=mad(rA[0][4],rB[0][1],rC[4][1]); \
            rC[5][1]=mad(rA[0][5],rB[0][1],rC[5][1]); \
            rC[0][2]=mad(rA[0][0],rB[0][2],rC[0][2]); \
            rC[1][2]=mad(rA[0][1],rB[0][2],rC[1][2]); \
            rC[2][2]=mad(rA[0][2],rB[0][2],rC[2][2]); \
            rC[3][2]=mad(rA[0][3],rB[0][2],rC[3][2]); \
            rC[4][2]=mad(rA[0][4],rB[0][2],rC[4][2]); \
            rC[5][2]=mad(rA[0][5],rB[0][2],rC[5][2]); \
            rC[0][3]=mad(rA[0][0],rB[0][3],rC[0][3]); \
            rC[1][3]=mad(rA[0][1],rB[0][3],rC[1][3]); \
            rC[2][3]=mad(rA[0][2],rB[0][3],rC[2][3]); \
            rC[3][3]=mad(rA[0][3],rB[0][3],rC[3][3]); \
            rC[4][3]=mad(rA[0][4],rB[0][3],rC[4][3]); \
            rC[5][3]=mad(rA[0][5],rB[0][3],rC[5][3]); \
            rC[0][4]=mad(rA[0][0],rB[0][4],rC[0][4]); \
            rC[1][4]=mad(rA[0][1],rB[0][4],rC[1][4]); \
            rC[2][4]=mad(rA[0][2],rB[0][4],rC[2][4]); \
            rC[3][4]=mad(rA[0][3],rB[0][4],rC[3][4]); \
            rC[4][4]=mad(rA[0][4],rB[0][4],rC[4][4]); \
            rC[5][4]=mad(rA[0][5],rB[0][4],rC[5][4]); \
            rC[0][5]=mad(rA[0][0],rB[0][5],rC[0][5]); \
            rC[1][5]=mad(rA[0][1],rB[0][5],rC[1][5]); \
            rC[2][5]=mad(rA[0][2],rB[0][5],rC[2][5]); \
            rC[3][5]=mad(rA[0][3],rB[0][5],rC[3][5]); \
            rC[4][5]=mad(rA[0][4],rB[0][5],rC[4][5]); \
            rC[5][5]=mad(rA[0][5],rB[0][5],rC[5][5]); \
            mem_fence(CLK_LOCAL_MEM_FENCE);

  __attribute__((reqd_work_group_size(16,16,1)))
  __kernel void sgemm_Col_NN_B1_MX096_NX096_KX16 (
    __global float * C,
    __global float const * restrict A,
    __global float const * restrict B,
    float const alpha,
    float const beta,
    uint ldc,
    uint lda,
    uint ldb,
    uint const M,
    uint const N,
    uint const K)
{
  float rC[6][6]  = { {(float)0} };
  float rA[1][6];
  float rB[1][6];

  __local float lA[1552];
  __local float lB[1552];

  uint gidx = get_group_id(0);
  uint gidy = get_group_id(1);
  uint idx = get_local_id(0);
  uint idy = get_local_id(1);

  A +=  gidx*96+ idx + idy*lda;
  B +=  gidy*96*ldb+ idx + idy*ldb;


  uint block_k = K >> 4;
  do {
    __local float* plA = lA + idy*97+idx;
    __local float* plB = lB + idx*97+idy;
    barrier(CLK_LOCAL_MEM_FENCE);
    plB[0] = B[0];
    plB[16] = B[16*ldb];
    plB[32] = B[32*ldb];
    plB[48] = B[48*ldb];
    plB[64] = B[64*ldb];
    plB[80] = B[80*ldb];

    plA[0] = A[0+0*lda];
    plA[16] = A[16+0*lda];
    plA[32] = A[32+0*lda];
    plA[48] = A[48+0*lda];
    plA[64] = A[64+0*lda];
    plA[80] = A[80+0*lda];


    barrier(CLK_LOCAL_MEM_FENCE);
    uint offA = idx;
    uint offB = idy;

    M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6
      M6x6

      A += lda<<4;
    B += 16;
  } while (--block_k > 0);

  C+= gidx*96+idx;
  C+= gidy*96*ldc;
  C+= idy*ldc;

  C[0*ldc] = alpha*rC[0][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[0][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[0][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[0][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[0][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[0][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[1][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[1][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[1][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[1][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[1][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[1][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[2][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[2][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[2][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[2][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[2][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[2][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[3][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[3][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[3][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[3][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[3][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[3][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[4][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[4][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[4][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[4][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[4][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[4][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[5][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[5][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[5][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[5][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[5][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[5][5] + beta*C[80*ldc];

}
)";
#endif


// NN - 6x6 micro tile
// unroll 8
// single source load (w/ PAD to eliminate bank conflict added from ssl)
// 81% valubusy 75%peak
// read B more registers but coalesced
#if 0
const char * kernelSource_NN = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
#define PAD               0
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
// for NN
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK)  ( (IDX0I)*strideA0I + (IDXK) *strideAK  )
#define GET_GLOBAL_INDEX_B(IDXK, IDX1J)  ( (IDXK) *strideBK  + (IDX1J)*1 )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

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
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    localPtr = localB + bJ+bK*(MT1J+PAD); // GET_LOCAL_INDEX_B(bK, bJ);
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);
    //printf("t=%03u g=%04u l=%03u\n", localSerial, GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK), bJ*NUM_UNROLL_ITER+bK*(MT1J+PAD) );
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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

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
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
#define PAD               1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER/2)

/* global memory indices */

// for NN
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK)  ( (IDX0I)*strideA0I + (IDXK) *strideAK  )
#define GET_GLOBAL_INDEX_B(IDXK, IDX1J)  ( (IDXK) *strideBK  + (IDX1J)*strideB1J )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*(MT0I+PAD)];
  __local float localB[NUM_UNROLL_ITER*(MT1J+PAD)];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

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
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    localPtr = localB + localSerial%128; // GET_LOCAL_INDEX_A(bJ, bK);
    globalPtr = B + (localSerial%128+groupIdx1J*MT1J)*strideBK;
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
      localPtr[ 0*(MT1J+PAD) ] = globalPtr[ 0 ];
      localPtr[ 1*(MT1J+PAD) ] = globalPtr[ 1 ];
      localPtr[ 2*(MT1J+PAD) ] = globalPtr[ 2 ];
      localPtr[ 3*(MT1J+PAD) ] = globalPtr[ 3 ];
      localPtr[ 4*(MT1J+PAD) ] = globalPtr[ 4 ];
      localPtr[ 5*(MT1J+PAD) ] = globalPtr[ 5 ];
      localPtr[ 6*(MT1J+PAD) ] = globalPtr[ 6 ];
      localPtr[ 7*(MT1J+PAD) ] = globalPtr[ 7 ];
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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif

























#if 0
// Cobalt sgemm_NT_128x128x8_prefetch


const char * kernelSource_NT = R"(

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
#define MAD(A,B,DST) mad(A,B,DST)

/* MADs */
#define TYPE_MAD(MULA,MULB,DST) DST = MAD(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

/* 8x8 micro-tile */
// load
#define UTPREFETCH \
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
  mem_fence(CLK_LOCAL_MEM_FENCE);
  

#define UT2 \
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
  mem_fence(CLK_LOCAL_MEM_FENCE); \
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
  mem_fence(CLK_LOCAL_MEM_FENCE); \
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
  mem_fence(CLK_LOCAL_MEM_FENCE); \
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
  mem_fence(CLK_LOCAL_MEM_FENCE);
)"
R"(
#define UT2_LAST \
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
  mem_fence(CLK_LOCAL_MEM_FENCE); \
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
  mem_fence(CLK_LOCAL_MEM_FENCE); \
  /* don't prefetch red, compute black */ \
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
  mem_fence(CLK_LOCAL_MEM_FENCE);


/* preprocessor definitions of kernel arguments*/
#define strideC0I 1
#define strideA0I 1
#define strideB1J 1

)"
R"(

/* kernel */
__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
__kernel void CT_SSSSS_Cij_Sk_Aik_Bjk_i16x8f_j16x8f_nl4x4_k8_O2(
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
  TYPE_C rC[UT_0I][UT_1J] = {{0}};
  TYPE_A rA_red[UT_0I];
  TYPE_B rB_red[UT_1J];
  TYPE_A rA_black[UT_0I];
  TYPE_B rB_black[UT_1J];

  /* allocate local memory */
  __local TYPE_A localBasePtrA[2*UNROLL*MT_0I_2];
  __local TYPE_B localBasePtrB[2*UNROLL*MT_1J_2];

  /* c indices (group) */
  unsigned int g0I = get_group_id(0); // d0, tensorA
  unsigned int g1J = get_group_id(1); // d1, tensorB

  /* c indices (local) */
  unsigned int l0I = get_local_id(0); // d0
  unsigned int l1J = get_local_id(1); // d1
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
  __local TYPE_A *localReadPtrA  = localBasePtrA + localReadOffsetA;
  __local TYPE_B *localReadPtrB  = localBasePtrB + localReadOffsetB;

  /* where will this thread write to local memory */
  unsigned int localWriteOffsetA = a0I + aK*(MT_0I+PAD);
  unsigned int localWriteOffsetB = b1J + bK*(MT_1J+PAD);
  __local TYPE_A *localWritePtrA = localBasePtrA + localWriteOffsetA;
  __local TYPE_B *localWritePtrB = localBasePtrB + localWriteOffsetB;

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
  barrier(CLK_LOCAL_MEM_FENCE);

  /* iterate over summation indice(s) */
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
    barrier(CLK_LOCAL_MEM_FENCE);

    /* unroll offsets */
    unsigned int offA = l0I; // d0
    unsigned int offB = l1J; // d1

    /* do fmas */
    UTPREFETCH
    UT2
    UT2
    UT2
    UT2_LAST

    barrier(CLK_LOCAL_MEM_FENCE);

  } while (--sumIterK > 1);

  /* swap local read offset and update pointers */
  localReadOffsetA ^= UNROLL*MT_0I_2;
  localReadOffsetB ^= UNROLL*MT_1J_2;
  localReadPtrA = localBasePtrA + localReadOffsetA;
  localReadPtrB = localBasePtrB + localReadOffsetB;

  UTPREFETCH
  UT2
  UT2
  UT2
  UT2_LAST

  /* which global Cij index */
  unsigned int globalC1J = g1J*MT_1J + l1J;
  unsigned int globalC0I = g0I*MT_0I + l0I;

  /* write global C */
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
#define WG_0I           16
#define WG_1J           16
#define UT0I       8
#define UT1J       8
#define MT0I       128
#define MT1J       128
#define PAD                 1
#define MT0I_POW2  256
#define MT1J_POW2  256
#define NUM_UNROLL_ITER     8
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
#define UTPREFETCH \
  rA_red[0] = compA[ offA + 0*WG_0I]; \
  rA_red[1] = compA[ offA + 1*WG_0I]; \
  rA_red[2] = compA[ offA + 2*WG_0I]; \
  rA_red[3] = compA[ offA + 3*WG_0I]; \
  rA_red[4] = compA[ offA + 4*WG_0I]; \
  rA_red[5] = compA[ offA + 5*WG_0I]; \
  rA_red[6] = compA[ offA + 6*WG_0I]; \
  rA_red[7] = compA[ offA + 7*WG_0I]; \
  rB_red[0] = compB[ offB + 0*WG_1J]; \
  rB_red[1] = compB[ offB + 1*WG_1J]; \
  rB_red[2] = compB[ offB + 2*WG_1J]; \
  rB_red[3] = compB[ offB + 3*WG_1J]; \
  rB_red[4] = compB[ offB + 4*WG_1J]; \
  rB_red[5] = compB[ offB + 5*WG_1J]; \
  rB_red[6] = compB[ offB + 6*WG_1J]; \
  rB_red[7] = compB[ offB + 7*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
  mem_fence(CLK_LOCAL_MEM_FENCE);
  

// load black, compute red
#define UT2 \
  rA_black[0] = compA[offA + 0*WG_0I]; \
  rA_black[1] = compA[offA + 1*WG_0I]; \
  rA_black[2] = compA[offA + 2*WG_0I]; \
  rA_black[3] = compA[offA + 3*WG_0I]; \
  rA_black[4] = compA[offA + 4*WG_0I]; \
  rA_black[5] = compA[offA + 5*WG_0I]; \
  rA_black[6] = compA[offA + 6*WG_0I]; \
  rA_black[7] = compA[offA + 7*WG_0I]; \
  \
  rB_black[0] = compB[offB + 0*WG_1J]; \
  rB_black[1] = compB[offB + 1*WG_1J]; \
  rB_black[2] = compB[offB + 2*WG_1J]; \
  rB_black[3] = compB[offB + 3*WG_1J]; \
  rB_black[4] = compB[offB + 4*WG_1J]; \
  rB_black[5] = compB[offB + 5*WG_1J]; \
  rB_black[6] = compB[offB + 6*WG_1J]; \
  rB_black[7] = compB[offB + 7*WG_1J]; \
  \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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
  rA_red[0] = compA[offA + 0*WG_0I]; \
  rA_red[1] = compA[offA + 1*WG_0I]; \
  rA_red[2] = compA[offA + 2*WG_0I]; \
  rA_red[3] = compA[offA + 3*WG_0I]; \
  rA_red[4] = compA[offA + 4*WG_0I]; \
  rA_red[5] = compA[offA + 5*WG_0I]; \
  rA_red[6] = compA[offA + 6*WG_0I]; \
  rA_red[7] = compA[offA + 7*WG_0I]; \
  \
  rB_red[0] = compB[offB + 0*WG_1J]; \
  rB_red[1] = compB[offB + 1*WG_1J]; \
  rB_red[2] = compB[offB + 2*WG_1J]; \
  rB_red[3] = compB[offB + 3*WG_1J]; \
  rB_red[4] = compB[offB + 4*WG_1J]; \
  rB_red[5] = compB[offB + 5*WG_1J]; \
  rB_red[6] = compB[offB + 6*WG_1J]; \
  rB_red[7] = compB[offB + 7*WG_1J]; \
  \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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

__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA_red[UT0I];
  float rB_red[UT1J];
  float rA_black[UT0I];
  float rB_black[UT1J];

  /* allocate local memory */
  __local float localA_raw[2*NUM_UNROLL_ITER*MT0I_POW2];
  __local float localB_raw[2*NUM_UNROLL_ITER*MT1J_POW2];

  // red   starts at 0*NUM_UNROLL_ITER*MTPOW2
  // black starts at 1*NUM_UNROLL_ITER*MTPOW2

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

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
    loadSwapBit = NUM_UNROLL_ITER*MT0I_POW2;
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    loadPtrOffset = GET_LOCAL_INDEX_A(bJ, bK);
    loadPtrBase = localB_raw;
    loadSwapBit = NUM_UNROLL_ITER*MT1J_POW2;
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);
    globalInc = strideBK*NUM_UNROLL_ITER;
  }
  // address from which to read for computation
  unsigned int compAOffset = NUM_UNROLL_ITER*MT0I_POW2;
  unsigned int compBOffset = NUM_UNROLL_ITER*MT1J_POW2;

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
    compAOffset ^= NUM_UNROLL_ITER*MT0I_POW2;
    compBOffset ^= NUM_UNROLL_ITER*MT1J_POW2;
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
    UTPREFETCH
    UT2
#if NUM_UNROLL_ITER>2
    UT2
#endif
#if NUM_UNROLL_ITER>4
    UT2
    UT2
#endif
#if NUM_UNROLL_ITER>8
    UT2
    UT2
    UT2
    UT2
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

  } while (--sumIterK > 1);

  /* swap compute pointers */
  compAOffset ^= NUM_UNROLL_ITER*MT0I_POW2;
  compBOffset ^= NUM_UNROLL_ITER*MT1J_POW2;
  compA = localA_raw + compAOffset;
  compB = localB_raw + compBOffset;

  /* do mads */
  unsigned int offA = localIdx0I; // d0
  unsigned int offB = localIdx1J; // d1
  UTPREFETCH
  UT2
#if NUM_UNROLL_ITER>4
  UT2
#endif
#if NUM_UNROLL_ITER>4
  UT2
  UT2
#endif
#if NUM_UNROLL_ITER>8
  UT2
  UT2
  UT2
  UT2
#endif
  
  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[0][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[0][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[1][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[1][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[2][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[2][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[3][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[3][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[4][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[4][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[5][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[5][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[6][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[6][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[6][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[6][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[6][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[6][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[6][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[6][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[7][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[7][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[7][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[7][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[7][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[7][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[7][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[7][7], beta)

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
#define WG_0I           16
#define WG_1J           16
#define UT0I       6
#define UT1J       6
#define MT0I       96
#define MT1J       96
#define MT0I_POW2  128
#define MT1J_POW2  128
#define NUM_UNROLL_ITER     8
#define PAD                 1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
#define UTPREFETCH \
  rA_red[0] = compA[ offA + 0*WG_0I]; \
  rA_red[1] = compA[ offA + 1*WG_0I]; \
  rA_red[2] = compA[ offA + 2*WG_0I]; \
  rA_red[3] = compA[ offA + 3*WG_0I]; \
  rA_red[4] = compA[ offA + 4*WG_0I]; \
  rA_red[5] = compA[ offA + 5*WG_0I]; \
  rB_red[0] = compB[ offB + 0*WG_1J]; \
  rB_red[1] = compB[ offB + 1*WG_1J]; \
  rB_red[2] = compB[ offB + 2*WG_1J]; \
  rB_red[3] = compB[ offB + 3*WG_1J]; \
  rB_red[4] = compB[ offB + 4*WG_1J]; \
  rB_red[5] = compB[ offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
  mem_fence(CLK_LOCAL_MEM_FENCE);
  

// load black, compute red
#define UT2 \
  rA_black[0] = compA[offA + 0*WG_0I]; \
  rA_black[1] = compA[offA + 1*WG_0I]; \
  rA_black[2] = compA[offA + 2*WG_0I]; \
  rA_black[3] = compA[offA + 3*WG_0I]; \
  rA_black[4] = compA[offA + 4*WG_0I]; \
  rA_black[5] = compA[offA + 5*WG_0I]; \
  rB_black[0] = compB[offB + 0*WG_1J]; \
  rB_black[1] = compB[offB + 1*WG_1J]; \
  rB_black[2] = compB[offB + 2*WG_1J]; \
  rB_black[3] = compB[offB + 3*WG_1J]; \
  rB_black[4] = compB[offB + 4*WG_1J]; \
  rB_black[5] = compB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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
  rA_red[0] = compA[ offA + 0*WG_0I]; \
  rA_red[1] = compA[ offA + 1*WG_0I]; \
  rA_red[2] = compA[ offA + 2*WG_0I]; \
  rA_red[3] = compA[ offA + 3*WG_0I]; \
  rA_red[4] = compA[ offA + 4*WG_0I]; \
  rA_red[5] = compA[ offA + 5*WG_0I]; \
  rB_red[0] = compB[ offB + 0*WG_1J]; \
  rB_red[1] = compB[ offB + 1*WG_1J]; \
  rB_red[2] = compB[ offB + 2*WG_1J]; \
  rB_red[3] = compB[ offB + 3*WG_1J]; \
  rB_red[4] = compB[ offB + 4*WG_1J]; \
  rB_red[5] = compB[ offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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

__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA_red[UT0I];
  float rB_red[UT1J];
  float rA_black[UT0I];
  float rB_black[UT1J];

  /* allocate local memory */
  __local float localA_raw[2*NUM_UNROLL_ITER*MT0I_POW2];
  __local float localB_raw[2*NUM_UNROLL_ITER*MT1J_POW2];

  // red   starts at 0*NUM_UNROLL_ITER*MTPOW2
  // black starts at 1*NUM_UNROLL_ITER*MTPOW2

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

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
    loadSwapBit = NUM_UNROLL_ITER*MT0I_POW2;
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    loadPtrOffset = GET_LOCAL_INDEX_A(bJ, bK);
    loadPtrBase = localB_raw;
    loadSwapBit = NUM_UNROLL_ITER*MT1J_POW2;
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);
    globalInc = strideBK*NUM_UNROLL_ITER;
  }
  // address from which to read for computation
  unsigned int compAOffset = NUM_UNROLL_ITER*MT0I_POW2;
  unsigned int compBOffset = NUM_UNROLL_ITER*MT1J_POW2;

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
    compAOffset ^= NUM_UNROLL_ITER*MT0I_POW2;
    compBOffset ^= NUM_UNROLL_ITER*MT1J_POW2;
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
    UTPREFETCH
    UT2
    UT2
    UT2
    UT2
#if NUM_UNROLL_ITER>8
    UT2
    UT2
    UT2
    UT2
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

  } while (--sumIterK > 1);

  /* swap compute pointers */
  compAOffset ^= NUM_UNROLL_ITER*MT0I_POW2;
  compBOffset ^= NUM_UNROLL_ITER*MT1J_POW2;
  compA = localA_raw + compAOffset;
  compB = localB_raw + compBOffset;

  /* do mads */
  unsigned int offA = localIdx0I; // d0
  unsigned int offB = localIdx1J; // d1
  UTPREFETCH
  UT2
  UT2
  UT2
  UT2
#if NUM_UNROLL_ITER>8
  UT2
  UT2
  UT2
  UT2
#endif
  
  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

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
#define WG_0I           16
#define WG_1J           16
#define UT0I       6
#define UT1J       6
#define MT0I       96
#define MT1J       96
#define MT0I_POW2  128
#define MT1J_POW2  128
#define NUM_UNROLL_ITER     8
#define PAD                 1
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER/2)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I+PAD) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J+PAD) )

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
  rA[0] = compA[offA + 0*WG_0I]; \
  rA[1] = compA[offA + 1*WG_0I]; \
  rA[2] = compA[offA + 2*WG_0I]; \
  rA[3] = compA[offA + 3*WG_0I]; \
  rA[4] = compA[offA + 4*WG_0I]; \
  rA[5] = compA[offA + 5*WG_0I]; \
  rB[0] = compB[offB + 0*WG_1J]; \
  rB[1] = compB[offB + 1*WG_1J]; \
  rB[2] = compB[offB + 2*WG_1J]; \
  rB[3] = compB[offB + 3*WG_1J]; \
  rB[4] = compB[offB + 4*WG_1J]; \
  rB[5] = compB[offB + 5*WG_1J]; \
  offA += (MT0I+PAD); \
  offB += (MT1J+PAD); \
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

__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA_raw[2*NUM_UNROLL_ITER*MT0I_POW2];
  __local float localB_raw[2*NUM_UNROLL_ITER*MT1J_POW2];

  // red   starts at 0*NUM_UNROLL_ITER*MTPOW2
  // black starts at 1*NUM_UNROLL_ITER*MTPOW2

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

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
    loadSwapBit = NUM_UNROLL_ITER*MT0I_POW2;
    globalPtr = A + GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
    globalInc = strideAK*NUM_UNROLL_ITER;
  } else { // B
    loadPtrOffset = GET_LOCAL_INDEX_A(bJ, bK);
    loadPtrBase = localB_raw;
    loadSwapBit = NUM_UNROLL_ITER*MT1J_POW2;
    globalPtr = B + GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);
    globalInc = strideBK*NUM_UNROLL_ITER;
  }
  // address from which to read for computation
  unsigned int compAOffset = NUM_UNROLL_ITER*MT0I_POW2;
  unsigned int compBOffset = NUM_UNROLL_ITER*MT1J_POW2;

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
    compAOffset ^= NUM_UNROLL_ITER*MT0I_POW2;
    compBOffset ^= NUM_UNROLL_ITER*MT1J_POW2;
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
  compAOffset ^= NUM_UNROLL_ITER*MT0I_POW2;
  compBOffset ^= NUM_UNROLL_ITER*MT1J_POW2;
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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

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
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
// threads-per-iteration (loading unroll)
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J) )

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
  rA[0] = local_A_comp[offA + 0*WG_0I]; \
  rA[1] = local_A_comp[offA + 1*WG_0I]; \
  rA[2] = local_A_comp[offA + 2*WG_0I]; \
  rA[3] = local_A_comp[offA + 3*WG_0I]; \
  rA[4] = local_A_comp[offA + 4*WG_0I]; \
  rA[5] = local_A_comp[offA + 5*WG_0I]; \
  rB[0] = local_B_comp[offB + 0*WG_1J]; \
  rB[1] = local_B_comp[offB + 1*WG_1J]; \
  rB[2] = local_B_comp[offB + 2*WG_1J]; \
  rB[3] = local_B_comp[offB + 3*WG_1J]; \
  rB[4] = local_B_comp[offB + 4*WG_1J]; \
  rB[5] = local_B_comp[offB + 5*WG_1J]; \
  offA += MT0I; \
  offB += MT1J; \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float local_AB[2*NUM_UNROLL_ITER*(MT0I+MT1J)];

  __local float *local_A_raw_red   = local_AB+0*NUM_UNROLL_ITER*MT0I+0*NUM_UNROLL_ITER*MT1J;
  __local float *local_B_raw_red   = local_AB+0*NUM_UNROLL_ITER*MT0I+1*NUM_UNROLL_ITER*MT1J;
  __local float *local_A_raw_black = local_AB+1*NUM_UNROLL_ITER*MT0I+0*NUM_UNROLL_ITER*MT1J;
  __local float *local_B_raw_black = local_AB+1*NUM_UNROLL_ITER*MT0I+1*NUM_UNROLL_ITER*MT1J;

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;
  //unsigned int localSerial = localIdx0I*WG_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%TPI;
  unsigned int aK = localSerial/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif


// NT 6x6 micro tile
// unroll 8
// double-buffer load global->lds
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8
// threads-per-iteration (loading unroll)
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J) )

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
  rA[0] = local_A_comp[offA + 0*WG_0I]; \
  rA[1] = local_A_comp[offA + 1*WG_0I]; \
  rA[2] = local_A_comp[offA + 2*WG_0I]; \
  rA[3] = local_A_comp[offA + 3*WG_0I]; \
  rA[4] = local_A_comp[offA + 4*WG_0I]; \
  rA[5] = local_A_comp[offA + 5*WG_0I]; \
  rB[0] = local_B_comp[offB + 0*WG_1J]; \
  rB[1] = local_B_comp[offB + 1*WG_1J]; \
  rB[2] = local_B_comp[offB + 2*WG_1J]; \
  rB[3] = local_B_comp[offB + 3*WG_1J]; \
  rB[4] = local_B_comp[offB + 4*WG_1J]; \
  rB[5] = local_B_comp[offB + 5*WG_1J]; \
  offA += MT0I; \
  offB += MT1J; \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float local_A_raw_red[NUM_UNROLL_ITER*MT0I];
  __local float local_B_raw_red[NUM_UNROLL_ITER*MT1J];

  __local float local_A_raw_black[NUM_UNROLL_ITER*MT0I];
  __local float local_B_raw_black[NUM_UNROLL_ITER*MT1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;
  //unsigned int localSerial = localIdx0I*WG_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%TPI;
  unsigned int aK = localSerial/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;
  //printf("%02u, %02u, %f\n", localIdx0I, localIdx1J, rC[0][0] );

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif

// NT 6x6 micro tile
// float4 - can't get to work
// unroll 16
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   16
#define LOAD_WIDTH        4

/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* global tile indices being loaded */
/* fast read */
#define globalIdxA0I(LID) (groupIdx0I*MT0I + (localSerial+(LID)*WG_0I*WG_1J)/NUM_UNROLL_ITER)
#define globalIdxAK(LID) (localSerial%NUM_UNROLL_ITER)
/* fast read */
#define globalIdxBK(LID) ((localSerial+(LID)*WG_0I*WG_1J)/MT1J)
#define globalIdxB1J(LID) (groupIdx1J*MT1J + (localSerial+(LID)*WG_0I*WG_1J)%MT1J)

/* global non-tile indices being loaded */


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J) )

/* local indices being written */
#define localA0I (localSerial / NUM_UNROLL_ITER)
#define localAK (localSerial % NUM_UNROLL_ITER)
#define localAStride (WG_0I*WG_1J/NUM_UNROLL_ITER)
#define localB1J ( localSerial / MT1J )
#define localBK ( localSerial % MT1J )
#define localBStride  (WG_0I*WG_1J)

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += MT0I; \
  offB += MT1J; \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*MT0I];
  __local float localB[NUM_UNROLL_ITER*MT1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;
  unsigned int subGroupA = localIdx1J > WG_1J/2;
  unsigned int subGroupSerial = localIdx0I + (localIdx1J%(WG_1J/2))*WG_0I;
  //unsigned int localSerial = localIdx0I*WG_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = subGroupSerial%(WG_0I*WG_1J/NUM_UNROLL_ITER);
  unsigned int aK = subGroupSerial/(WG_0I*WG_1J/NUM_UNROLL_ITER);
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;


  A +=  aI+groupIdx0I*MT0I/LOAD_WIDTH + aK*strideAK/LOAD_WIDTH;
  B +=  bJ+groupIdx1J*MT1J/LOAD_WIDTH + bK*strideBK/LOAD_WIDTH;
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
      for (unsigned int i = 0; i < NUM_UNROLL_ITER*MT0I; i++) {
        printf("localA[%u]=%f\n", i, localA[i]);
      }
      for (unsigned int i = 0; i < NUM_UNROLL_ITER*MT0I; i++) {
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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif




// NT 4x4 micro tile
// float2
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     4
#define UT1J     4
#define MT0I     64
#define MT1J     64
#define NUM_UNROLL_ITER   8


/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* global tile indices being loaded */
/* fast read */
#define globalIdxA0I(LID) (groupIdx0I*MT0I + (localSerial+(LID)*WG_0I*WG_1J)/NUM_UNROLL_ITER)
#define globalIdxAK(LID) (localSerial%NUM_UNROLL_ITER)
/* fast read */
#define globalIdxBK(LID) ((localSerial+(LID)*WG_0I*WG_1J)/MT1J)
#define globalIdxB1J(LID) (groupIdx1J*MT1J + (localSerial+(LID)*WG_0I*WG_1J)%MT1J)

/* global non-tile indices being loaded */


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J) )

/* local indices being written */
#define localA0I (localSerial / NUM_UNROLL_ITER)
#define localAK (localSerial % NUM_UNROLL_ITER)
#define localAStride (WG_0I*WG_1J/NUM_UNROLL_ITER)
#define localB1J ( localSerial / MT1J )
#define localBK ( localSerial % MT1J )
#define localBStride  (WG_0I*WG_1J)

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  offA += MT0I; \
  offB += MT1J; \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*MT0I];
  __local float localB[NUM_UNROLL_ITER*MT1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;
  //unsigned int localSerial = localIdx0I*WG_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%(WG_0I*WG_1J/NUM_UNROLL_ITER);
  unsigned int aK = localSerial/(WG_0I*WG_1J/NUM_UNROLL_ITER);
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

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

};
)";
#endif


// NT 6x6 micro tile
// float2
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   8


/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* global tile indices being loaded */
/* fast read */
#define globalIdxA0I(LID) (groupIdx0I*MT0I + (localSerial+(LID)*WG_0I*WG_1J)/NUM_UNROLL_ITER)
#define globalIdxAK(LID) (localSerial%NUM_UNROLL_ITER)
/* fast read */
#define globalIdxBK(LID) ((localSerial+(LID)*WG_0I*WG_1J)/MT1J)
#define globalIdxB1J(LID) (groupIdx1J*MT1J + (localSerial+(LID)*WG_0I*WG_1J)%MT1J)

/* global non-tile indices being loaded */


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J) )

/* local indices being written */
#define localA0I (localSerial / NUM_UNROLL_ITER)
#define localAK (localSerial % NUM_UNROLL_ITER)
#define localAStride (WG_0I*WG_1J/NUM_UNROLL_ITER)
#define localB1J ( localSerial / MT1J )
#define localBK ( localSerial % MT1J )
#define localBStride  (WG_0I*WG_1J)

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += MT0I; \
  offB += MT1J; \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*MT0I];
  __local float localB[NUM_UNROLL_ITER*MT1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;
  //unsigned int localSerial = localIdx0I*WG_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%(WG_0I*WG_1J/NUM_UNROLL_ITER);
  unsigned int aK = localSerial/(WG_0I*WG_1J/NUM_UNROLL_ITER);
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";
#endif



// NT 8x8 micro tile
#if 0
const char * kernelSource_NT = R"(

/* tile parameters */
#define WG_0I         16
#define WG_1J         16
#define UT0I     8
#define UT1J     8
#define MT0I     128
#define MT1J     128
#define NUM_UNROLL_ITER   8
#define TPI (WG_0I*WG_1J/NUM_UNROLL_ITER)


/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* global tile indices being loaded */
/* fast read */
#define globalIdxA0I(LID) (groupIdx0I*MT0I + (localSerial+(LID)*WG_0I*WG_1J)/NUM_UNROLL_ITER)
#define globalIdxAK(LID) (localSerial%NUM_UNROLL_ITER)
/* fast read */
#define globalIdxBK(LID) ((localSerial+(LID)*WG_0I*WG_1J)/MT1J)
#define globalIdxB1J(LID) (groupIdx1J*MT1J + (localSerial+(LID)*WG_0I*WG_1J)%MT1J)

/* global non-tile indices being loaded */


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J) )

/* local indices being written */
#define localA0I (localSerial / NUM_UNROLL_ITER)
#define localAK (localSerial % NUM_UNROLL_ITER)
#define localAStride (WG_0I*WG_1J/NUM_UNROLL_ITER)
#define localB1J ( localSerial / MT1J )
#define localBK ( localSerial % MT1J )
#define localBStride  (WG_0I*WG_1J)

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rA[6] = localA[offA + 6*WG_0I]; \
  rA[7] = localA[offA + 7*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  rB[6] = localB[offB + 6*WG_1J]; \
  rB[7] = localB[offB + 7*WG_1J]; \
  offA += MT0I; \
  offB += MT1J; \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  float rC[UT0I][UT1J] = {{0}};
  float rA[UT0I];
  float rB[UT1J];

  /* allocate local memory */
  __local float localA[NUM_UNROLL_ITER*MT0I];
  __local float localB[NUM_UNROLL_ITER*MT1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;
  //unsigned int localSerial = localIdx0I*WG_1J + localIdx1J; // 15% global mem busy -> 90% global mem busy

  unsigned int aI = localSerial%TPI;
  unsigned int aK = localSerial/TPI;
  unsigned int bJ = aI; // only for NT
  unsigned int bK = aK;

  A +=  GET_GLOBAL_INDEX_A(aI+groupIdx0I*MT0I, aK);
  B +=  GET_GLOBAL_INDEX_B(bJ+groupIdx1J*MT1J, bK);

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
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

  /* write global C */
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[0][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[0][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[1][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[1][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[2][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[2][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[3][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[3][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[4][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[4][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[5][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[5][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[6][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[6][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[6][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[6][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[6][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[6][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[6][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 6*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[6][7], beta)

  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[7][0], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[7][1], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[7][2], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[7][3], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[7][4], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[7][5], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 6*WG_1J) ], alpha, rC[7][6], beta)
  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 7*WG_0I, globalIdxC1J + 7*WG_1J) ], alpha, rC[7][7], beta)

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
#define WG_0I         16
#define WG_1J         16
#define UT0I     6
#define UT1J     6
#define MT0I     96
#define MT1J     96
#define NUM_UNROLL_ITER   16


/* global memory indices */
#define GET_GLOBAL_INDEX_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GET_GLOBAL_INDEX_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GET_GLOBAL_INDEX_B(IDX1J, IDXK) ( (IDX1J)*strideB1J + (IDXK)*strideBK )

/* global tile indices being loaded */
/* fast read */
#define globalIdxA0I(LID) (groupIdx0I*MT0I + (localSerial+(LID)*WG_0I*WG_1J)/NUM_UNROLL_ITER)
#define globalIdxAK(LID) (localSerial%NUM_UNROLL_ITER)
/* fast read */
#define globalIdxBK(LID) ((localSerial+(LID)*WG_0I*WG_1J)/MT1J)
#define globalIdxB1J(LID) (groupIdx1J*MT1J + (localSerial+(LID)*WG_0I*WG_1J)%MT1J)

/* global non-tile indices being loaded */


/* local memory indices */
#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I) )
#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J) )

/* local indices being written */
#define localA0I (localSerial / NUM_UNROLL_ITER)
#define localAK (localSerial % NUM_UNROLL_ITER)
#define localAStride (WG_0I*WG_1J/NUM_UNROLL_ITER)
#define localB1J ( localSerial / MT1J )
#define localBK ( localSerial % MT1J )
#define localBStride  (WG_0I*WG_1J)

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
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rA[4] = localA[offA + 4*WG_0I]; \
  rA[5] = localA[offA + 5*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  rB[4] = localB[offB + 4*WG_1J]; \
  rB[5] = localB[offB + 5*WG_1J]; \
  offA += MT0I; \
  offB += MT1J; \
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


__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
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
  DATA_TYPE_STR_C rC[UT0I][UT1J] = {{0}};
  DATA_TYPE_STR_A rA[UT0I];
  DATA_TYPE_STR_B rB[UT1J];

  /* allocate local memory */
  __local DATA_TYPE_STR_A localA[NUM_UNROLL_ITER*MT0I];
  __local DATA_TYPE_STR_B localB[NUM_UNROLL_ITER*MT1J];

  /* c indices */
  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA
  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB
  unsigned int localIdx0I = get_local_id(0); // d0
  unsigned int localIdx1J = get_local_id(1); // d1
  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;

#if COBALT_PATH
#else
  A +=  groupIdx0I*96 + localIdx0I + localIdx1J*strideAK;
  B +=  groupIdx1J*96 + localIdx0I + localIdx1J*strideBK;
#endif

  /* which global Cij index */
  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;
  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;

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
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
    TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

};
)";


#endif

// NT Cobalt branched kernel
#if 0
const char * kernelSource_NT = "\n"
"/* CT_SSSSS_Cij_Sk_Aik_Bjk_i16b6f_j16b6f_k8_O2 */\n"
"\n"
"/* tile parameters */\n"
"#define WG_0I          16\n"
"#define WG_1J          16\n"
"#define UT0I  6\n"
"#define UT1J  6\n"
"#define MT0I  96\n"
"#define MT1J  96\n"
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
"#define globalIdxA0I(LID) (groupIdx0I*MT0I + (localSerial+(LID)*WG_0I*WG_1J)/NUM_UNROLL_ITER)\n"
"#define globalIdxAK(LID) (localSerial%NUM_UNROLL_ITER)\n"
"/* fast read */\n"
"#define globalIdxBK(LID) ((localSerial+(LID)*WG_0I*WG_1J)/MT1J)\n"
"#define globalIdxB1J(LID) (groupIdx1J*MT1J + (localSerial+(LID)*WG_0I*WG_1J)%MT1J)\n"
"\n"
"/* global non-tile indices being loaded */\n"
"\n"
"\n"
"/* local memory indices */\n"
"#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MT0I) )\n"
"#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MT1J) )\n"
"\n"
"/* local indices being written */\n"
"#define localA0I (localSerial / NUM_UNROLL_ITER)\n"
"#define localAK (localSerial % NUM_UNROLL_ITER)\n"
"#define localAStride (WG_0I*WG_1J/NUM_UNROLL_ITER)\n"
"#define localB1J ( localSerial / MT1J )\n"
"#define localBK ( localSerial % MT1J )\n"
"#define localBStride  (WG_0I*WG_1J)\n"
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
"  rA[0] = localA[offA + 0*WG_0I]; \\\n"
"  rA[1] = localA[offA + 1*WG_0I]; \\\n"
"  rA[2] = localA[offA + 2*WG_0I]; \\\n"
"  rA[3] = localA[offA + 3*WG_0I]; \\\n"
"  rA[4] = localA[offA + 4*WG_0I]; \\\n"
"  rA[5] = localA[offA + 5*WG_0I]; \\\n"
"  rB[0] = localB[offB + 0*WG_1J]; \\\n"
"  rB[1] = localB[offB + 1*WG_1J]; \\\n"
"  rB[2] = localB[offB + 2*WG_1J]; \\\n"
"  rB[3] = localB[offB + 3*WG_1J]; \\\n"
"  rB[4] = localB[offB + 4*WG_1J]; \\\n"
"  rB[5] = localB[offB + 5*WG_1J]; \\\n"
"  offA += MT0I; \\\n"
"  offB += MT1J; \\\n"
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
"__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))\n"
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
"  DATA_TYPE_STR_C rC[UT0I][UT1J] = {{0}};\n"
"  DATA_TYPE_STR_A rA[UT0I];\n"
"  DATA_TYPE_STR_B rB[UT1J];\n"
"\n"
"  /* allocate local memory */\n"
"  __local DATA_TYPE_STR_A localA[NUM_UNROLL_ITER*MT0I];\n"
"  __local DATA_TYPE_STR_B localB[NUM_UNROLL_ITER*MT1J];\n"
"\n"
"  /* c indices */\n"
"  unsigned int groupIdx0I = get_group_id(0); // d0, tensorA\n"
"  unsigned int groupIdx1J = get_group_id(1); // d1, tensorB\n"
"  unsigned int localIdx0I = get_local_id(0); // d0\n"
"  unsigned int localIdx1J = get_local_id(1); // d1\n"
"  unsigned int localSerial = localIdx0I + localIdx1J*WG_0I;\n"
"\n"
"  /* which global Cij index */\n"
"  unsigned int globalIdxC1J = groupIdx1J*MT1J + localIdx1J;\n"
"  unsigned int globalIdxC0I = groupIdx0I*MT0I + localIdx0I;\n"
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
"  if (globalIdxC0I + 0*WG_0I < size0I) {  if (globalIdxC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[0][0], beta) } }\n"
"  if (globalIdxC0I + 0*WG_0I < size0I) {  if (globalIdxC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[0][1], beta) } }\n"
"  if (globalIdxC0I + 0*WG_0I < size0I) {  if (globalIdxC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[0][2], beta) } }\n"
"  if (globalIdxC0I + 0*WG_0I < size0I) {  if (globalIdxC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[0][3], beta) } }\n"
"  if (globalIdxC0I + 0*WG_0I < size0I) {  if (globalIdxC1J + 4*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[0][4], beta) } }\n"
"  if (globalIdxC0I + 0*WG_0I < size0I) {  if (globalIdxC1J + 5*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 0*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[0][5], beta) } }\n"
"  if (globalIdxC0I + 1*WG_0I < size0I) {  if (globalIdxC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[1][0], beta) } }\n"
"  if (globalIdxC0I + 1*WG_0I < size0I) {  if (globalIdxC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[1][1], beta) } }\n"
"  if (globalIdxC0I + 1*WG_0I < size0I) {  if (globalIdxC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[1][2], beta) } }\n"
"  if (globalIdxC0I + 1*WG_0I < size0I) {  if (globalIdxC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[1][3], beta) } }\n"
"  if (globalIdxC0I + 1*WG_0I < size0I) {  if (globalIdxC1J + 4*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[1][4], beta) } }\n"
"  if (globalIdxC0I + 1*WG_0I < size0I) {  if (globalIdxC1J + 5*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 1*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[1][5], beta) } }\n"
"  if (globalIdxC0I + 2*WG_0I < size0I) {  if (globalIdxC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[2][0], beta) } }\n"
"  if (globalIdxC0I + 2*WG_0I < size0I) {  if (globalIdxC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[2][1], beta) } }\n"
"  if (globalIdxC0I + 2*WG_0I < size0I) {  if (globalIdxC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[2][2], beta) } }\n"
"  if (globalIdxC0I + 2*WG_0I < size0I) {  if (globalIdxC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[2][3], beta) } }\n"
"  if (globalIdxC0I + 2*WG_0I < size0I) {  if (globalIdxC1J + 4*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[2][4], beta) } }\n"
"  if (globalIdxC0I + 2*WG_0I < size0I) {  if (globalIdxC1J + 5*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 2*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[2][5], beta) } }\n"
"  if (globalIdxC0I + 3*WG_0I < size0I) {  if (globalIdxC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[3][0], beta) } }\n"
"  if (globalIdxC0I + 3*WG_0I < size0I) {  if (globalIdxC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[3][1], beta) } }\n"
"  if (globalIdxC0I + 3*WG_0I < size0I) {  if (globalIdxC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[3][2], beta) } }\n"
"  if (globalIdxC0I + 3*WG_0I < size0I) {  if (globalIdxC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[3][3], beta) } }\n"
"  if (globalIdxC0I + 3*WG_0I < size0I) {  if (globalIdxC1J + 4*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[3][4], beta) } }\n"
"  if (globalIdxC0I + 3*WG_0I < size0I) {  if (globalIdxC1J + 5*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 3*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[3][5], beta) } }\n"
"  if (globalIdxC0I + 4*WG_0I < size0I) {  if (globalIdxC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[4][0], beta) } }\n"
"  if (globalIdxC0I + 4*WG_0I < size0I) {  if (globalIdxC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[4][1], beta) } }\n"
"  if (globalIdxC0I + 4*WG_0I < size0I) {  if (globalIdxC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[4][2], beta) } }\n"
"  if (globalIdxC0I + 4*WG_0I < size0I) {  if (globalIdxC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[4][3], beta) } }\n"
"  if (globalIdxC0I + 4*WG_0I < size0I) {  if (globalIdxC1J + 4*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[4][4], beta) } }\n"
"  if (globalIdxC0I + 4*WG_0I < size0I) {  if (globalIdxC1J + 5*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 4*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[4][5], beta) } }\n"
"  if (globalIdxC0I + 5*WG_0I < size0I) {  if (globalIdxC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 0*WG_1J) ], alpha, rC[5][0], beta) } }\n"
"  if (globalIdxC0I + 5*WG_0I < size0I) {  if (globalIdxC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 1*WG_1J) ], alpha, rC[5][1], beta) } }\n"
"  if (globalIdxC0I + 5*WG_0I < size0I) {  if (globalIdxC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 2*WG_1J) ], alpha, rC[5][2], beta) } }\n"
"  if (globalIdxC0I + 5*WG_0I < size0I) {  if (globalIdxC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 3*WG_1J) ], alpha, rC[5][3], beta) } }\n"
"  if (globalIdxC0I + 5*WG_0I < size0I) {  if (globalIdxC1J + 4*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 4*WG_1J) ], alpha, rC[5][4], beta) } }\n"
"  if (globalIdxC0I + 5*WG_0I < size0I) {  if (globalIdxC1J + 5*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalIdxC0I + 5*WG_0I, globalIdxC1J + 5*WG_1J) ], alpha, rC[5][5], beta) } }\n"
"\n"
"}\n"
"";


#endif
