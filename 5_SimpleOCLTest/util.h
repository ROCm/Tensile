#ifndef __UTIL_H
#define __UTIL_H

#include<sys/time.h>
#include<iostream>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<CL/cl.h>

#define TEST_ITER 10

#define CALC_SIZE(x, f, p, s) (((x) + (p) * 2 - (f)) / (s) + 1); 

__inline__ double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}


__inline__ void reverse(float * x, int s)
{
  for(int i = 0; i < s/2; i++)
  {
    float tmp = x[i];
    x[i] = x[s-i-1];
    x[s-i-1] = tmp;
  }
}

#define CEIL(x, y) ( ((x) + (y) - 1) / (y) )
#define SAME(x, y, scale) (fabs(((x) - (y)) / fabs(scale)) < 0.001)

__inline__ bool verify(float * ans, float * res, int cnt, int isFull)
{
    float mx = 0;
    for(int i = 0; i < cnt; i++)
    {
      mx = (fabs(mx) > fabs(ans[i]))?fabs(mx):fabs(ans[i]);
    }
    for(int i = 0; i < cnt; i++)
    {
      if(!SAME(ans[i], res[i], mx))
      {
        fprintf(stderr, "Verification failed at %d (ans = %f, res = %f)\n", i, ans[i], res[i]);
        return false;
      }
    }
    return true;
}

#define CEIL(x, y) ( ((x) + (y) - 1) / (y) )
#define SAME(x, y, scale) (fabs(((x) - (y)) / fabs(scale)) < 0.001)

__inline__ bool verify_gemm(float * A, float * B, float * C, int X, int M, int N, int K, int tr_A, int tr_B)
{
  for(int test_id = 0; test_id < 10; test_id++)
  {
    int x = rand() % X;
    int m = rand() % M;
    int n = rand() % N;

    float sum = 0;
    for(int k = 0; k < K; k++)
      sum += A[(tr_A)?(x * M * K + k * M + m):(x * M * K + m * K + k)]
           * B[(tr_B)?(x * K * N + n * K + k):(x * K * N + k * N + n)];

    float ans = C[x * M * N + m * N + n];
    if(!SAME(sum, ans, ans))
    {
        fprintf(stderr, "Verification failed at (%d, %d, %d) (ans = %f, res = %f)\n", x, m, n, sum, ans);
        return false;
    }
  }
  return true;
}




#define chkCUDA(exp)                                           \
  {                                                            \
    cudaError_t status = (exp);                                \
    if (status != cudaSuccess) {                               \
      fprintf(stderr, "[%s] Error on line %d: %s\n",           \
        __FILE__, __LINE__, cudaGetErrorString(status));       \
      exit(99);                                                \
    }                                                          \
  }

#define chkCUDNN(exp)                                          \
  {                                                            \
    cudnnStatus_t status = (exp);                              \
    if (status != CUDNN_STATUS_SUCCESS) {                      \
      fprintf(stderr, "[%s] Error on line %d: %s\n",           \
        __FILE__, __LINE__, cudnnGetErrorString(status));      \
      exit(99);                                                \
    }                                                          \
  }

#define chkCUBLAS(exp)                                         \
  {                                                            \
    cublasStatus_t status = (exp);                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                     \
      fprintf(stderr, "[%s] Error on line %d\n",               \
        __FILE__, __LINE__);                                   \
      exit(99);                                                \
    }                                                          \
  }


#define CheckCl(exp) \
  do {\
    cl_int status = (exp);\
    if (status != CL_SUCCESS) {\
      fprintf(stderr, "[%s] Error on line %d: (code=%d) \n", \
          __FILE__, __LINE__, static_cast<int>(status)) ;\
      exit(EXIT_FAILURE);\
    }\
  } while (0)

inline char *GetSourceCode(const char *file_name, size_t *len) {
    char *source_code;
    size_t length;
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    length = (size_t)ftell(file);           
    rewind(file);

    source_code = (char *)malloc(length);
    fread(source_code, length, 1, file);

    fclose(file);

    *len = length;
    return source_code;
}

inline void CheckBuildProgram(cl_device_id device, cl_program program, cl_int err)
{
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    char *log;
    CheckCl(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
        0, NULL, &log_size));
    log = (char*) malloc (log_size + 1);
    CheckCl(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
        log_size, log, NULL));
    log[log_size] = '\0';
    printf("Compiler error:\n%s\n", log);
    free(log);
    exit(0);
  }
}
#endif
