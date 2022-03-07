#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "util.h"

#define NUM_TEST 100
#define NUM_WARMUP 10

typedef cl_float16 half;

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_int status, err;
cl_event events[NUM_TEST];


void clInit(const char * kernel_file, const char * kernel_name)
{
  CheckCl(clGetPlatformIDs(1, &platform, NULL));
  CheckCl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CheckCl(err);
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  CheckCl(err);
  size_t kernel_source_size;
  char *kernel_source = GetSourceCode(kernel_file, &kernel_source_size);
  program = clCreateProgramWithBinary(context, 1, &device, &kernel_source_size,
    (const unsigned char**)&kernel_source, &status, &err);
  CheckCl(err);
  err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", NULL, NULL);
  CheckBuildProgram(device, program, err); 
}

int main(int argc, char ** argv)
{
  const char * kernel_file = (argv[1]);
  const char * kernel_name = (argv[2]);
  int M = atoi(argv[3]);
  int N = atoi(argv[4]);
  int K = atoi(argv[5]);
  int TILE_M = atoi(argv[6]);
  int TILE_N = atoi(argv[7]);
  int TILE_K = atoi(argv[8]);
  size_t lws = 256;

  srand(0);

  clInit(kernel_file, kernel_name);

  half * A = (half *)malloc(sizeof(half) * M * K);
  half * B = (half *)malloc(sizeof(half) * K * N);
  half * C = (half *)malloc(sizeof(half) * M * N);

  cl_mem d_A, d_B, d_C;
  cl_kernel kernel;
  d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(half) * M * K, A, &err);
  CheckCl(err);
  d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(half) * K * N, B, &err);
  CheckCl(err);
  d_C = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(half) * M * N, NULL, &err);
  CheckCl(err);


  kernel = clCreateKernel(program, kernel_name, &err);
  CheckCl(err);

  //TODO-wookeun find the proper gws/lws.. 
  size_t gws = ((M + TILE_M - 1) / TILE_M) * ((N + TILE_N - 1) / TILE_N) * lws;
  double gflop = ((double)M * (double)N * (double)K) * 2 / 1e9f;

  cl_ulong sizeC = M * N;
  cl_ulong sizeA = M * K;
  cl_ulong sizeB = K * N;
  //TODO-wookeun value type fp16 for alpha, beta?
  cl_float alpha = 1;
  cl_float beta = 0;
  cl_uint strideD0, strideD1, strideC0, strideC1, strideA0, strideA1, strideB0, strideB1;
  cl_uint SizesFree0, SizesFree1, SizesFree2, SizesSum0;
  cl_int OrigStaggerUIter;
  cl_uint NumWorkGroups0, NumWorkGroups1;
  cl_uint NumFullBlocks, WgmRemainder1, MagicNumberWgmRemainder1;
  cl_uint OffsetD, OffsetC, OffsetA, OffsetB;
  cl_uint padding;

  //TODO-wookeun fix the following paramteters..
  SizesFree0 = M;
  SizesFree1 = N;
  SizesFree2 = 1;
  SizesSum0 = K;
  strideD0 = N;
  strideD1 = M * N;
  strideC0 = strideD0;
  strideC1 = strideC1;
  strideA0 = K;
  strideA1 = M * K;
  strideB0 = N;
  strideB1 = K * N;
  OrigStaggerUIter = (K + TILE_K - 1) / TILE_K;
  NumWorkGroups0 = gws / lws;
  NumWorkGroups1 = 1;
  NumFullBlocks = ((M / TILE_M) * (N / TILE_N));
  WgmRemainder1 = 0;
  MagicNumberWgmRemainder1 = 0;
  OffsetD = OffsetC = OffsetB = OffsetA = padding = 0;
  


  CheckCl(clSetKernelArg(kernel, 0, sizeof(cl_ulong), &sizeC)); 
  CheckCl(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &sizeA)); 
  CheckCl(clSetKernelArg(kernel, 2, sizeof(cl_ulong), &sizeB)); 
  CheckCl(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_C));  
  CheckCl(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_C));  
  CheckCl(clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_A));  
  CheckCl(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_B));  
  CheckCl(clSetKernelArg(kernel, 7, sizeof(cl_float), &alpha));  
  CheckCl(clSetKernelArg(kernel, 8, sizeof(cl_float), &beta));  
  CheckCl(clSetKernelArg(kernel, 9, sizeof(cl_uint), &strideD0));  
  CheckCl(clSetKernelArg(kernel, 10, sizeof(cl_uint), &strideD1));  
  CheckCl(clSetKernelArg(kernel, 11, sizeof(cl_uint), &strideC0));  
  CheckCl(clSetKernelArg(kernel, 12, sizeof(cl_uint), &strideC1));  
  CheckCl(clSetKernelArg(kernel, 13, sizeof(cl_uint), &strideA0));  
  CheckCl(clSetKernelArg(kernel, 14, sizeof(cl_uint), &strideA1));  
  CheckCl(clSetKernelArg(kernel, 15, sizeof(cl_uint), &strideB0));  
  CheckCl(clSetKernelArg(kernel, 16, sizeof(cl_uint), &strideB1));  
  CheckCl(clSetKernelArg(kernel, 17, sizeof(cl_uint), &SizesFree0));  
  CheckCl(clSetKernelArg(kernel, 18, sizeof(cl_uint), &SizesFree1));  
  CheckCl(clSetKernelArg(kernel, 19, sizeof(cl_uint), &SizesFree2));  
  CheckCl(clSetKernelArg(kernel, 20, sizeof(cl_uint), &SizesSum0));  
  CheckCl(clSetKernelArg(kernel, 21, sizeof(cl_int), &OrigStaggerUIter));  
  CheckCl(clSetKernelArg(kernel, 22, sizeof(cl_uint), &NumWorkGroups0));  
  CheckCl(clSetKernelArg(kernel, 23, sizeof(cl_uint), &NumWorkGroups1));  
  CheckCl(clSetKernelArg(kernel, 24, sizeof(cl_uint), &NumFullBlocks));  
  CheckCl(clSetKernelArg(kernel, 25, sizeof(cl_uint), &WgmRemainder1));  
  CheckCl(clSetKernelArg(kernel, 26, sizeof(cl_uint), &MagicNumberWgmRemainder1));  
  CheckCl(clSetKernelArg(kernel, 27, sizeof(cl_uint), &OffsetD));  
  CheckCl(clSetKernelArg(kernel, 28, sizeof(cl_uint), &OffsetC));  
  CheckCl(clSetKernelArg(kernel, 29, sizeof(cl_uint), &OffsetA));  
  CheckCl(clSetKernelArg(kernel, 30, sizeof(cl_uint), &OffsetB));  
  CheckCl(clSetKernelArg(kernel, 31, sizeof(cl_uint), &padding));  


  for(int i = 0; i < NUM_WARMUP; i++)
    CheckCl(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, &lws, 0, NULL, NULL));
  clFinish(queue);

  double d1 = get_time(); 
  for(int i = 0; i < NUM_TEST; i++)
  {
    CheckCl(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, &lws, 0, NULL, events + i));
    clFinish(queue);
  }
  double d2 = get_time();
  double exec_time = 1000 * (d2 - d1) / NUM_TEST;

  cl_ulong global_start;
  cl_ulong times_sum[5] = {0, 0, 0, 0, 0};
  for(int i = 0; i < NUM_TEST; i++)
  {
    cl_event event = events[i];
    cl_ulong times[4];
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), times + 0, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), times + 1, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), times + 2, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), times + 3, NULL);
    times_sum[2] += times[3] - times[2];
  }
  
  double execute_finish_latency = (((double) times_sum[2]) / NUM_TEST) / 1000000;

  printf("Real: %.3f ms, %.3f TFLOPS\n", exec_time,  gflop / exec_time); 
  printf("Kernel: %.3f ms, %.3f TFLOPS\n", execute_finish_latency, gflop / execute_finish_latency);
  
  return 0;
}
