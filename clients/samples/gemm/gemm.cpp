#include "Cobalt.h"


Cobalt::TensorDescriptor createDescriptorForMatrix( bool colMajor, bool trans, size_t numRows, size_t numCols );
void gemm( bool colMajor, bool transA, bool transB, size_t M, size_t N, size_t K );


/*******************************************************************************
 * main
 ******************************************************************************/
int main( char * argv[], int argc ) {

  for (size_t order = 1; order < 2; order++) {
    for (size_t transA = 0; transA < 2; transA++) {
      for (size_t transB = 0; transB < 2; transB++) {
        for (size_t size = 256; size <= 1024; size += 256) {
          gemm(
            order==1, // true means colMajor
            transA==1, // true means do transpose
            transB==1, // true means do transpose
            size, // M
            size, // N
            size);// K
        } // size
      } // transB
    } // transA
  } // order

  return 0;

} // main


/*******************************************************************************
 * gemm
 ******************************************************************************/
void gemm(
    bool colMajor,
    bool transA,
    bool transB,
    size_t M,
    size_t N,
    size_t K ) {

  // Matrix A
  size_t numRowsA;
  size_t numColsA;
  if (transA==colMajor) {
    numRowsA = K;
    numColsA = M;
  } else {
    numRowsA = M;
    numColsA = K;
  }

  // Matrix B
  size_t numRowsB;
  size_t numColsB;
  if (transB==colMajor) {
    numRowsB = N;
    numColsB = K;
  } else {
    numRowsB = K;
    numColsB = N;
  }

  // Matrix C
  size_t numRowsC = M;
  size_t numColsC = N;
  if (colMajor) {
    numRowsC = M;
    numColsC = N;
  } else {
    numRowsC = N;
    numColsC = M;
  }


  // matrix descriptors
  Cobalt::TensorDescriptor mdA = createDescriptorForMatrix(colMajor, transA, numRowsA, numColsA);
  Cobalt::TensorDescriptor mdB = createDescriptorForMatrix(colMajor, transB, numRowsB, numColsB);
  Cobalt::TensorDescriptor mdC = createDescriptorForMatrix(colMajor,  false, numRowsC, numColsC);

  // gemm operation descriptor
  Cobalt::OperationDescriptor operation(Cobalt::OperationType::TensorContraction, 1);
  operation.dimensions[0].a = 1;
  operation.dimensions[0].b = 0;

  // device
  Cobalt::DeviceProfile deviceProfile(1);
  deviceProfile.devices[0].deviceName = "Hawaii";
  deviceProfile.devices[0].numComputeUnits = 44;
  deviceProfile.devices[0].clockFrequency = 900; // MHz


  // construct problem
  Cobalt::ProblemDescriptor problem(mdA, mdB, mdC, operation, deviceProfile);

  // assign solution
  problem.assignSolution();

  // control
  Cobalt::Control ctrl;

  // data
  Cobalt::TensorData dataA;
  Cobalt::TensorData dataB;
  Cobalt::TensorData dataC;

  // enqueue solution
  problem.enqueueSolution(dataA, dataB, dataC, ctrl);
}

// TODO - debug this
Cobalt::TensorDescriptor createDescriptorForMatrix(
    bool colMajor,
    bool trans,
    size_t numRows,
    size_t numCols
    ) {
  Cobalt::TensorDescriptor desc(2);
  if (colMajor != trans) {
    // 0th dimension is col
    desc.dimensions[0].stride = 1; // incr to get to next row
    desc.dimensions[0].size = numRows; // how many times can we incr in this dimension
    // 1th dimensions is row
    desc.dimensions[1].stride = numRows; // incr to get to next col
    desc.dimensions[1].size = numCols; // how many times can we incr in this dimension
  } else {
    // 0th dimension is col
    desc.dimensions[0].stride = 1; // incr to get to next col
    desc.dimensions[0].size = numCols; // how many time can we incr in this dimension
    // 1th dimensions is row
    desc.dimensions[1].stride = numCols; // incr to get to next row
    desc.dimensions[1].size = numRows; // how many times can we incr in this dimension
  }
  return desc;
}