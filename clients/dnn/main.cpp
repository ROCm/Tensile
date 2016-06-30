#include "Cobalt.h"
#include <cstdio>
#include <string>
#include <vector>
#include <array>

unsigned int H; // image height
unsigned int W; // image width
unsigned int C; // image num channels (RGB=3)
unsigned int N; // num images in minibatch
unsigned int R; // filter height
unsigned int S; // filter width
unsigned int K; // num filters (num output features)
unsigned int U; // virtical stride
unsigned int V; // horizontal stride
unsigned int P; // padding
unsigned int OH; // output height
unsigned int OW; // output width

CobaltTensor image;
CobaltTensor filter;
CobaltTensor output;

CobaltDeviceProfile deviceProfile;

CobaltProblem createProblemForConvolutionAsContraction();

/*
  tensor dimension order
  dim img flt out
  d0  W   S   OW
  d1  H   R   OH
  d2  C   C   K
  d3  N   K   N
 */

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char * argv[] ) {

  std::string logFilePath = Cobalt_DIR_PROBLEMS;
  cobaltSetup(logFilePath.c_str());
  CobaltProblem problem;
  CobaltSolution solution;
  CobaltStatus status;

  // device profile
  deviceProfile = cobaltCreateEmptyDeviceProfile();
  deviceProfile.numDevices = 1;
  sprintf(deviceProfile.devices[0].name, "Fiji" );

  /* Caffe Layer 1 */
  H = 32;
  W = 32;
  C = 3;
  N = 100;
  R = 5;
  S = 5;
  K = 32;
  U = 1;
  V = 1;
  P = 0;
  OH = (H-S+2*P)/U + 1;
  OW = (W-S+2*P)/V + 1;
  problem = createProblemForConvolutionAsContraction();
  status = cobaltGetSolutionForProblem( &solution, problem );
  cobaltStatusCheck(status);
  /* C[i:28,j:28,k:32,l:100] = Sum(m:3,n:5,o:5) A[o,n,i,j,m,l] * B[o,n,j,m,k,l] */

#if 0
  /* Caffe Layer 2 */
  H = 16;
  W = 16;
  C = 32;
  N = 2; // 100
  R = 5;
  S = 5;
  K = 32;
  U = 1;
  V = 1;
  P = 0;
  OH = (H-S+2*P)/U + 1;
  OW = (W-S+2*P)/V + 1;
  problem = createProblemForConvolutionAsContraction();
  status = cobaltGetSolutionForProblem( &solution, problem );
  cobaltStatusCheck(status);

  /* Caffe Layer 3 */
  H = 8;
  W = 8;
  C = 32;
  N = 2; // 100
  R = 5;
  S = 5;
  K = 64;
  U = 1;
  V = 1;
  P = 0;
  OH = (H-S+2*P)/U + 1;
  OW = (W-S+2*P)/V + 1;
  problem = createProblemForConvolutionAsContraction();
  status = cobaltGetSolutionForProblem( &solution, problem );
  cobaltStatusCheck(status);
#endif
  cobaltTeardown();
}

CobaltProblem createProblemForConvolutionAsContraction() {

  // reset tensors
  image = cobaltCreateEmptyTensor();
  filter = cobaltCreateEmptyTensor();
  output = cobaltCreateEmptyTensor();

  // data types
  image.dataType = cobaltDataTypeSingle;
  filter.dataType = cobaltDataTypeSingle;
  output.dataType = cobaltDataTypeSingle;


  /* image tensor (row major) */
  image.numDimensions = 6;

  // dim0: sub-image col
  image.dimensions[0].stride = 1;
  image.dimensions[0].size   = S;
  // dim1: sub-image row
  image.dimensions[1].stride = W;
  image.dimensions[1].size   = R;

  // dim2: sub-image col idx
  image.dimensions[2].stride = V*1;
  image.dimensions[2].size   = OW;
  // dim3: sub-image row idx
  image.dimensions[3].stride = U*W;
  image.dimensions[3].size   = OH;

  // dim4: channel
  image.dimensions[4].stride = H*W;
  image.dimensions[4].size   = C;
  // dim4: filter idx (free index of filter)
  // image.dimensions[4].stride = 0;
  // image.dimensions[4].size   = K;
  // dim5: minibatch
  image.dimensions[5].stride = H*W*C;
  image.dimensions[5].size   = N;


  /* filter tensor (row major) */
  filter.numDimensions = 6;

  // dim0: filter col
  filter.dimensions[0].stride = 1;
  filter.dimensions[0].size   = S;
  // dim1: filter row
  filter.dimensions[1].stride = S;
  filter.dimensions[1].size   = R;

  // dim2: sub-image col idx (free index of image)
  // filter.dimensions[2].stride = 0;
  // filter.dimensions[2].size   = OW;
  // dim3: sub-image row idx (dummy)
  filter.dimensions[2].stride = 0;
  filter.dimensions[2].size   = OH;

  // dim4: channel
  filter.dimensions[3].stride = R*S;
  filter.dimensions[3].size   = C;
  // dim5: filter idx
  filter.dimensions[4].stride = R*S*C;
  filter.dimensions[4].size   = K;
  // dim5: minibatch (dummy)
  filter.dimensions[5].stride = 0;
  filter.dimensions[5].size   = N;


  /* output tensor (row major) */
  output.numDimensions = 4;

  // dim0: output col (free index of image)
  output.dimensions[0].stride = 1;
  output.dimensions[0].size   = OW;
  // dim1: output row
  output.dimensions[1].stride = OW;
  output.dimensions[1].size   = OH;

  // dim4: filter idx (free index of filter)
  output.dimensions[2].stride = OW*OH;
  output.dimensions[2].size   = K;
  // dim5: minibatch
  output.dimensions[3].stride = OH*OW*K;
  output.dimensions[3].size   = N;

  /* O[i,j,k,n] = Sum[c,r,s] I[s,r,i,j,c,n] * F[s,r,j,c,k,n] */
  /*   0 1 2 3        4 5 6    6 5 0 1 4 3      6 5 1 4 2 3  */

  /* index assignments */
  unsigned int  imageIndexAssignments[6] = { 6, 5, 0, 1, 4, 3 };
  unsigned int filterIndexAssignments[6] = { 6, 5, 1, 4, 2, 3 };


  // create problem
  CobaltProblem problem;
  CobaltStatus status = cobaltCreateProblem(
    &problem,
    output,
    image,
    filter,
    imageIndexAssignments,
    filterIndexAssignments,
    cobaltOperationTypeContraction,
    cobaltDataTypeSingle, // alpha
    cobaltDataTypeSingle, // beta
    true, // use offsets?
    deviceProfile );
  cobaltStatusCheck(status);


  // validate problem
  CobaltStatus validationStatus = cobaltValidateProblem( problem );
  cobaltStatusCheck(validationStatus);
  if (validationStatus != cobaltStatusSuccess) {
    cobaltValidateProblem( problem );
  }

  // print problem
  unsigned int problemStringSize;
  cobaltProblemToString(problem, nullptr, &problemStringSize);
  char *problemString = new char[problemStringSize];
  cobaltProblemToString(problem, problemString, &problemStringSize);
  printf("%s\n", problemString);
  delete[] problemString;
  return problem;
}


