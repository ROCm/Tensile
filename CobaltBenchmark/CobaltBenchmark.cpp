/*******************************************************************************
 * Cobalt Benchmark
 ******************************************************************************/

#include "Cobalt.h"
#include "Tools.h"
#include "CobaltSolutionCandidates.h"



/*******************************************************************************
 * timeSolution
 ******************************************************************************/
double timeSolution(
    CobaltSolution *solution,
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltControl &ctrl) {

  size_t numEnqueuesPerSample = 6;
  const size_t numSamples = 5;

  double sampleTimes[numSamples];
  Timer timer;

  for ( size_t sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {

    // start timer
    timer.start();
    for (size_t i = 0; i < numEnqueuesPerSample; i++) {
      cobaltEnqueueSolution(
          solution,
          tensorDataC,
          tensorDataA,
          tensorDataB,
          &ctrl );
    }
    // wait for queue
    // stop timer
    double time = timer.elapsed();
    sampleTimes[sampleIdx] = time;
  } // samples

  // for median, selection sort and take middle
  for (size_t i = 0; i < numSamples; i++) {
    size_t fastestIdx = i;
    for (size_t j = i+1; j < numSamples; j++) {
      if (sampleTimes[j] < sampleTimes[fastestIdx]) {
        fastestIdx = j;
      }
    }
    // swap i and fastest
    double tmp = sampleTimes[i];
    sampleTimes[i] = sampleTimes[fastestIdx];
    sampleTimes[fastestIdx] = tmp;
  }
  return sampleTimes[ numSamples/2 ];
}

/*******************************************************************************
 * main
 ******************************************************************************/
int main( void ) {

  // creat CobaltControl
  CobaltControl ctrl;

  CobaltTensorData tensorDataC;
  CobaltTensorData tensorDataA;
  CobaltTensorData tensorDataB;
  tensorDataC.data = nullptr;
  tensorDataC.offset = 0;
  tensorDataA.data = nullptr;
  tensorDataA.offset = 0;
  tensorDataB.data = nullptr;
  tensorDataB.offset = 0;

  // initialize Candidates
  initializeSolutionCandidates();

  size_t problemStartIdx = 0;
  size_t problemEndIdx = 0;
  size_t solutionStartIdx = 0;
  size_t solutionEndIdx;

  // for each problem
  for ( size_t problemIdx = problemStartIdx; problemIdx < problemEndIdx;
      problemIdx++ ) {

    solutionEndIdx = numSolutionsPerProblem[problemIdx];
    for ( size_t solutionIdx = solutionStartIdx; solutionIdx < solutionEndIdx;
        solutionIdx++ ) {

      // get solution candidate
      CobaltSolution *solution = solutionCandidates[ solutionIdx ];

      // time solution
      timeSolution( solution, tensorDataC, tensorDataA, tensorDataB, ctrl );

      // write time to result xml file

    } // solution loop

    solutionStartIdx = solutionEndIdx;
    
  } // problem loop

  return 0;
}



