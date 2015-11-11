
#include "Solution.h"
#include "Logger.h"


CobaltSolution::CobaltSolution( CobaltProblem inputProblem)
  : problem(inputProblem) {
}


LogSolution::LogSolution( CobaltProblem inputProblem)
  : CobaltSolution(inputProblem) {
}

CobaltStatus LogSolution::enqueue(
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltTensorData tensorDataC,
    CobaltControl & ctrl ) {
  CobaltStatus status;
  status.numCodes = 0;
  printf("CobaltSolution::enqueue() virtual function not overrided\n");
  return status;
}

  std::string LogSolution::toString( size_t indentLevel ) const {
    //std::string state = Logger::indent(indentLevel);
    return "<LogSolution type=\"virtual\" />\n";
  }

