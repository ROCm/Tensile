
#include "Solution.h"
#include "Logger.h"


/*******************************************************************************
 * CobaltSolution:: constructor
 ******************************************************************************/
CobaltSolution::CobaltSolution( CobaltProblem inputProblem)
  : problem(inputProblem) {
}


/*******************************************************************************
 * LogSolution:: constructor
 ******************************************************************************/
LogSolution::LogSolution( CobaltProblem inputProblem)
  : CobaltSolution(inputProblem) {
}

/*******************************************************************************
 * LogSolution:: enqueue
 ******************************************************************************/
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

/*******************************************************************************
 * LogSolution:: toString - TODO
 ******************************************************************************/
std::string LogSolution::toString( size_t indentLevel ) const {
  //std::string state = Logger::indent(indentLevel);
  return "<LogSolution type=\"virtual\" />\n";
}

