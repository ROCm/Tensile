
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
  std::string state = indent(indentLevel) + "<Solution>\n";
  state += ::toStringXML(problem, indentLevel+1);
  state += indent(indentLevel) + "</Solution>";
  return state;
}

