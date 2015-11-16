
#ifndef SOLUTION_H
#define SOLUTION_H

#include "Cobalt.h"

#include <string>


/*******************************************************************************
 * CobaltSolution (abstract)
 ******************************************************************************/
struct CobaltSolution {

  CobaltSolution( CobaltProblem inputProblem );

  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltControl & ctrl ) = 0;

  virtual std::string toString( size_t indentLevel ) const = 0;

  CobaltProblem problem; // problem used to get this solution

}; // class Solution


/*******************************************************************************
 * LogSolution - used in LOG_ONLY mode
 ******************************************************************************/
class LogSolution : public CobaltSolution {
public:
  LogSolution( CobaltProblem inputProblem );

  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltControl & ctrl );

  virtual std::string toString( size_t indentLevel ) const;

};


#endif