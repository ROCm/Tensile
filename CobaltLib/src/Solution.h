
#ifndef SOLUTION_H
#define SOLUTION_H

#include "Cobalt.h"

#include <string>


struct CobaltSolution {
public:

  CobaltSolution( CobaltProblem inputProblem );

  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltControl & ctrl ) = 0;

  virtual std::string toString( size_t indentLevel ) const = 0;

  CobaltProblem problem; // problem used to get this solution

}; // class Solution

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