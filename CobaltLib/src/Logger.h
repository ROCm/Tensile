
#pragma once
#include "Tensor.h"
#include "Operation.h"
#include "Device.h"
#include "Problem.h"
#include "Solution.h"
#include "Status.h"
#include <string>
#include <queue>
#include <map>
#include <iostream>
#include <fstream>

namespace Cobalt {
  
/*******************************************************************************
 * Logger
 * - keeps a trace and a summary of assignSolutions and enqueueSolutions
 * - writes trace and summary to a log file in xml format
 ******************************************************************************/
class Logger {

/*******************************************************************************
 * trace entry type
 * - assign or enqueue solution
 ******************************************************************************/
  enum TraceEntryType {
    assignSolution,
    enqueueSolution
  };
  
/*******************************************************************************
 * trace entry
 * - contains a Problem and an entry type
 ******************************************************************************/
  typedef struct TraceEntry {
    TraceEntryType type;
    Problem problem;
    Status status;
    TraceEntry(
        TraceEntryType inputType,
        const Problem & inputProblem,
        Status inputStatus);
    std::string toString( size_t indentLevel );
  } TraceEntry;

public:
  
/*******************************************************************************
 * constructor - default is fine
 ******************************************************************************/
  Logger(std::string logFilePrefix);
  
/*******************************************************************************
 * open - initialize logger by opening file for writing
 ******************************************************************************/
  void open( std::string fileName );

/*******************************************************************************
 * destructor
 * - flushes log state to file and closes it
 ******************************************************************************/
  ~Logger();
  
/*******************************************************************************
 * logAssignSolution
 * - record a Problem.assignSolution() call
 ******************************************************************************/
  void logAssignSolution(
      const Problem *problem,
      const Status & status );

/*******************************************************************************
 * logEnqueueSolution
 * - record a Problem.enqueueSolution() call
 ******************************************************************************/
  void logEnqueueSolution(
      const Problem *problem,
      const Status & status,
      const Control & ctrl );

/*******************************************************************************
 * indent
 * - returns string of spaces to indent objects' toString for pretty xml
 ******************************************************************************/
  static std::string indent(size_t level);
  static std::string comment(std::string comment);
  
/*******************************************************************************
 * xml tags for objects, store in central location here
 ******************************************************************************/
  static const std::string tensorTag;
  static const std::string dimensionTag;
  static const std::string dimPairTag;
  static const std::string operationTag;
  static const std::string deviceTag;
  static const std::string deviceProfileTag;
  static const std::string problemTag;
  static const std::string solutionTag;
  static const std::string statusTag;
  static const std::string traceEntryTag;
  static const std::string traceTag;
  static const std::string assignSummaryTag;
  static const std::string enqueueSummaryTag;
  static const std::string documentTag;
  static const std::string numDimAttr;
  static const std::string operationAttr;
  static const std::string dimNumberAttr;
  static const std::string dimStrideAttr;
  static const std::string nameAttr;
  static const std::string typeEnumAttr;
  static const std::string typeStringAttr;

private:
  
/*******************************************************************************
 * log state
 ******************************************************************************/
  std::queue<TraceEntry> trace;
  std::map<Problem, unsigned long long> assignSummary;
  std::map<Problem, unsigned long long> enqueueSummary;

/*******************************************************************************
 * xml file
 ******************************************************************************/
  std::ofstream file;
  
/*******************************************************************************
 * flush
 * - write trace if too large or upon completion
 ******************************************************************************/
  void flush();
  
/*******************************************************************************
 * writeSummary
 * - write summaries to file upon completion
 ******************************************************************************/
  void writeSummary();

  
/*******************************************************************************
 * getStringFor
 ******************************************************************************/
  static std::string getStringFor( TraceEntryType type );

}; // class Logger


/*******************************************************************************
 * global logger object
 ******************************************************************************/
extern Logger logger;

} // namespace Cobalt