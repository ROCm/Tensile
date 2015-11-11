
#ifndef LOGGER_H
#define LOGGER_H

#include "Cobalt.h"
#include "Solution.h"
#include "StructOperations.h"

#include <string>
#include <queue>
#include <map>
#include <iostream>
#include <fstream>

namespace Cobalt {
  

/*******************************************************************************
 * Logger
 * - keeps a trace and a summary of getSolutions and enqueueSolutions
 * - writes trace and summary to a log file in xml format
 ******************************************************************************/
class Logger {
  
public:

/*******************************************************************************
 * trace entry type
 * - get or enqueue solution
 ******************************************************************************/
  typedef enum TraceEntryType_ {
    getSolution,
    enqueueSolution
  } TraceEntryType;
  static std::string toString( TraceEntryType type );
  
/*******************************************************************************
 * trace entry
 * - contains a Problem and an entry type
 ******************************************************************************/
  class TraceEntry {
  public:
    TraceEntryType type;
    const CobaltSolution *solution;
    CobaltStatus status;
    TraceEntry(
        TraceEntryType inputType,
        const CobaltSolution *inputSolution,
        CobaltStatus inputStatus);
    std::string toString( size_t indentLevel );
  };

  
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
 * logGetSolution
 * - record a cobaltGetSolution() call
 ******************************************************************************/
  void logGetSolution(
      const CobaltSolution *solution,
      CobaltStatus status );

/*******************************************************************************
 * logEnqueueSolution
 * - record a Problem.enqueueSolution() call
 ******************************************************************************/
  void logEnqueueSolution(
      const CobaltSolution *solution,
      CobaltStatus status,
      const CobaltControl *ctrl );

/*******************************************************************************
 * toString xml
 * - returns string of spaces to indent objects' toString for pretty xml
 ******************************************************************************/
  static std::string comment(std::string comment);

private:
  
/*******************************************************************************
 * log state
 ******************************************************************************/
  std::queue<TraceEntry> trace;
  std::map<const CobaltSolution*, unsigned long long, CobaltSolutionPtrComparator> getSummary;
  std::map<const CobaltSolution*, unsigned long long, CobaltSolutionPtrComparator> enqueueSummary;

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

  
}; // class Logger


/*******************************************************************************
 * global logger object
 ******************************************************************************/
extern Logger logger;

} // namespace Cobalt


#endif
