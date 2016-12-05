/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#ifndef LOGGER_H
#define LOGGER_H

#define USE_QUEUE 0

#include "Tensile.h"
#include "Problem.h"
#include "Solution.h"
#include "StructOperations.h"

#include <string>
#include <queue>
#include <map>
#include <iostream>
#include <fstream>

namespace Tensile {

  typedef enum ValidationStatus_ {
    statusValid,
    statusInvalid,
    statusNotValidated
  } ValidationStatus;

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
 * - contains a Solution and an entry type
 ******************************************************************************/
  class TraceEntry {
  public:
    TraceEntryType type;
    const Tensile::Solution *solution;
    TensileStatus status;
    ValidationStatus validationStatus;
    std::vector<double> benchmarkTimes;
    TraceEntry();
    std::string toString( size_t indentLevel ) const;
  };

  
/*******************************************************************************
 * constructor - default is fine
 ******************************************************************************/
  Logger();
  void init( std::string logFileName);
  void close();
  
/*******************************************************************************
 * open - initialize logger by opening file for writing
 ******************************************************************************/
//void open( std::string fileName );

/*******************************************************************************
 * destructor
 * - flushes log state to file and closes it
 ******************************************************************************/
  ~Logger();
  
/*******************************************************************************
 * logGetSolution
 * - record a tensileGetSolution() call
 ******************************************************************************/
 // void logGetSolution(
 //     const Tensile::Solution *solution,
 //     TensileStatus status );

/*******************************************************************************
 * logEnqueueSolution
 * - record a Problem.enqueueSolution() call
 ******************************************************************************/
 // void logEnqueueSolution(
 //     const Tensile::Solution *solution,
 //     TensileStatus status,
 //     const TensileControl *ctrl );
  void log(const TraceEntry & entry);

/*******************************************************************************
 * toString xml
 * - returns string of spaces to indent objects' toString for pretty xml
 ******************************************************************************/
  static std::string comment(std::string comment);

private:
  
/*******************************************************************************
 * log state
 ******************************************************************************/
#if USE_QUEUE
  std::queue<TraceEntry> trace;
#endif
  //std::map<const Tensile::Solution*, unsigned long long,
  //    TensileSolutionPtrComparator> getSummary;
  //std::map<const Tensile::Solution*, unsigned long long,
  //    TensileSolutionPtrComparator> enqueueSummary;

/*******************************************************************************
 * xml file
 ******************************************************************************/
  std::ofstream file;
  
/*******************************************************************************
 * flush
 * - write trace if too large or upon completion
 ******************************************************************************/
#if USE_QUEUE
  void flush();
#endif

/*******************************************************************************
 * writeSummary
 * - write summaries to file upon completion
 ******************************************************************************/
//  void writeSummary();

}; // class Logger


/*******************************************************************************
 * global logger object
 ******************************************************************************/
extern Logger logger;

} // namespace Tensile


#endif

