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


#include "Logger.h"
#include "Tools.h"

namespace Tensile {
  
/*******************************************************************************
 * TraceEntry:: constructor
 ******************************************************************************/
Logger::TraceEntry::TraceEntry() :
    type(TraceEntryType::getSolution),
    solution(nullptr),
    status(tensileStatusSuccess),
    validationStatus(statusNotValidated)
{ };

/*******************************************************************************
 * TraceEntry:: toString
 ******************************************************************************/
std::string Logger::TraceEntry::toString( size_t indentLevel ) const {
  std::string state = Tensile::indent(indentLevel);
  state += "<TE>\n";
  if (solution) {
    state += solution->toStringXML(indentLevel+1);
  }

  // validation
  if (validationStatus != statusNotValidated) {
    state += Tensile::indent(indentLevel+1);
    state += "<V s=\"";
    if (validationStatus == statusValid) {
      state += "P";
    } else {
      state += "F";
    }
    state += "\" />\n";
  }

  // benchmarking
  for (size_t i = 0; i < benchmarkTimes.size(); i++) {
    state += Tensile::indent(indentLevel+1);
    state += "<B t=\"";
    state += std::to_string(benchmarkTimes[i]);
    state += "\" u=\"ms\" />\n";
  }

  state += indent(indentLevel) + "</TE>\n";
  return state;
}

/*******************************************************************************
 * TraceEntryType:: toString
 ******************************************************************************/
std::string Logger::toString( Logger::TraceEntryType type ) {

#define TRACEENTRYTYPE_TO_STRING_HANDLE_CASE(X) case X: return #X;
  switch( type ) {
    TRACEENTRYTYPE_TO_STRING_HANDLE_CASE(TraceEntryType::getSolution);
    TRACEENTRYTYPE_TO_STRING_HANDLE_CASE(TraceEntryType::enqueueSolution);
  };
  printf("Invalid Logger::TraceEntryType\n");
  std::abort();
}

/*******************************************************************************
 * constructor
 ******************************************************************************/
Logger::Logger() {
}

void Logger::init( std::string inputLogFilePath) {
  size_t slashIdx = inputLogFilePath.find_last_of('/');
  size_t backslashIdx = inputLogFilePath.find_last_of('\\');
  size_t splitIdx = tensileMin(slashIdx, backslashIdx);
  splitIdx++;
  std::string logFilePath = inputLogFilePath.substr(0, splitIdx);
  std::string logFileName = inputLogFilePath.substr(splitIdx, inputLogFilePath.size() - splitIdx);

  // append log to list of xmls
  std::string listFileName = logFilePath + "list_of_xmls.txt";
  std::ofstream listFile;
  listFile.open( listFileName, std::fstream::app );
  listFile << logFileName << std::endl;
  listFile.close();

  //printf("Logger::init(%s)\n", logFilePath.c_str() );
  file.open( logFilePath+logFileName, std::fstream::out );
  file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n";
  file << "<TensileLog>\n\n";
  file << Logger::comment("Trace");
  //file << "<Trace>\n";
}

void Logger::close() {
  file << "</TensileLog>\n";
  file.close();
}


/*******************************************************************************
 * destructor
 ******************************************************************************/
Logger::~Logger() {
#if USE_QUEUE
  flush();
#endif
}


/*******************************************************************************
* log Entry
******************************************************************************/
void Logger::log( const TraceEntry & entry) {
  // add to trace
#if USE_QUEUE
  trace.push(entry);
  flush();
#else
  std::string state = entry.toString(1);
  file << state;
#endif
  file.flush();
}

/*******************************************************************************
 * comment
 ******************************************************************************/
std::string Logger::comment(std::string comment) {
  std::string xmlComment;
  xmlComment += "<!-- ~~~~~~~~~~~~~~~~ ";
  xmlComment += comment;
  xmlComment += " ~~~~~~~~~~~~~~~~ -->\n";
  return xmlComment;
}

/*******************************************************************************
 * flush - flush trace to file
 ******************************************************************************/
#if USE_QUEUE
void Logger::flush() {
  for ( ; !trace.empty(); trace.pop() ) {
    TraceEntry entry = trace.front();
    std::string state = entry.toString(1);
    file << state;
  }
}
#endif

/*******************************************************************************
 * summaryEntryToString
 ******************************************************************************/
std::string summaryEntryToString(
    std::string tag, const Tensile::Solution *solution,
    size_t count, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<" + tag + " count=\"" + std::to_string(count) + "\" >\n";
  state += solution->toStringXML(indentLevel+1);
  state += indent(indentLevel) + "</" + tag + ">\n";
  return state;
}

/*******************************************************************************
 * writeSummary
 ******************************************************************************/
#if 0
void Logger::writeSummary() {
  // close trace
  file << "</Trace>\n\n";

  // get summary
  file << comment("Summary of Problem::getSolution()");
  file << "<SummaryGetSolution numEntries=\""
      + std::to_string(getSummary.size()) + "\" >\n";
  std::map<const Tensile::Solution*, unsigned long long>::iterator i;
  for ( i = getSummary.begin(); i != getSummary.end(); i++) {
    const Tensile::Solution *solution = i->first;
    size_t count = i->second;

    // write state of entry
    std::string state = summaryEntryToString("GetSolution", solution, count, 1);
    file << state;
  }
  file << "</SummaryGetSolution>\n\n";

  // enqueue summary
  file << comment("Summary of Problem::enqueueSolution()");
  file << "<SummaryEnqueueSolution numEntries=\""
      + std::to_string(enqueueSummary.size()) + "\" >\n";
  for ( i = enqueueSummary.begin(); i != enqueueSummary.end(); i++) {
    const Tensile::Solution *solution = i->first;
    size_t count = i->second;

    // write state of entry
    std::string state = summaryEntryToString("EnqueueSolution",
        solution, count, 1);
    file << state;
  }
  file << "</SummaryEnqueueSolution>\n\n";

  // end document
  file << "</TensileLog>\n";

}
#endif

// global logger object
//Logger logger(LOG_FILE_PREFIX);

} // namespace Tensile

