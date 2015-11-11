
#include "Logger.h"

namespace Cobalt {

Logger::TraceEntry::TraceEntry(
    TraceEntryType inputType,
    const CobaltSolution *inputSolution,
    CobaltStatus inputStatus
    )
    : type(inputType),
    solution(inputSolution),
    status(inputStatus) {
}

std::string Logger::toString( Logger::TraceEntryType type ) {

#define TRACEENTRYTYPE_TO_STRING_HANDLE_CASE(X) case X: return #X;
  switch( type ) {
    TRACEENTRYTYPE_TO_STRING_HANDLE_CASE(TraceEntryType::getSolution);
    TRACEENTRYTYPE_TO_STRING_HANDLE_CASE(TraceEntryType::enqueueSolution);
  default:
    return "Error in toString(TraceEntryType): no switch case for: " + std::to_string(type);
  };
}

std::string Logger::TraceEntry::toString( size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<" + traceEntryTag;
  state += " " + typeEnumAttr + "=\"" + std::to_string(type) + "\"";
  state += " " + typeStringAttr + "=\"" + Logger::toString(type) + "\" >\n";
  if (solution) {
    state += solution->toString(indentLevel+1);
  }
  state += ::toString(status, indentLevel+1);
  state += indent(indentLevel) + "</" + traceEntryTag + ">\n";
  return state;
}


/*******************************************************************************
 * constructor
 ******************************************************************************/
Logger::Logger( std::string logFilePrefix) {
  printf("Logger::Logger(%s)\n", logFilePrefix.c_str() );
  std::string logFileName = logFilePrefix;
  file.open( logFileName, std::fstream::out );
  file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n";
  file << "<" + documentTag + ">\n\n";
  file << Logger::comment("Trace");
  file << "<" + traceTag + ">\n";
}


/*******************************************************************************
 * destructor
 ******************************************************************************/
Logger::~Logger() {
  printf("Logger::~Logger()\n");
  flush();
  writeSummary();
  file.close();
}

/*******************************************************************************
 * log get solution
 ******************************************************************************/
void Logger::logGetSolution(
    const CobaltSolution *solution,
    CobaltStatus status ) {
  // create entry
  TraceEntry entry(getSolution, solution, status);
  // add to trace
  trace.push(entry); // append to end of list
  if (solution) {
    getSummary[solution]++;
  }
}

/*******************************************************************************
 * log enqueue solution
 ******************************************************************************/
void Logger::logEnqueueSolution(
    const CobaltSolution *solution,
    CobaltStatus status,
    const CobaltControl *ctrl ) {
  // create entry
  CobaltProblem *problem = nullptr;
  TraceEntry entry(TraceEntryType::enqueueSolution, solution, status);
  // add to trace
  trace.push(entry); // append to end of list
  if (solution) {
    enqueueSummary[solution]++;
  }
}




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
void Logger::flush() {
  for ( ; !trace.empty(); trace.pop() ) {
    TraceEntry entry = trace.front();
    std::string state = entry.toString(1);
    file << state;
  }
}

std::string summaryEntryToString( std::string tag, const CobaltSolution *summary, size_t count, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<" + tag + " count=\"" + std::to_string(count) + "\" >\n";
  state += ::toString(summary->problem, indentLevel+1);
  state += indent(indentLevel) + "</" + tag + ">\n";
  return state;
}

void Logger::writeSummary() {
  // close trace
  file << "</" + traceTag + ">\n\n";

  // get summary
  file << comment("Summary of Problem::getSolution()");
  file << "<" + getSummaryTag + " numEntries=\"" + std::to_string(getSummary.size()) + "\" >\n";
  std::map<const CobaltSolution*, unsigned long long>::iterator i;
  for ( i = getSummary.begin(); i != getSummary.end(); i++) {
    const CobaltSolution *solution = i->first;
    size_t count = i->second;

    // write state of entry
    std::string state = summaryEntryToString("GetProblem", solution, count, 1);
    file << state;
  }
  file << "</" + getSummaryTag + ">\n\n";

  // enqueue summary
  file << comment("Summary of Problem::enqueueSolution()");
  file << "<" + enqueueSummaryTag + " numEntries=\"" + std::to_string(enqueueSummary.size()) + "\" >\n";
  for ( i = enqueueSummary.begin(); i != enqueueSummary.end(); i++) {
    const CobaltSolution *solution = i->first;
    size_t count = i->second;

    // write state of entry
    std::string state = summaryEntryToString("EnqueueProblem", solution, count, 1);
    file << state;
  }
  file << "</" + enqueueSummaryTag + ">\n\n";

  // end document
  file << "</" + documentTag + ">\n";

}








// global logger object
Logger logger(LOG_FILE_PREFIX);

} // namespace Cobalt
