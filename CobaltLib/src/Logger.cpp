
#include "Logger.h"

namespace Cobalt {

  
const std::string Logger::tensorTag = "Tensor";
const std::string Logger::dimensionTag = "Dim";
const std::string Logger::dimPairTag = "DimPair";
const std::string Logger::operationTag = "Operation";
const std::string Logger::deviceTag = "Device";
const std::string Logger::deviceProfileTag = "DeviceProfile";
const std::string Logger::problemTag = "Problem";
const std::string Logger::solutionTag = "Solution";
const std::string Logger::statusTag = "Status";
const std::string Logger::traceEntryTag = "Entry";
const std::string Logger::traceTag = "Trace";
const std::string Logger::assignSummaryTag = "SummaryOfAssign";
const std::string Logger::enqueueSummaryTag = "SummaryOfEnqueue";
const std::string Logger::documentTag = "ApplicationProblemProfile";
const std::string Logger::numDimAttr = "numDim";
const std::string Logger::operationAttr = "operation";
const std::string Logger::dimNumberAttr = "number";
const std::string Logger::dimStrideAttr = "stride";
const std::string Logger::nameAttr = "name";
const std::string Logger::typeEnumAttr = "typeEnum";
const std::string Logger::typeStringAttr = "typeString";


Logger::TraceEntry::TraceEntry(
    TraceEntryType inputType,
    const Problem & inputProblem,
    Status inputStatus
    )
    : type(inputType),
    problem(inputProblem),
    status(inputStatus) {
  //
}

std::string Logger::getStringFor( Logger::TraceEntryType type ) {

#define TYPE_TO_STRING_HANDLE_CASE(X) case X: return #X;
  switch( type ) {
    TYPE_TO_STRING_HANDLE_CASE(TraceEntryType::assignSolution);
    TYPE_TO_STRING_HANDLE_CASE(TraceEntryType::enqueueSolution);
  default:
    return "Error in getStringFor(TraceEntryType): no switch case for: " + std::to_string(type);
  };

}

std::string Logger::TraceEntry::toString( size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<" + Logger::traceEntryTag;
  state += " " + Logger::typeEnumAttr + "=\"" + std::to_string(type) + "\"";
  state += " " + Logger::typeStringAttr + "=\"" + Logger::getStringFor(type) + "\" >\n";
  state += problem.toString(indentLevel+1);
  state += status.toString(indentLevel+1);
  state += indent(indentLevel) + "</" + Logger::traceEntryTag + ">\n";
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
 * log assign solution
 ******************************************************************************/
void Logger::logAssignSolution(
    const Problem *problem,
    const Status & status ) {
  // create entry
  TraceEntry entry(assignSolution, *problem, status);
  // add to trace
  trace.push(entry); // append to end of list
  assignSummary[*problem]++;
}

/*******************************************************************************
 * log enqueue solution
 ******************************************************************************/
void Logger::logEnqueueSolution(
    const Problem *problem,
    const Status & status,
    const Control & ctrl ) {
  // create entry
  TraceEntry entry(enqueueSolution, *problem, status);
  // add to trace
  trace.push(entry); // append to end of list
  enqueueSummary[*problem]++;
}


/*******************************************************************************
 * indent - for spacing of xml output
 ******************************************************************************/
std::string Logger::indent(size_t level) {
  std::string indentStr = "";
  for (size_t i = 0; i < level; i++) {
    indentStr += "  ";
  }
  return indentStr;
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

std::string summaryEntryToString( std::string tag, Problem & problem, size_t count, size_t indentLevel ) {
  std::string state = Logger::indent(indentLevel);
  state += "<" + tag + " count=\"" + std::to_string(count) + "\" >\n";
  state += problem.toString(indentLevel+1);
  state += Logger::indent(indentLevel) + "</" + tag + ">\n";
  return state;
}

void Logger::writeSummary() {
  // close trace
  file << "</" + traceTag + ">\n\n";

  // assign summary
  file << comment("Summary of Problem::assignSolution()");
  file << "<" + assignSummaryTag + " numEntries=\"" + std::to_string(assignSummary.size()) + "\" >\n";
  std::map<Problem, unsigned long long>::iterator i;
  for ( i = assignSummary.begin(); i != assignSummary.end(); i++) {
    Problem problem = i->first;
    size_t count = i->second;

    // write state of entry
    std::string state = summaryEntryToString("AssignProblem", problem, count, 1);
    file << state;
  }
  file << "</" + assignSummaryTag + ">\n\n";

  // enqueue summary
  file << comment("Summary of Problem::enqueueSolution()");
  file << "<" + enqueueSummaryTag + " numEntries=\"" + std::to_string(enqueueSummary.size()) + "\" >\n";
  for ( i = enqueueSummary.begin(); i != enqueueSummary.end(); i++) {
    Problem problem = i->first;
    size_t count = i->second;

    // write state of entry
    std::string state = summaryEntryToString("EnqueueProblem", problem, count, 1);
    file << state;
  }
  file << "</" + enqueueSummaryTag + ">\n\n";

  // end document
  file << "</" + documentTag + ">\n";

}






// global logger object
Logger logger(LOG_FILE_PREFIX);

} // namespace Cobalt