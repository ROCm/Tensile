
#include "Logger.h"
#include "Tools.h"

namespace Cobalt {
  
/*******************************************************************************
 * TraceEntry:: constructor
 ******************************************************************************/
Logger::TraceEntry::TraceEntry() :
    type(TraceEntryType::getSolution),
    solution(nullptr),
    status(cobaltStatusProblemIsNull),
    validationStatus(statusNotValidated)
{ };

/*******************************************************************************
 * TraceEntry:: toString
 ******************************************************************************/
std::string Logger::TraceEntry::toString( size_t indentLevel ) const {
  std::string state = Cobalt::indent(indentLevel);
  state += "<TraceEntry";
  state += " enum=\"" + std::to_string(type) + "\"";
  state += " string=\"" + Logger::toString(type) + "\"";
  state += " status=\"" + Cobalt::toString(status) + "\" >\n";
  if (solution) {
    state += solution->toStringXML(indentLevel+1);
  }

  // validation
  if (validationStatus != statusNotValidated) {
    state += Cobalt::indent(indentLevel+1);
    state += "<Validation status=\"";
    if (validationStatus == statusValid) {
      state += "PASSED";
    } else {
      state += "FAILED";
    }
    state += "\" />\n";
  }

  // benchmarking
  for (size_t i = 0; i < benchmarkTimes.size(); i++) {
    state += Cobalt::indent(indentLevel+1);
    state += "<Benchmark time=\"";
    state += std::to_string(benchmarkTimes[i]);
    state += "\" />\n";
  }

  state += indent(indentLevel) + "</TraceEntry>\n";
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
  default:
    return "Error in toString(TraceEntryType): no switch case for: "
        + std::to_string(type);
  };
}

/*******************************************************************************
 * constructor
 ******************************************************************************/
Logger::Logger() {
}

void Logger::init( std::string logFilePath) {
  printf("Logger::init(%s)\n", logFilePath.c_str() );
  file.open( logFilePath, std::fstream::out );
  file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n";
  file << "<CobaltLog>\n\n";
  file << Logger::comment("Trace");
  //file << "<Trace>\n";
}


/*******************************************************************************
 * destructor
 ******************************************************************************/
Logger::~Logger() {
  printf("Logger::~Logger()\n");
#if USE_QUEUE
  flush();
#endif
  file << "</CobaltLog>\n";
  file.close();
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
    std::string tag, const Cobalt::Solution *solution,
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
  std::map<const Cobalt::Solution*, unsigned long long>::iterator i;
  for ( i = getSummary.begin(); i != getSummary.end(); i++) {
    const Cobalt::Solution *solution = i->first;
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
    const Cobalt::Solution *solution = i->first;
    size_t count = i->second;

    // write state of entry
    std::string state = summaryEntryToString("EnqueueSolution",
        solution, count, 1);
    file << state;
  }
  file << "</SummaryEnqueueSolution>\n\n";

  // end document
  file << "</CobaltLog>\n";

}
#endif

// global logger object
//Logger logger(LOG_FILE_PREFIX);

} // namespace Cobalt
