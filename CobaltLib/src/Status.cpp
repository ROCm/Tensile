
#include "Status.h"
#include "Logger.h"

namespace Cobalt {



void Status::add( StatusCode inputCode ) {
  codes.push_back( inputCode );
}

void Status::add( const std::string & inputMessage ) {
  messages.push_back( inputMessage );
}

std::vector<StatusCode> Status::getCodes() const {
  return codes;
}

std::string Status::getStringForCode(StatusCode code) const {
  
#define STATUS_TO_STRING_HANDLE_CASE(X) case X: return #X;
  switch( code ) {
    STATUS_TO_STRING_HANDLE_CASE(StatusCode::success);
    STATUS_TO_STRING_HANDLE_CASE(StatusCode::solutionsDisabled);
  default:
    return "Error in Status::getStringForCode(): no switch case for StatusCode " + std::to_string(code);
  };

} // getStringForCode

bool Status::isSuccess() const {
  return codes.size() == 0;
}


std::string Status::toString( size_t indentLevel ) const {
  std::string state = Logger::indent(indentLevel);
  state += "<" + Logger::statusTag;
  state += " numCodes=\"" + std::to_string(codes.size()) + "\"";
  state += " numMsgs=\"" + std::to_string(messages.size()) + "\"";
  state += ">\n";
  if ( isSuccess() ) {
      state += Logger::indent(indentLevel+1) + "<Code " + Logger::typeEnumAttr + "=\"" + std::to_string(StatusCode::success) + "\"";
      state += " " + Logger::typeStringAttr + "=\"" + getStringForCode(StatusCode::success) + "\" />";
  } else {
    for (size_t i = 0; i < codes.size(); i++) {
      state += Logger::indent(indentLevel+1) + "<Code " + Logger::typeEnumAttr + "=\"" + std::to_string(codes[i]) + "\"";
      state += " " + Logger::typeStringAttr + "=\"" + getStringForCode(codes[i]) + "\" />\n";
    }
  }
  for (size_t i = 0; i < messages.size(); i++) {
    state += Logger::indent(indentLevel+1) + "<Msg str=\"" + messages[i] + "\" />\n";
  }
  state += Logger::indent(indentLevel) + "</" + Logger::statusTag + ">\n";
  return state;
} // toString

} // namespace Cobalt