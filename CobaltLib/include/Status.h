
#pragma once

#include <string>
#include <vector>

namespace Cobalt {

enum StatusCode {
  // success
  success = 0,

  // correctness errors start here
  correctnessErrors = 1,
  solutionsDisabled,

  // performance warnings start here
  performanceWarnings = 1024,
  assignSolutionAlreadyRequested,
  problemSizeTooSmall
};

class Status {
public:
  
/*******************************************************************************
 * constructor - default is fine
 ******************************************************************************/
  //Status();

/*******************************************************************************
 * add code
 ******************************************************************************/
  void add( StatusCode inputCode );
  
/*******************************************************************************
 * add message
 ******************************************************************************/
  void add( const std::string & inputMessage );

/*******************************************************************************
 * getCodes
 ******************************************************************************/
  std::vector<StatusCode> getCodes() const;
  std::string getStringForCode( StatusCode code ) const;
  
/*******************************************************************************
 * isSuccess
 ******************************************************************************/
  bool isSuccess() const;

/*******************************************************************************
 * toString for writing xml
 ******************************************************************************/
  std::string toString( size_t indentLevel ) const;

private:
  std::vector<StatusCode> codes;
  std::vector<std::string> messages;
};


} // namespace Cobalt