#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#ifdef WIN32
#include "Windows.h"
#else
#include <time.h>
#endif

namespace Cobalt {

/*******************************************************************************
 * Timer
 ******************************************************************************/
class Timer {
public:
  Timer();
  void start();
  double elapsed_sec();
  double elapsed_ms();
  double elapsed_us();

private:
#ifdef WIN32
  LARGE_INTEGER startTime;
  LARGE_INTEGER frequency; 
#else
  timespec startTime;
#endif
};


/*******************************************************************************
 * xml tags for toString
 ******************************************************************************/
std::string indent(size_t level);

bool factor( size_t input, unsigned int & a, unsigned int & b);


} // namespace

#define cobaltMin(a,b) (((a) < (b)) ? (a) : (b))
#define cobaltMax(a,b) (((a) > (b)) ? (a) : (b))

#endif
