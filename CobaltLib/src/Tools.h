#ifdef WIN32
#include "Windows.h"
#endif

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
  LARGE_INTEGER startTime;
  LARGE_INTEGER frequency; 
};