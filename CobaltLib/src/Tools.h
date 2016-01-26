#ifdef WIN32
#include "Windows.h"
#endif

/*******************************************************************************
 * Timer
 ******************************************************************************/
class Timer {
  public:
  void start();
  double elapsed();

  private:
  LARGE_INTEGER startTime;
  LARGE_INTEGER frequency; 
};