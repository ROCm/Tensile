
#pragma once

#if COBALT_BACKEND_OPENCL_ENABLED
//#include <CL/cl.h>
#endif

namespace Cobalt {

  class Dependency {
  public:
    void add( void *dependency );
    size_t size();
    void get( void *dependencies );
    void wait();

  protected:
  private:
    size_t numDependencies;
#if COBALT_BACKEND_OPENCL_ENABLED
    clEvent *events;
#endif
  }; // class Dependency

} // namespace Cobalt