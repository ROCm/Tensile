
#pragma once

#include "Dependency.h"

namespace Cobalt {
  
/*******************************************************************************
 * Control
 * - handle dependencies / queueing for backends (e.g. OpenCL cl_event)
 ******************************************************************************/
 class Control {
  public:
  protected:
  private:
    Dependency dependency;
#if COBALT_BACKEND_OPENCL_ENABLED
    cl_command_queue queue;
#endif

  }; // class Control

} // namespace Cobalt