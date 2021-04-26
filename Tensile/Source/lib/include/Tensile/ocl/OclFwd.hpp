#ifndef OCL_FWD_HPP
#define OCL_FWD_HPP

/*
* This file is intended to fwd declare OpenCL objects
* so that we can reference them in header files and
* not have to pull in the OpenCL headers. We want to
* limit use of OpenCL to actual compilation units (cpp)
* where they are needed.
*/

// Fwd declarations
namespace cl
{
    class Program;
    class Context;
    class Device;
    class Buffer;
} // namespace cl

#endif // OCL_FWD_HPP
