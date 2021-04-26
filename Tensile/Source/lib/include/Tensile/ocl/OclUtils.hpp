#ifndef OCL_UTILS_HPP
#define OCL_UTILS_HPP

/*
* A place to put utilities to help with OpenCL
* application workflow.
* Approximates Tensile 'module' loads of
* ELF binaries from file and formatting them as
* executable kernels
*/

#include <Tensile/ocl/OclFwd.hpp>
#include <iostream>

#define CL_CHECK(x)                                                                     \
    if(CL_SUCCESS != (x))                                                               \
    {                                                                                   \
        throw std::runtime_error(                                                       \
            concatenate("File: ", __FILE__, " (", __LINE__, ") CL Error code: ", (x))); \
    }

namespace Tensile
{

    namespace ocl
    {

        cl::Program clModuleLoad(cl::Context                    context,
                                 std::vector<cl::Device> const& devices,
                                 std::string const&             path);
        cl::Program clModuleLoadData(cl::Context                       context,
                                     std::vector<cl::Device> const&    devices,
                                     std::vector<unsigned char> const& bytes);

    } // namespace ocl

} // namespace Tensile

namespace cl
{
    std::ostream& operator<<(std::ostream& stream, Buffer buffer);
}

#endif // OCL_UTILS_HPP
