
#include <CL/cl.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/ocl/OclUtils.hpp>

// std includes
#include <fstream>
#include <string>
#include <vector>

namespace Tensile
{

    namespace ocl
    {

        cl::Program clModuleLoad(cl::Context                    context,
                                 std::vector<cl::Device> const& devices,
                                 std::string const&             path)
        {
            std::ifstream file(path, std::ios::binary | std::ios::in | std::ios::ate);
            if(file.fail())
            {
                throw std::runtime_error(concatenate("Failed to load binary file: ", path));
            }

            std::vector<unsigned char> bytes(file.tellg());
            file.seekg(0, std::ios::beg);
            file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
            file.close();

            return clModuleLoadData(context, devices, bytes);
        }

        cl::Program clModuleLoadData(cl::Context                       context,
                                     std::vector<cl::Device> const&    devices,
                                     std::vector<unsigned char> const& bytes)
        {
            std::vector<cl_int> binErrs;
            cl_int              err = 0;
            cl::Program         program
                = cl::Program{context, devices, {{bytes.data(), bytes.size()}}, &binErrs, &err};

            if(err != CL_SUCCESS)
            {
                throw std::runtime_error(
                    concatenate("Failed to create program from binary file with code: ", err));
            }

            for(auto& binErr : binErrs)
            {
                if(binErr != CL_SUCCESS)
                {
                    throw std::runtime_error(concatenate("Binary error with code: ", binErr));
                }
            }

            if((err = program.build(devices)) != CL_SUCCESS)
            {
                auto buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);

                throw std::runtime_error(
                    concatenate("Build error with code: ", err, ": ", buildLog));
            }

            return program;
        }

    } // namespace ocl

} // namespace Tensile

namespace cl
{
    std::ostream& operator<<(std::ostream& stream, Buffer buffer)
    {
        return stream << static_cast<void*>(buffer());
    }
} // namespace cl
