/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
