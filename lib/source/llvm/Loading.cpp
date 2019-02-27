/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <Tensile/Tensile.hpp>
#include <Tensile/llvm/Loading.hpp>
#include <Tensile/llvm/YAML.hpp>

namespace Tensile
{

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> LLVMLoadLibraryFile(std::string const& filename)
    {
        std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> rv;

        auto inputFile = llvm::MemoryBuffer::getFile(filename);
        llvm::yaml::Input yin((*inputFile)->getMemBufferRef());

        yin >> rv;

        if(yin.error())
        {
            return nullptr;
        }

        return rv;
    }

    template
    std::shared_ptr<SolutionLibrary<GEMMProblem, GEMMSolution>>
    LLVMLoadLibraryFile<GEMMProblem, GEMMSolution>(std::string const& filename);

    template
    std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>
    LLVMLoadLibraryFile<ContractionProblem, ContractionSolution>(std::string const& filename);

}


