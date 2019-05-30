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

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#ifdef TENSILE_DEFAULT_SERIALIZATION
#include <Tensile/llvm/Loading.hpp>
#endif

namespace Tensile
{
    TENSILE_API Problem::~Problem()                 = default;
    TENSILE_API ProblemInputs::~ProblemInputs()     = default;
    TENSILE_API Hardware::Hardware()                = default;
    TENSILE_API Hardware::~Hardware()               = default;
    TENSILE_API Solution::~Solution()               = default;
    TENSILE_API SolutionAdapter::~SolutionAdapter() = default;

#ifdef TENSILE_DEFAULT_SERIALIZATION
    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        LoadLibraryFile(std::string const& filename)
    {
        return LLVMLoadLibraryFile<MyProblem, MySolution>(filename);
    }

    template std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>
        LoadLibraryFile<ContractionProblem, ContractionSolution>(std::string const& filename);
#endif
}
