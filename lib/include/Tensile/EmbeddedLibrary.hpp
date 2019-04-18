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

#pragma once

#ifdef TENSILE_DEFAULT_SERIALIZATION

#include <Tensile/Tensile.hpp>
#include <Tensile/Singleton.hpp>

namespace Tensile
{

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    TENSILE_API
    class EmbeddedLibrary: public LazySingleton<EmbeddedLibrary<MyProblem, MySolution>>
    {
    public:
        using Base = LazySingleton<EmbeddedLibrary<MyProblem, MySolution>>;

        /**
         * Constructs and returns a new SolutionLibrary instance from the static data.
         */
        static std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> NewLibrary();

        /**
         * Constructs (if necessary) and returns the shared SolutionLibrary for this problem type.
         */
        static std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> Get()
        {
            return Base::Instance().Library();
        }

        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> Library()
        {
            if(!m_library)
                m_library = NewLibrary();

            return m_library;
        }

    private:
        friend Base;
        EmbeddedLibrary() = default;

        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> m_library;
    };

}

#endif
