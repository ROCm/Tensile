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

#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/MapLibrary.hpp>

#include <Tensile/Serialization/Base.hpp>
#include <Tensile/Serialization/Properties.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename MyProblem, typename MySolution, typename Key, typename IO>
        struct MappingTraits<ProblemMapLibrary<MyProblem, MySolution, Key>, IO, SolutionMap<MySolution>>
        {
            using Library = ProblemMapLibrary<MyProblem, MySolution, Key>;
            using iot = IOTraits<IO>;

            static void mapping(IO & io, Library & lib, SolutionMap<MySolution> & ctx)
            {
                iot::setContext(io, &ctx);

                iot::mapRequired(io, "property", lib.property);
                iot::mapRequired(io, "map",      lib.map);
            }
        };

        template <typename MyProblem, typename MySolution, typename IO, typename Context>
        struct CustomMappingTraits<Tensile::LibraryMap<MyProblem, MySolution, std::string>, IO, Context>:
        public DefaultCustomMappingTraits<Tensile::LibraryMap<MyProblem, MySolution, std::string>, IO, Tensile::SolutionMap<MySolution>>
        {
        };
    }
}

