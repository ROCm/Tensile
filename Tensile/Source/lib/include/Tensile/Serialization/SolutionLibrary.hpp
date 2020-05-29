/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
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

#pragma once

#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/CachingLibrary.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/MapLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>

#include <Tensile/Serialization/Base.hpp>
#include <Tensile/Serialization/Predicates.hpp>

#include <Tensile/Serialization/ExactLogicLibrary.hpp>
#include <Tensile/Serialization/GranularitySelectionLibrary.hpp>
#include <Tensile/Serialization/MapLibrary.hpp>
#include <Tensile/Serialization/MatchingLibrary.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<std::shared_ptr<SolutionLibrary<ContractionProblem>>, IO>
            : public BaseClassMappingTraits<SolutionLibrary<ContractionProblem>, IO, false>
        {
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct SubclassMappingTraits<SolutionLibrary<MyProblem, MySolution>, IO>
            : public DefaultSubclassMappingTraits<
                  SubclassMappingTraits<SolutionLibrary<MyProblem, MySolution>, IO>,
                  SolutionLibrary<MyProblem, MySolution>,
                  IO>
        {
            using Self = SubclassMappingTraits<SolutionLibrary<MyProblem, MySolution>, IO>;
            using Base
                = DefaultSubclassMappingTraits<Self, SolutionLibrary<MyProblem, MySolution>, IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static typename Base::SubclassMap GetSubclasses()
            {
                return typename Base::SubclassMap(
                    {Base::template Pair<SingleSolutionLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<HardwareSelectionLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<ProblemSelectionLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<ProblemMapLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<ProblemMatchingLibrary<MyProblem, MySolution>>(),
                     Base::template Pair<GranularitySelectionLibrary<MyProblem, MySolution>>()});
            }
        };

        template <typename MyProblem, typename MySolution, typename IO>
        using dsmt = SubclassMappingTraits<SolutionLibrary<MyProblem, MySolution>, IO>;

        template <typename MyProblem, typename MySolution, typename IO>
        const typename dsmt<MyProblem, MySolution, IO>::SubclassMap
            dsmt<MyProblem, MySolution, IO>::subclasses
            = dsmt<MyProblem, MySolution, IO>::GetSubclasses();

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<SingleSolutionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = SingleSolutionLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                SolutionMap<MySolution>* ctx
                    = static_cast<SolutionMap<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "SingleSolutionLibrary requires that context be set to "
                                  "a SolutionMap.");
                }

                int index;

                if(iot::outputting(io))
                    index = lib.solution->index;

                iot::mapRequired(io, "index", index);

                if(!iot::outputting(io))
                {
                    auto iter = ctx->find(index);
                    if(iter == ctx->end())
                    {
                        std::ostringstream msg;
                        msg << "Invalid solution index: " << index;
                        iot::setError(io, msg.str());
                    }
                    else
                    {
                        lib.solution = iter->second;
                    }
                }
            }

            const static bool flow = true;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<MasterSolutionLibrary<MyProblem, MySolution>, IO, EmptyContext>
        {
            using Library = MasterSolutionLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                iot::mapOptional(io, "version", lib.version);

                std::vector<std::shared_ptr<MySolution>> solutions;

                if(iot::outputting(io))
                {
                    solutions.reserve(lib.solutions.size());
                    for(auto const& pair : lib.solutions)
                        solutions.push_back(pair.second);
                }

                iot::mapRequired(io, "solutions", solutions);

                if(!iot::outputting(io))
                {
                    for(auto const& s : solutions)
                        lib.solutions[s->index] = s;
                }

                iot::setContext(io, &lib.solutions);

                std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> innerLibrary;

                if(iot::outputting(io))
                {
                    auto cache = std::dynamic_pointer_cast<CachingLibrary<MyProblem, MySolution>>(
                        lib.library);
                    if(cache)
                    {
                        innerLibrary = cache->library();
                    }
                    else
                    {
                        innerLibrary = lib.library;
                    }
                }

                iot::mapRequired(io, "library", innerLibrary);

                if(!iot::outputting(io))
                {
                    auto cache
                        = std::make_shared<CachingLibrary<MyProblem, MySolution>>(innerLibrary);

                    lib.library = cache;
                }
            }

            const static bool flow = false;
        };
    } // namespace Serialization
} // namespace Tensile
