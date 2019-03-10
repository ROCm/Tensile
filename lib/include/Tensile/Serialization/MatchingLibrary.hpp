
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

#include <Tensile/MatchingLibrary.hpp>
#include <Tensile/Distance.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename MyProblem, typename Element, typename Return, typename IO>
        struct MappingTraits<Matching::DistanceMatchingTable<MyProblem, Element, Return>, IO>
        {
            using Table = Matching::DistanceMatchingTable<MyProblem, Element, Return>;
            using iot = IOTraits<IO>;

            static void mapping(IO & io, Table & table)
            {
                iot::mapRequired(io, "properties", table.properties);
                iot::mapRequired(io, "table",      table.table);
                iot::mapRequired(io, "distance",   table.distance);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<ProblemMatchingLibrary<MyProblem, MySolution>, IO>
        {
            using Library = ProblemMatchingLibrary<MyProblem, MySolution>;
            using iot = IOTraits<IO>;

            static void mapping(IO & io, Library & lib)
            {
                // The inner table will be invisible in the YAML hierarchy.
                MappingTraits<typename Library::Table, IO>::mapping(io, lib.table);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<Matching::MatchingTableEntry<std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>>, IO>
        {
            using Entry = Matching::MatchingTableEntry<std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>>;
            using iot = IOTraits<IO>;

            static void mapping(IO & io, Entry & entry)
            {
                iot::mapRequired(io, "key",   entry.key);
                iot::mapRequired(io, "value", entry.value);
                iot::mapRequired(io, "speed", entry.speed);
            }

            const static bool flow = true;
        };

        template <typename IO>
        struct MappingTraits<std::shared_ptr<Matching::Distance>, IO>:
        public BaseClassMappingTraits<Matching::Distance, IO, true>
        {
        };

        template <typename IO>
        struct SubclassMappingTraits<Matching::Distance, IO>:
            public DefaultSubclassMappingTraits<SubclassMappingTraits<Matching::Distance, IO>, Matching::Distance, IO>
        {
            using Self = SubclassMappingTraits<Matching::Distance, IO>;
            using Base = DefaultSubclassMappingTraits<SubclassMappingTraits<Matching::Distance, IO>, Matching::Distance, IO>;

            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static SubclassMap GetSubclasses()
            {
                return SubclassMap(
                {
                    Base::template Pair<Matching::RatioDistance>(),
                    Base::template Pair<Matching::ManhattanDistance>(),
                    Base::template Pair<Matching::EuclideanDistance>(),
                    Base::template Pair<Matching::RandomDistance>()
                });
            }
        };

        template <typename IO>
        const typename SubclassMappingTraits<Matching::Distance, IO>::SubclassMap
            SubclassMappingTraits<Matching::Distance, IO>::subclasses =
                SubclassMappingTraits<Matching::Distance, IO>::GetSubclasses();

        template <typename IO> struct MappingTraits<Matching::RatioDistance,     IO>:
                           public AutoMappingTraits<Matching::RatioDistance,     IO> {};

        template <typename IO> struct MappingTraits<Matching::ManhattanDistance, IO>:
                           public AutoMappingTraits<Matching::ManhattanDistance, IO> {};

        template <typename IO> struct MappingTraits<Matching::EuclideanDistance, IO>:
                           public AutoMappingTraits<Matching::EuclideanDistance, IO> {};

        template <typename IO> struct MappingTraits<Matching::RandomDistance,    IO>:
                           public AutoMappingTraits<Matching::RandomDistance,    IO> {};
    }
}
