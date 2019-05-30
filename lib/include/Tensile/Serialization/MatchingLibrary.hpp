
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

#include <Tensile/Distance.hpp>
#include <Tensile/MatchingLibrary.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename Key, typename MyProblem, typename Element, typename Return, typename IO>
        struct MappingTraits<Matching::DistanceMatchingTable<Key, MyProblem, Element, Return>, IO>
        {
            using Table = Matching::DistanceMatchingTable<Key, MyProblem, Element, Return>;
            using iot   = IOTraits<IO>;

            static void mapping(IO& io, Table& table)
            {
                iot::mapRequired(io, "table", table.table);
                iot::mapRequired(io, "distance", table.distance);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<ProblemMatchingLibrary<MyProblem, MySolution>, IO>
        {
            using Library    = ProblemMatchingLibrary<MyProblem, MySolution>;
            using Properties = typename Library::Table::Properties;
            using Element    = typename Library::Element;

            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                Properties properties;
                if(iot::outputting(io))
                {
                    properties = lib.table->properties;
                }

                iot::mapRequired(io, "properties", properties);

                bool success = false;
                if(properties.size() == 0)
                    iot::setError(io, "Matching table must have at least one property.");
                else if(properties.size() == 1)
                    success = mappingKey<std::array<size_t, 1>>(io, lib, properties);
                else if(properties.size() == 2)
                    success = mappingKey<std::array<size_t, 2>>(io, lib, properties);
                else if(properties.size() == 3)
                    success = mappingKey<std::array<size_t, 3>>(io, lib, properties);
                else if(properties.size() == 4)
                    success = mappingKey<std::array<size_t, 4>>(io, lib, properties);
                else if(properties.size() == 5)
                    success = mappingKey<std::array<size_t, 5>>(io, lib, properties);
                else if(properties.size() == 6)
                    success = mappingKey<std::array<size_t, 6>>(io, lib, properties);
                else if(properties.size() == 7)
                    success = mappingKey<std::array<size_t, 7>>(io, lib, properties);
                else if(properties.size() == 8)
                    success = mappingKey<std::array<size_t, 8>>(io, lib, properties);
                else if(properties.size() == 9)
                    success = mappingKey<std::array<size_t, 9>>(io, lib, properties);
                else if(properties.size() == 10)
                    success = mappingKey<std::array<size_t, 10>>(io, lib, properties);

                if(!success)
                    success = mappingKey<std::vector<size_t>>(io, lib, properties);

                if(!success)
                    iot::setError(io, "Can't write out key: wrong type.");
            }

            template <typename Key>
            static bool mappingKey(IO& io, Library& lib, Properties const& properties)
            {
                using Table = Matching::
                    DistanceMatchingTable<Key, MyProblem, Element, std::shared_ptr<MySolution>>;

                std::shared_ptr<Table> table;

                if(iot::outputting(io))
                {
                    table = std::dynamic_pointer_cast<Table>(lib.table);
                    if(!table)
                        return false;
                }
                else
                {
                    table             = std::make_shared<Table>();
                    table->properties = properties;
                    lib.table         = table;
                }

                MappingTraits<Table, IO>::mapping(io, *table);

                return true;
            }

            const static bool flow = false;
        };

        template <typename Key, typename Value, typename IO>
        struct MappingTraits<Matching::MatchingTableEntry<Key, Value>, IO>
        {
            using Entry = Matching::MatchingTableEntry<Key, Value>;
            using iot   = IOTraits<IO>;

            static void mapping(IO& io, Entry& entry)
            {
                iot::mapRequired(io, "key", entry.key);
                iot::mapRequired(io, "value", entry.value);
                iot::mapRequired(io, "speed", entry.speed);
            }

            const static bool flow = true;
        };

        template <typename Key, typename IO>
        struct MappingTraits<std::shared_ptr<Matching::Distance<Key>>, IO>
            : public BaseClassMappingTraits<Matching::Distance<Key>, IO, true>
        {
        };

        template <typename Key, typename IO>
        struct SubclassMappingTraits<Matching::Distance<Key>, IO>
            : public DefaultSubclassMappingTraits<
                  SubclassMappingTraits<Matching::Distance<Key>, IO>,
                  Matching::Distance<Key>,
                  IO>
        {
            using Self = SubclassMappingTraits<Matching::Distance<Key>, IO>;
            using Base
                = DefaultSubclassMappingTraits<SubclassMappingTraits<Matching::Distance<Key>, IO>,
                                               Matching::Distance<Key>,
                                               IO>;

            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static SubclassMap GetSubclasses()
            {
                return SubclassMap({Base::template Pair<Matching::RatioDistance<Key>>(),
                                    Base::template Pair<Matching::ManhattanDistance<Key>>(),
                                    Base::template Pair<Matching::EuclideanDistance<Key>>(),
                                    Base::template Pair<Matching::RandomDistance<Key>>()});
            }
        };

        template <typename Key, typename IO>
        const typename SubclassMappingTraits<Matching::Distance<Key>, IO>::SubclassMap
            SubclassMappingTraits<Matching::Distance<Key>, IO>::subclasses
            = SubclassMappingTraits<Matching::Distance<Key>, IO>::GetSubclasses();

        template <typename Key, typename IO>
        struct MappingTraits<Matching::RatioDistance<Key>, IO>
            : public AutoMappingTraits<Matching::RatioDistance<Key>, IO>
        {
        };

        template <typename Key, typename IO>
        struct MappingTraits<Matching::ManhattanDistance<Key>, IO>
            : public AutoMappingTraits<Matching::ManhattanDistance<Key>, IO>
        {
        };

        template <typename Key, typename IO>
        struct MappingTraits<Matching::EuclideanDistance<Key>, IO>
            : public AutoMappingTraits<Matching::EuclideanDistance<Key>, IO>
        {
        };

        template <typename Key, typename IO>
        struct MappingTraits<Matching::RandomDistance<Key>, IO>
            : public AutoMappingTraits<Matching::RandomDistance<Key>, IO>
        {
        };
    }
}
