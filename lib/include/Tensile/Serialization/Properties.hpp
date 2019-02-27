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

#include <Tensile/Serialization/Base.hpp>

#include <Tensile/PropertyMatching.hpp>
#include <Tensile/ContractionProblemProperties.hpp>

namespace Tensile
{
    namespace Serialization
    {

        template <typename Object, typename Value, typename IO>
        struct MappingTraits<std::shared_ptr<Property<Object, Value>>, IO>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO & io, std::shared_ptr<Property<Object, Value>> & p)
            {
                std::string type;

                if(iot::outputting(io))
                    type = p->type();

                iot::mapRequired(io, "type", type);

                if(!SubclassMappingTraits<Property<Object, Value>, IO>::mapping(io, type, p))
                    iot::setError(io, "Unknown predicate type " + type);
            }
        };

        template <typename IO>
        struct SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>:
            public DefaultSubclassMappingTraits<SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>,
                                                Property<ContractionProblem, size_t>,
                                                IO>
        {
            using Self = SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>;
            using Base = DefaultSubclassMappingTraits<SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>,
                                                      Property<ContractionProblem, size_t>,
                                                      IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static SubclassMap GetSubclasses()
            {
                return SubclassMap(
                {
                    Base::template Pair<Contraction::FreeSizeA>(),
                    Base::template Pair<Contraction::FreeSizeB>(),
                    Base::template Pair<Contraction::BatchSize>(),
                    Base::template Pair<Contraction::BoundSize>(),
                    Base::template Pair<Contraction::AStride>(),
                    Base::template Pair<Contraction::BStride>(),
                    Base::template Pair<Contraction::CStride>(),
                    Base::template Pair<Contraction::DStride>()
                });
            }
        };

        template <typename IO>
        struct SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>:
            public DefaultSubclassMappingTraits<SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>,
                                                Property<ContractionProblem, std::string>,
                                                IO>
        {
            using Self = SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>;
            using Base = DefaultSubclassMappingTraits<SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>,
                                                      Property<ContractionProblem, std::string>,
                                                      IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static SubclassMap GetSubclasses()
            {
                return SubclassMap(
                {
                    Base::template Pair<Contraction::OperationIdentifier>()
                });
            }
        };

        template <typename Object, typename Value, typename IO>
        using PropertySMT = SubclassMappingTraits<Property<Object, Value>, IO>;

        template <typename IO>
        const typename PropertySMT<ContractionProblem, size_t, IO>::SubclassMap
            PropertySMT<ContractionProblem, size_t, IO>::subclasses =
                PropertySMT<ContractionProblem, size_t, IO>::GetSubclasses();

        template <typename IO>
        const typename PropertySMT<ContractionProblem, std::string, IO>::SubclassMap
            PropertySMT<ContractionProblem, std::string, IO>::subclasses =
                PropertySMT<ContractionProblem, std::string, IO>::GetSubclasses();

        template <typename IO> struct MappingTraits<Contraction::FreeSizeA,           IO>:
                           public AutoMappingTraits<Contraction::FreeSizeA,           IO> {};

        template <typename IO> struct MappingTraits<Contraction::FreeSizeB,           IO>:
                           public AutoMappingTraits<Contraction::FreeSizeB,           IO> {};

        template <typename IO> struct MappingTraits<Contraction::BatchSize,           IO>:
                           public AutoMappingTraits<Contraction::BatchSize,           IO> {};

        template <typename IO> struct MappingTraits<Contraction::BoundSize,           IO>:
                           public AutoMappingTraits<Contraction::BoundSize,           IO> {};

        template <typename IO> struct MappingTraits<Contraction::AStride,             IO>:
                           public AutoMappingTraits<Contraction::AStride,             IO> {};

        template <typename IO> struct MappingTraits<Contraction::BStride,             IO>:
                           public AutoMappingTraits<Contraction::BStride,             IO> {};

        template <typename IO> struct MappingTraits<Contraction::CStride,             IO>:
                           public AutoMappingTraits<Contraction::CStride,             IO> {};

        template <typename IO> struct MappingTraits<Contraction::DStride,             IO>:
                           public AutoMappingTraits<Contraction::DStride,             IO> {};

        template <typename IO> struct MappingTraits<Contraction::OperationIdentifier, IO>:
                           public AutoMappingTraits<Contraction::OperationIdentifier, IO> {};

        template <typename IO>
        struct SubclassMappingTraits<Property<GEMMProblem, size_t>, IO>:
            public DefaultSubclassMappingTraits<SubclassMappingTraits<Property<GEMMProblem, size_t>, IO>,
                                                Property<GEMMProblem, size_t>,
                                                IO>
        {
            using Self = SubclassMappingTraits<Property<GEMMProblem, size_t>, IO>;
            using Base = DefaultSubclassMappingTraits<SubclassMappingTraits<Property<GEMMProblem, size_t>, IO>,
                                                      Property<GEMMProblem, size_t>,
                                                      IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static SubclassMap GetSubclasses()
            {
                return SubclassMap(
                {
                });
            }
        };

        template <typename IO>
        struct SubclassMappingTraits<Property<GEMMProblem, std::string>, IO>:
            public DefaultSubclassMappingTraits<SubclassMappingTraits<Property<GEMMProblem, std::string>, IO>,
                                                Property<GEMMProblem, std::string>,
                                                IO>
        {
            using Self = SubclassMappingTraits<Property<GEMMProblem, std::string>, IO>;
            using Base = DefaultSubclassMappingTraits<SubclassMappingTraits<Property<GEMMProblem, std::string>, IO>,
                                                      Property<GEMMProblem, std::string>,
                                                      IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static SubclassMap GetSubclasses()
            {
                return SubclassMap(
                {
                });
            }
        };

        template <typename IO>
        const typename PropertySMT<GEMMProblem, size_t, IO>::SubclassMap
            PropertySMT<GEMMProblem, size_t, IO>::subclasses =
                PropertySMT<GEMMProblem, size_t, IO>::GetSubclasses();

        template <typename IO>
        const typename PropertySMT<GEMMProblem, std::string, IO>::SubclassMap
            PropertySMT<GEMMProblem, std::string, IO>::subclasses =
                PropertySMT<GEMMProblem, std::string, IO>::GetSubclasses();
    }
}

