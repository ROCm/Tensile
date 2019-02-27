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

#include <Tensile/Predicates.hpp>
#include <Tensile/GEMMProblemPredicates.hpp>
#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename IO>
        struct SubclassMappingTraits<Predicates::Predicate<GEMMProblem>, IO>:
            public DefaultSubclassMappingTraits<SubclassMappingTraits<Predicates::Predicate<GEMMProblem>, IO>,
                                                Predicates::Predicate<GEMMProblem>,
                                                IO>
        {
            using Self = SubclassMappingTraits<Predicates::Predicate<GEMMProblem>, IO>;
            using Base = DefaultSubclassMappingTraits<SubclassMappingTraits<Predicates::Predicate<GEMMProblem>, IO>,
                                                      Predicates::Predicate<GEMMProblem>,
                                                      IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            using Generic = PredicateMappingTraits<GEMMProblem, IO>;

            static SubclassMap GetSubclasses()
            {
                SubclassMap rv(
                {
                    Base::template Pair<Predicates::GEMM::ADimensionOrder>(),
                    Base::template Pair<Predicates::GEMM::BDimensionOrder>(),
                    Base::template Pair<Predicates::GEMM::CDimensionOrder>(),
                    Base::template Pair<Predicates::GEMM::DDimensionOrder>(),
                    Base::template Pair<Predicates::GEMM::IDivisible     >(),
                    Base::template Pair<Predicates::GEMM::JDivisible     >(),
                    Base::template Pair<Predicates::GEMM::KDivisible     >(),
                    Base::template Pair<Predicates::GEMM::LDivisible     >(),
                    Base::template Pair<Predicates::GEMM::CDStridesEqual >(),
                    Base::template Pair<Predicates::GEMM::LDCEqualsLDD   >(),
                    Base::template Pair<Predicates::GEMM::UseBeta        >()
                });

                auto gmap = Generic::GetSubclasses();
                rv.insert(gmap.begin(), gmap.end());

                return rv;
            }

        };

        template <typename IO>
        using GEMMProblemPredicateSMT = SubclassMappingTraits<Predicates::Predicate<GEMMProblem>, IO>;

        template <typename IO>
        const typename GEMMProblemPredicateSMT<IO>::SubclassMap
        GEMMProblemPredicateSMT<IO>::subclasses = GEMMProblemPredicateSMT<IO>::GetSubclasses();

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::ADimensionOrder, IO>:
        public AutoMappingTraits<Predicates::GEMM::ADimensionOrder, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::BDimensionOrder, IO>:
        public AutoMappingTraits<Predicates::GEMM::BDimensionOrder, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::CDimensionOrder, IO>:
        public AutoMappingTraits<Predicates::GEMM::CDimensionOrder, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::DDimensionOrder, IO>:
        public AutoMappingTraits<Predicates::GEMM::DDimensionOrder, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::IDivisible, IO>:
        public AutoMappingTraits<Predicates::GEMM::IDivisible, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::JDivisible, IO>:
        public AutoMappingTraits<Predicates::GEMM::JDivisible, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::KDivisible, IO>:
        public AutoMappingTraits<Predicates::GEMM::KDivisible, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::LDivisible, IO>:
        public AutoMappingTraits<Predicates::GEMM::LDivisible, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::CDStridesEqual, IO>:
        public AutoMappingTraits<Predicates::GEMM::CDStridesEqual, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::LDCEqualsLDD, IO>:
        public AutoMappingTraits<Predicates::GEMM::LDCEqualsLDD, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::GEMM::UseBeta, IO>:
        public AutoMappingTraits<Predicates::GEMM::UseBeta, IO> {};

    }
}

