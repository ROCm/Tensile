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
#include <Tensile/Serialization/Predicates.hpp>

#include <Tensile/Predicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename IO>
        struct SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>:
            public DefaultSubclassMappingTraits<SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>,
                                                Predicates::Predicate<ContractionProblem>,
                                                IO>
        {
            using Self = SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>;
            using Base = DefaultSubclassMappingTraits<SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>,
                                                      Predicates::Predicate<ContractionProblem>,
                                                      IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            using Generic = PredicateMappingTraits<ContractionProblem, IO>;

            static SubclassMap GetSubclasses()
            {
                SubclassMap rv(
                {
                    Base::template Pair<Predicates::Contraction::FreeSizeAMultiple        >(),
                    Base::template Pair<Predicates::Contraction::FreeSizeBMultiple        >(),
                    Base::template Pair<Predicates::Contraction::BatchSizeMultiple        >(),
                    Base::template Pair<Predicates::Contraction::BoundSizeMultiple        >(),
                    Base::template Pair<Predicates::Contraction::MaxProblemSizeGreaterThan>(),
                    Base::template Pair<Predicates::Contraction::CDStridesEqual           >(),
                    Base::template Pair<Predicates::Contraction::LDCEqualsLDD             >(),
                    Base::template Pair<Predicates::Contraction::BetaZero                 >(),
                    Base::template Pair<Predicates::Contraction::BetaOne                  >(),
                    Base::template Pair<Predicates::Contraction::TypesEqual               >(),
                });

                auto gmap = Generic::GetSubclasses();
                rv.insert(gmap.begin(), gmap.end());

                return rv;
            }

        };

        template <typename IO>
        using ContractionProblemPredicateSMT = SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>;

        template <typename IO>
        const typename ContractionProblemPredicateSMT<IO>::SubclassMap
        ContractionProblemPredicateSMT<IO>::subclasses = ContractionProblemPredicateSMT<IO>::GetSubclasses();

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::FreeSizeAMultiple, IO>:
        public AutoMappingTraits<Predicates::Contraction::FreeSizeAMultiple, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::FreeSizeBMultiple, IO>:
        public AutoMappingTraits<Predicates::Contraction::FreeSizeBMultiple, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BatchSizeMultiple, IO>:
        public AutoMappingTraits<Predicates::Contraction::BatchSizeMultiple, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BoundSizeMultiple, IO>:
        public AutoMappingTraits<Predicates::Contraction::BoundSizeMultiple, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::MaxProblemSizeGreaterThan, IO>:
        public AutoMappingTraits<Predicates::Contraction::MaxProblemSizeGreaterThan, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::CDStridesEqual, IO>:
        public AutoMappingTraits<Predicates::Contraction::CDStridesEqual, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::LDCEqualsLDD, IO>:
        public AutoMappingTraits<Predicates::Contraction::LDCEqualsLDD, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BetaZero, IO>:
        public AutoMappingTraits<Predicates::Contraction::BetaZero, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BetaOne, IO>:
        public AutoMappingTraits<Predicates::Contraction::BetaOne, IO> {};

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::TypesEqual, IO>:
        public AutoMappingTraits<Predicates::Contraction::TypesEqual, IO> {};
    }
}


