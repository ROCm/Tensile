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

#include <Tensile/Serialization.hpp>
#include <Tensile/GEMMLibrary.hpp>
#include <Tensile/ContractionLibrary.hpp>

#include <llvm/ObjectYAML/YAML.h>

namespace Tensile
{
    namespace Serialization
    {
        template <>
        struct IOTraits<llvm::yaml::IO>
        {
            using IO = llvm::yaml::IO;

            template <typename T>
            static void mapRequired(IO & io, const char* key, T & obj)
            {
                io.mapRequired(key, obj);
            }

            template <typename T, typename Context>
            static void mapRequired(IO & io, const char* key, T & obj, Context & ctx)
            {
                io.mapRequired(key, obj, ctx);
            }

            static bool outputting(IO & io)
            {
                return io.outputting();
            }

            static void setError(IO & io, std::string const& msg)
            {
                throw std::runtime_error(msg);
                return io.setError(msg);
            }

            static void setContext(IO & io, void * ctx)
            {
                io.setContext(ctx);
            }

            static void * getContext(IO & io)
            {
                return io.getContext();
            }

            template <typename T>
            static void enumCase(IO & io, T & member, const char * key, T value)
            {
                io.enumCase(member, key, value);
            }
        };

        using GEMMHardwareRow =  typename Tensile::ExactLogicLibrary<Tensile::GEMMProblem,
                                                                     Tensile::GEMMSolution,
                                                                     Tensile::HardwarePredicate>::Row;
        using GEMMProblemRow =  typename Tensile::ExactLogicLibrary<Tensile::GEMMProblem,
                                                                    Tensile::GEMMSolution,
                                                                    Tensile::ProblemPredicate<GEMMProblem>>::Row;

        using ContractionHardwareRow =  typename Tensile::ExactLogicLibrary<Tensile::ContractionProblem,
                                                                            Tensile::ContractionSolution,
                                                                            Tensile::HardwarePredicate>::Row;
        using ContractionProblemRow =  typename Tensile::ExactLogicLibrary<Tensile::ContractionProblem,
                                                                           Tensile::ContractionSolution,
                                                                           Tensile::ProblemPredicate<ContractionProblem>>::Row;
    }
}

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::shared_ptr<Tensile::Predicates::Predicate<Tensile::GEMMProblem>>);
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::shared_ptr<Tensile::Predicates::Predicate<Tensile::ContractionProblem>>);
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::shared_ptr<Tensile::Predicates::Predicate<Tensile::Hardware>>);
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::shared_ptr<Tensile::Predicates::Predicate<Tensile::AMDGPU>>);
LLVM_YAML_IS_SEQUENCE_VECTOR(Tensile::Serialization::GEMMHardwareRow);
LLVM_YAML_IS_SEQUENCE_VECTOR(Tensile::Serialization::GEMMProblemRow);
LLVM_YAML_IS_SEQUENCE_VECTOR(std::shared_ptr<Tensile::GEMMSolution>);
LLVM_YAML_IS_SEQUENCE_VECTOR(Tensile::Serialization::ContractionHardwareRow);
LLVM_YAML_IS_SEQUENCE_VECTOR(Tensile::Serialization::ContractionProblemRow);
LLVM_YAML_IS_SEQUENCE_VECTOR(std::shared_ptr<Tensile::ContractionSolution>);

namespace llvm
{
    namespace yaml
    {

        namespace sn = Tensile::Serialization;

        LLVM_YAML_STRONG_TYPEDEF(size_t, FooType);

        using mysize_t = std::conditional<std::is_same<size_t, uint64_t>::value, FooType, size_t>::type;

        template<>
        struct ScalarTraits<mysize_t> {
          static void output(const mysize_t &value, void * ctx, raw_ostream & stream)
          {
              uint64_t tmp = value;
              ScalarTraits<uint64_t>::output(tmp, ctx, stream);
          }

          static StringRef input(StringRef str, void * ctx, mysize_t & value)
          {
              uint64_t tmp;
              auto rv = ScalarTraits<uint64_t>::input(str, ctx, tmp);
              value = tmp;
              return rv;
          }

          static bool mustQuote(StringRef) { return false; }
        };

        template <>
        struct MappingTraits<Tensile::GEMMSolution>
        {
            static void mapping(IO & io, Tensile::GEMMSolution & s)
            {
                sn::MappingTraits<Tensile::GEMMSolution, IO>::mapping(io, s);
            }
        };

        template <>
        struct MappingTraits<Tensile::ContractionSolution>
        {
            static void mapping(IO & io, Tensile::ContractionSolution & s)
            {
                sn::MappingTraits<Tensile::ContractionSolution, IO>::mapping(io, s);
            }
        };

        template <typename Data>
        struct SequenceTraits<Tensile::vector2<Data>>
        {
            using tt = sn::SequenceTraits<Tensile::vector2<Data>, IO>;
            static size_t size(IO & io, Tensile::vector2<Data> & v)                  { return tt::size(io, v); }
            static Data & element(IO & io, Tensile::vector2<Data> & v, size_t index) { return tt::element(io, v, index); }

            static const bool flow = true;
        };

        template <typename Data>
        struct SequenceTraits<Tensile::vector3<Data>>
        {
            using tt = sn::SequenceTraits<Tensile::vector3<Data>, IO>;
            static size_t size(IO & io, Tensile::vector3<Data> & v)                  { return tt::size(io, v); }
            static Data & element(IO & io, Tensile::vector3<Data> & v, size_t index) { return tt::element(io, v, index); }

            static const bool flow = true;
        };

        template <typename Data>
        struct SequenceTraits<Tensile::vector4<Data>>
        {
            using tt = sn::SequenceTraits<Tensile::vector4<Data>, IO>;
            static size_t size(IO & io, Tensile::vector4<Data> & v)                  { return tt::size(io, v); }
            static Data & element(IO & io, Tensile::vector4<Data> & v, size_t index) { return tt::element(io, v, index); }

            static const bool flow = true;
        };

        template <typename Object>
        struct ObjectMappingTraits
        {
            static void mapping(IO & io, Object & o)
            {
                sn::MappingTraits<Object, IO>::mapping(io, o);
            }
        };

        template <typename Object, typename Context>
        struct ObjectMappingContextTraits
        {
            static void mapping(IO & io, Object & o, Context & ctx)
            {
                sn::MappingTraits<Object, IO, Context>::mapping(io, o, ctx);
            }
        };

        template <typename Object>
        struct MappingTraits<std::shared_ptr<Tensile::Predicates::Predicate<Object>>>:
            public ObjectMappingTraits<std::shared_ptr<Tensile::Predicates::Predicate<Object>>>
        {
            static const bool flow = true;
        };

        template <typename Object, typename Value>
        struct MappingTraits<std::shared_ptr<Tensile::Property<Object, Value>>>:
            public ObjectMappingTraits<std::shared_ptr<Tensile::Property<Object, Value>>>
        {
            static const bool flow = true;
        };

        template <typename MyProblem, typename MySolution>
        struct MappingContextTraits<std::shared_ptr<Tensile::SolutionLibrary<MyProblem, MySolution>>,
                                    Tensile::SolutionMap<MySolution>>:
            public ObjectMappingContextTraits<std::shared_ptr<Tensile::SolutionLibrary<MyProblem, MySolution>>,
                                              Tensile::SolutionMap<MySolution>>
        {
        };

        template <typename MyProblem, typename MySolution>
        struct CustomMappingTraits<Tensile::LibraryMap<MyProblem, MySolution, std::string>>
        {
            using Value = Tensile::LibraryMap<MyProblem, MySolution, std::string>;
            using Impl = sn::CustomMappingTraits<Value, IO, Tensile::SolutionMap<MySolution>>;


            static void inputOne(IO &io, StringRef key, Value &elem)
            {
                Impl::inputOne(io, key, elem);
            }

            static void output(IO &io, Value &elem)
            {
                Impl::output(io, elem);
            }
        };

        template <>
        struct MappingTraits<std::shared_ptr<Tensile::GEMMSolution>>
        {
            using obj = Tensile::GEMMSolution;

            static void mapping(IO & io, std::shared_ptr<obj> & o)
            {
                sn::PointerMappingTraits<obj, IO>::mapping(io, o);
            }
        };

        template <>
        struct MappingTraits<std::shared_ptr<Tensile::MasterGEMMLibrary>>
        {
            using obj = Tensile::MasterGEMMLibrary;

            static void mapping(IO & io, std::shared_ptr<obj> & o)
            {
                sn::PointerMappingTraits<obj, IO>::mapping(io, o);
            }
        };

        template <>
        struct MappingTraits<Tensile::MasterGEMMLibrary>:
        public ObjectMappingTraits<Tensile::MasterGEMMLibrary>
        {
        };

        template <>
        struct MappingTraits<Tensile::Serialization::GEMMProblemRow>:
            public ObjectMappingTraits<Tensile::Serialization::GEMMProblemRow>
        {
        };

        template <>
        struct MappingTraits<Tensile::Serialization::GEMMHardwareRow>:
            public ObjectMappingTraits<Tensile::Serialization::GEMMHardwareRow>
        {
        };

        template <>
        struct MappingTraits<std::shared_ptr<Tensile::ContractionSolution>>
        {
            using obj = Tensile::ContractionSolution;

            static void mapping(IO & io, std::shared_ptr<obj> & o)
            {
                sn::PointerMappingTraits<obj, IO>::mapping(io, o);
            }
        };

        template <>
        struct MappingTraits<std::shared_ptr<Tensile::MasterContractionLibrary>>
        {
            using obj = Tensile::MasterContractionLibrary;

            static void mapping(IO & io, std::shared_ptr<obj> & o)
            {
                sn::PointerMappingTraits<obj, IO>::mapping(io, o);
            }
        };

        template <>
        struct MappingTraits<Tensile::MasterContractionLibrary>:
        public ObjectMappingTraits<Tensile::MasterContractionLibrary>
        {
        };

        template <>
        struct MappingTraits<Tensile::Serialization::ContractionProblemRow>:
            public ObjectMappingTraits<Tensile::Serialization::ContractionProblemRow>
        {
        };

        template <>
        struct MappingTraits<Tensile::Serialization::ContractionHardwareRow>:
            public ObjectMappingTraits<Tensile::Serialization::ContractionHardwareRow>
        {
        };

        template <>
        struct ScalarEnumerationTraits<Tensile::AMDGPU::Processor>
        {
            static void enumeration(IO & io, Tensile::AMDGPU::Processor & value)
            {
                sn::EnumTraits<Tensile::AMDGPU::Processor, IO>::enumeration(io, value);
            }
        };

        template <>
        struct ScalarEnumerationTraits<Tensile::DataType>
        {
            static void enumeration(IO & io, Tensile::DataType & value)
            {
                sn::EnumTraits<Tensile::DataType, IO>::enumeration(io, value);
            }
        };

    }
}

