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

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Serialization.hpp>

#include <llvm/ObjectYAML/YAML.h>

#include <cstddef>

namespace llvm
{
    namespace yaml
    {
        namespace sn = Tensile::Serialization;

        template <typename T>
        struct Hide
        {
            T& _value;

            Hide(T& value)
                : _value(value)
            {
            }

            T& operator*()
            {
                return _value;
            }
        };

        template <typename T>
        struct has_SequenceTraits<Hide<T>>
        {
            const static bool value = sn::has_SequenceTraits<T, IO>::value;
        };

        template <typename T>
        struct has_ScalarEnumerationTraits<Hide<T>>
        {
            const static bool value = sn::has_EnumTraits<T, IO>::value;
        };

        template <typename T>
        struct has_MappingTraits<Hide<T>, EmptyContext>
        {
            const static bool value = sn::has_EmptyMappingTraits<T, IO>::value;
        };

        template <typename T>
        struct has_CustomMappingTraits<Hide<T>>
        {
            const static bool value = sn::has_CustomMappingTraits<T, IO>::value;
        };

        template <typename T>
        struct missingTraits<T, EmptyContext>
            : public std::integral_constant<
                  bool,
                  !has_ScalarEnumerationTraits<T>::value && !has_ScalarBitSetTraits<T>::value
                      && !has_ScalarTraits<T>::value && !has_BlockScalarTraits<T>::value
                      && !has_MappingTraits<T, EmptyContext>::value && !has_SequenceTraits<T>::value
                      && !has_CustomMappingTraits<T>::value && !has_DocumentListTraits<T>::value
                      && !sn::has_SerializationTraits<T, IO>::value>
        {
        };

        template <typename T>
        typename std::enable_if<sn::has_SerializationTraits<T, IO>::value, void>::type
            yamlize(IO& io, T& Val, bool b, EmptyContext& ctx)
        {
            Hide<T> hide(Val);

            yamlize(io, hide, b, ctx);
        }

        template <typename T>
        typename std::enable_if<sn::has_SerializationTraits<T, IO>::value, Input&>::type
            operator>>(Input& input, T& Val)
        {
            Hide<T> hide(Val);

            return input >> hide;
        }

        template <typename T>
        typename std::enable_if<sn::has_SerializationTraits<T, IO>::value, Output&>::type
            operator<<(Output& output, T& Val)
        {
            Hide<T> hide(Val);

            return output << hide;
        }
    } // namespace yaml
} // namespace llvm

namespace Tensile
{
    namespace Serialization
    {
        template <>
        struct IOTraits<llvm::yaml::IO>
        {
            using IO = llvm::yaml::IO;

            template <typename T>
            static void mapRequired(IO& io, const char* key, T& obj)
            {
                io.mapRequired(key, obj);
            }

            template <typename T, typename Context>
            static void mapRequired(IO& io, const char* key, T& obj, Context& ctx)
            {
                io.mapRequired(key, obj, ctx);
            }

            template <typename T>
            static void mapOptional(IO& io, const char* key, T& obj)
            {
                io.mapOptional(key, obj);
            }

            template <typename T, typename Context>
            static void mapOptional(IO& io, const char* key, T& obj, Context& ctx)
            {
                io.mapOptional(key, obj, ctx);
            }

            static bool outputting(IO& io)
            {
                return io.outputting();
            }

            static void setError(IO& io, std::string const& msg)
            {
                // throw std::runtime_error(msg);
                return io.setError(msg);
            }

            static void setContext(IO& io, void* ctx)
            {
                io.setContext(ctx);
            }

            static void* getContext(IO& io)
            {
                return io.getContext();
            }

            template <typename T>
            static void enumCase(IO& io, T& member, const char* key, T value)
            {
                io.enumCase(member, key, value);
            }
        };

    } // namespace Serialization
} // namespace Tensile

namespace llvm
{
    namespace yaml
    {
        LLVM_YAML_STRONG_TYPEDEF(size_t, FooType);

        using mysize_t
            = std::conditional<std::is_same<size_t, uint64_t>::value, FooType, size_t>::type;

        template <>
        struct ScalarTraits<mysize_t>
        {
            static void output(const mysize_t& value, void* ctx, raw_ostream& stream)
            {
                uint64_t tmp = value;
                ScalarTraits<uint64_t>::output(tmp, ctx, stream);
            }

            static StringRef input(StringRef str, void* ctx, mysize_t& value)
            {
                uint64_t tmp;
                auto     rv = ScalarTraits<uint64_t>::input(str, ctx, tmp);
                value       = tmp;
                return rv;
            }

            static bool mustQuote(StringRef)
            {
                return false;
            }
        };

        template <typename T>
        struct MappingTraits<Hide<T>>
        {
            static void mapping(IO& io, Hide<T>& value)
            {
                sn::MappingTraits<T, IO>::mapping(io, *value);
            }

            static const bool flow = sn::MappingTraits<T, IO>::flow;
        };

        template <typename T>
        struct SequenceTraits<Hide<T>>
        {
            using Impl  = sn::SequenceTraits<T, IO>;
            using Value = typename Impl::Value;

            static size_t size(IO& io, Hide<T>& t)
            {
                return Impl::size(io, *t);
            }
            static Value& element(IO& io, Hide<T>& t, size_t index)
            {
                return Impl::element(io, *t, index);
            }

            static const bool flow = Impl::flow;
        };

        template <typename T>
        struct ScalarEnumerationTraits<Hide<T>>
        {
            static void enumeration(IO& io, Hide<T>& value)
            {
                sn::EnumTraits<T, IO>::enumeration(io, *value);
            }
        };

        template <typename T>
        struct CustomMappingTraits<Hide<T>>
        {
            using Impl = sn::CustomMappingTraits<T, IO>;

            static void inputOne(IO& io, StringRef key, Hide<T>& value)
            {
                Impl::inputOne(io, key, *value);
            }

            static void output(IO& io, Hide<T>& value)
            {
                Impl::output(io, *value);
            }
        };

        template <>
        struct MappingTraits<std::shared_ptr<Tensile::MasterContractionLibrary>>
        {
            using obj = Tensile::MasterContractionLibrary;

            static void mapping(IO& io, std::shared_ptr<obj>& o)
            {
                sn::PointerMappingTraits<obj, IO>::mapping(io, o);
            }
        };

        static_assert(sn::has_EmptyMappingTraits<
                          std::shared_ptr<Tensile::SolutionLibrary<Tensile::ContractionProblem>>,
                          IO>::value,
                      "asdf2");

        static_assert(sn::has_SerializationTraits<
                          std::shared_ptr<Tensile::SolutionLibrary<Tensile::ContractionProblem>>,
                          IO>::value,
                      "asdf");

        static_assert(
            !has_SequenceTraits<Hide<
                std::shared_ptr<Tensile::SolutionLibrary<Tensile::ContractionProblem>>>>::value,
            "fdsa");
        static_assert(
            has_MappingTraits<
                Hide<std::shared_ptr<Tensile::SolutionLibrary<Tensile::ContractionProblem>>>,
                EmptyContext>::value,
            "fdsa");

        static_assert(
            !missingTraits<
                Hide<std::shared_ptr<Tensile::SolutionLibrary<Tensile::ContractionProblem>>>,
                EmptyContext>::value,
            "fdsa");
    } // namespace yaml
} // namespace llvm
