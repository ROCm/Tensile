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

#include <functional>
#include <memory>
#include <unordered_map>

#include <Tensile/geom.hpp>

namespace Tensile
{
    namespace Serialization
    {
        struct EmptyContext {};

        template <typename IO>
        struct IOTraits
        {
        };

        template <typename T, typename IO, typename Context = EmptyContext>
        struct MappingTraits
        {
        };

        template <typename Object, typename IO>
        struct EmptyMappingTraits
        {
            using iot = IOTraits<IO>;
            static_assert(Object::HasValue == false, "Object has a value.  Use the value base class.");
            static void mapping(IO & io, Object & obj)
            {
            }
        };

        template <typename Object, typename IO>
        struct ValueMappingTraits
        {
            using iot = IOTraits<IO>;
            static_assert(Object::HasValue == true, "Object has no value.  Use the empty base class.");
            static void mapping(IO & io, Object & obj)
            {
                iot::mapRequired(io, "value", obj.value);
            }
        };

        template <typename T, typename IO, typename Context = EmptyContext>
        struct SequenceTraits
        {
        };

        template <typename T, typename IO, typename Context = EmptyContext>
        struct SubclassMappingTraits
        {
        };

        template <typename T, typename IO>
        struct EnumTraits
        {
        };

        template <typename Subclass, typename IO, typename Context = EmptyContext>
        struct PointerMappingTraits;

        template <typename Subclass, typename IO, typename Context>
        struct PointerMappingTraits
        {
            using iot = IOTraits<IO>;
            template <typename Base>
            static bool mapping(IO & io, std::shared_ptr<Base> & p, Context & ctx)
            {
                std::shared_ptr<Subclass> sc;

                if(iot::outputting(io))
                {
                    sc = std::dynamic_pointer_cast<Subclass>(p);
                }
                else
                {
                    sc = std::make_shared<Subclass>();
                    p = sc;
                }

                MappingTraits<Subclass, IO, Context>::mapping(io, *sc, ctx);

                return true;
            }
        };

        template <typename Subclass, typename IO>
        struct PointerMappingTraits<Subclass, IO, EmptyContext>
        {
            using iot = IOTraits<IO>;
            template <typename Base>
            static bool mapping(IO & io, std::shared_ptr<Base> & p)
            {
                std::shared_ptr<Subclass> sc;

                if(iot::outputting(io))
                {
                    sc = std::dynamic_pointer_cast<Subclass>(p);
                }
                else
                {
                    sc = std::make_shared<Subclass>();
                    p = sc;
                }

                MappingTraits<Subclass, IO>::mapping(io, *sc);

                return true;
            }
        };

        template <typename CRTP_Traits, typename T, typename IO, typename Context = EmptyContext>
        struct DefaultSubclassMappingTraits;

        template <typename CRTP_Traits, typename T, typename IO, typename Context>
        struct DefaultSubclassMappingTraits
        {
            using iot = IOTraits<IO>;
            using SubclassFn = bool(IO &, typename std::shared_ptr<T> &, Context &);
            using SubclassMap = std::unordered_map<std::string, std::function<SubclassFn>>;

            template <typename Subclass>
            static typename SubclassMap::value_type Pair()
            {
                auto f = PointerMappingTraits<Subclass, IO, Context>::template mapping<T>;
                return typename SubclassMap::value_type(Subclass::Key(), f);
            }

            static bool mapping(IO & io, std::string const& key, std::shared_ptr<T> & p, Context & ctx)
            {
                auto iter = CRTP_Traits::subclasses.find(key);
                if(iter != CRTP_Traits::subclasses.end())
                    return iter->second(io, p, ctx);
                return false;
            }
        };

        template <typename CRTP_Traits, typename T, typename IO>
        struct DefaultSubclassMappingTraits<CRTP_Traits, T, IO, EmptyContext>
        {
            using iot = IOTraits<IO>;
            using SubclassFn = bool(IO &, typename std::shared_ptr<T> &);
            using SubclassMap = std::unordered_map<std::string, std::function<SubclassFn>>;

            template <typename Subclass>
            static typename SubclassMap::value_type Pair()
            {
                auto f = PointerMappingTraits<Subclass, IO>::template mapping<T>;
                return typename SubclassMap::value_type(Subclass::Key(), f);
            }

            static bool mapping(IO & io, std::string const& key, std::shared_ptr<T> & p)
            {
                auto iter = CRTP_Traits::subclasses.find(key);
                if(iter != CRTP_Traits::subclasses.end())
                    return iter->second(io, p);
                return false;
            }
        };

        template <typename Data, typename IO>
        struct SequenceTraits<vector2<Data>, IO>
        {
            static size_t size(IO & io, vector2<Data> & v) { return 2; }
            static Data & element(IO & io, vector2<Data> & v, size_t index)
            {
                if(index == 0) return v.x;
                return v.y;
            }
        };

        template <typename Data, typename IO>
        struct SequenceTraits<vector3<Data>, IO>
        {
            static size_t size(IO & io, vector3<Data> & v) { return 3; }
            static Data & element(IO & io, vector3<Data> & v, size_t index)
            {
                if(index == 0) return v.x;
                if(index == 1) return v.y;
                return v.z;
            }
        };

        template <typename Data, typename IO>
        struct SequenceTraits<vector4<Data>, IO>
        {
            static size_t size(IO & io, vector4<Data> & v) { return 4; }
            static Data & element(IO & io, vector4<Data> & v, size_t index)
            {
                if(index == 0) return v.x;
                if(index == 1) return v.y;
                if(index == 2) return v.z;
                return v.w;
            }
        };

    }
}


