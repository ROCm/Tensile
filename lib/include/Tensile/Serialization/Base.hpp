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

#include <algorithm>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Tensile/geom.hpp>
#include <Tensile/DataTypes.hpp>

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

        template <typename Map, typename IO, typename Context = EmptyContext>
        struct CustomMappingTraits
        {
            using iot = IOTraits<IO>;
            using key_type = typename Map::key_type;
            using mapped_type = typename Map::mapped_type;

            static void inputOne(IO & io, std::string const& key, Map & value)
            {
                Context * ctx = static_cast<Context *>(iot::getContext(io));
                iot::mapRequired(io, key.c_str(), value[key], *ctx);
            }

            static void output(IO & io, Map & value)
            {
                Context * ctx = static_cast<Context *>(iot::getContext(io));

                std::vector<key_type> keys;
                keys.reserve(value.size());
                for(auto const& pair: value)
                    keys.push_back(pair.first);
                std::sort(keys.begin(), keys.end());

                for(auto const& key: keys)
                    iot::mapRequired(io, key.c_str(), value.find(key)->second, *ctx);
            }
        };

        template <typename Map, typename IO>
        struct CustomMappingTraits<Map, IO, EmptyContext>
        {
            using iot = IOTraits<IO>;
            using key_type = typename Map::key_type;
            using mapped_type = typename Map::mapped_type;

            static void inputOne(IO & io, std::string const& key, Map & value)
            {
                iot::mapRequired(io, key.c_str(), value[key]);
            }

            static void output(IO & io, Map & value)
            {
                std::vector<key_type> keys;
                keys.reserve(value.size());
                for(auto const& pair: value)
                    keys.push_back(pair.first);
                std::sort(keys.begin(), keys.end());

                for(auto const& key: keys)
                    iot::mapRequired(io, key.c_str(), value.find(key)->second);
            }
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

        template <typename Object, typename IO>
        struct IndexMappingTraits
        {
            using iot = IOTraits<IO>;
            static_assert(Object::HasIndex == true, "Object doesn't have index/value.  Use the empty base class.");
            static void mapping(IO & io, Object & obj)
            {
                iot::mapRequired(io, "index", obj.index);
            }
        };

        template <typename Object, typename IO>
        struct IndexValueMappingTraits
        {
            using iot = IOTraits<IO>;
            static_assert(Object::HasIndex == true && Object::HasValue == true,
                          "Object doesn't have index/value.  Use the empty base class.");
            static void mapping(IO & io, Object & obj)
            {
                iot::mapRequired(io, "index", obj.index);
                iot::mapRequired(io, "value", obj.value);
            }
        };

        template <typename Object, typename IO, bool HasIndex = Object::HasIndex, bool HasValue = Object::HasValue>
        struct AutoMappingTraits
        {
        };

        template <typename Object, typename IO>
        struct AutoMappingTraits<Object, IO, false, false>:
            public EmptyMappingTraits<Object, IO>
        {
        };

        template <typename Object, typename IO>
        struct AutoMappingTraits<Object, IO, false,  true>:
            public ValueMappingTraits<Object, IO>
        {
        };

        template <typename Object, typename IO>
        struct AutoMappingTraits<Object, IO,  true, false>:
            public IndexMappingTraits<Object, IO>
        {
        };

        template <typename Object, typename IO>
        struct AutoMappingTraits<Object, IO,  true,  true>:
            public IndexValueMappingTraits<Object, IO>
        {
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
                return typename SubclassMap::value_type(Subclass::Type(), f);
            }

            static bool mapping(IO & io, std::string const& type, std::shared_ptr<T> & p, Context & ctx)
            {
                auto iter = CRTP_Traits::subclasses.find(type);
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
                return typename SubclassMap::value_type(Subclass::Type(), f);
            }

            static bool mapping(IO & io, std::string const& type, std::shared_ptr<T> & p)
            {
                auto iter = CRTP_Traits::subclasses.find(type);
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

        template <typename IO>
        struct EnumTraits<DataType, IO>
        {
            using iot = IOTraits<IO>;

            static void enumeration(IO & io, DataType & value)
            {
                iot::enumCase(io, value, "Half",  DataType::Half);
                iot::enumCase(io, value, "Float", DataType::Float);
                iot::enumCase(io, value, "Int32", DataType::Int32);
                iot::enumCase(io, value, "Int8",  DataType::Int8);
            }
        };
    }
}


