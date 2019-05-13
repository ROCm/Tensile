/**
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <string>

#include <Tensile/Utils.hpp>

namespace Tensile
{
    template <typename Class, bool HasIndex = Class::HasIndex, bool HasValue = Class::HasValue>
    struct PropertyHelper
    { };

    template <typename Class>
    struct PropertyHelper<Class, false, false>
    {
        static std::string ToString(Class const& obj)
        {
            return obj.type();
        }
    };

    template <typename Class>
    struct PropertyHelper<Class, false, true>
    {
        static std::string ToString(Class const& obj)
        {
            return concatenate(obj.type(), "(", obj.value, ")");
        }
    };

    template <typename Class>
    struct PropertyHelper<Class, true, false>
    {
        static std::string ToString(Class const& obj)
        {
            return concatenate(obj.type(), "(", obj.index, ")");
        }
    };

    template <typename Class>
    struct PropertyHelper<Class, true, true>
    {
        static std::string ToString(Class const& obj)
        {
            return concatenate(obj.type(), "(index=", obj.index, ", value=", obj.value, ")");
        }
    };

    template<typename Object, typename Value = size_t>
    class Property
    {
    public:
        virtual std::string type() const = 0;
        virtual ~Property() = default;
        virtual Value operator()(Object const& object) const = 0;

        virtual std::string toString() const = 0;

        virtual Value debugEval(Object const& object, std::ostream & stream) const
        {
            Value rv = (*this)(object);
            stream << *this << ": " << rv;
            return rv;
        }
    };

    template<typename Class, typename Object, typename Value = size_t>
    class Property_CRTP: public Property<Object, Value>
    {
    public:
        virtual std::string type() const final { return Class::Type(); }

        virtual std::string toString() const
        {
            Class const& this_typed = dynamic_cast<Class const&>(*this);
            return PropertyHelper<Class>::ToString(this_typed);
        }
    };

    template <typename Object, typename Value>
    inline std::ostream & operator<<(std::ostream & stream, Property<Object, Value> const& prop)
    {
        return stream << prop.toString();
    }

    template <typename Object, typename Value>
    inline std::ostream & operator<<(std::ostream & stream, std::vector<std::shared_ptr<Property<Object, Value>>> const& props)
    {
        stream << "(";

        bool first = true;
        for(auto const& v: props)
        {
            if(!first)
                stream << ", ";
            first = false;

            stream << *v;
        }
                
        stream << ")";

        return stream;
    }
}

