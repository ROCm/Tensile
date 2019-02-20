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

#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace Tensile
{

    class KernelArguments
    {
    public:
        KernelArguments(bool log = true);
        virtual ~KernelArguments();

        template <typename T>
        void append(std::string const& name, T value);

        template <typename T>
        void appendUnbound(std::string const& name);

        template <typename T>
        void bind(std::string const& name, T value);

        bool isFullyBound() const;

        void const* data() const;
        size_t size() const;

        friend std::ostream& operator<<(std::ostream& stream, const KernelArguments& t);

    private:
        enum { ArgOffset, ArgSize, ArgBound, ArgString, NumArgFields };
        using Arg = std::tuple<size_t, size_t, bool, std::string>;
        static_assert(std::tuple_size<Arg>::value == NumArgFields, "Enum for fields of Arg tuple doesn't match size of tuple.");

        void alignTo(size_t alignment);

        template <typename T>
        void append(std::string const& name, T value, bool bound);

        template <typename T>
        std::string stringForValue(T value, bool bound);

        void appendRecord(std::string const& name, Arg info);

        template <typename T>
        void writeValue(size_t offset, T value);

        std::vector<uint8_t> m_data;

        std::vector<std::string> m_names;
        std::unordered_map<std::string, Arg> m_argRecords;

        bool m_log;
    };

    std::ostream& operator<<(std::ostream& stream, const KernelArguments& t);

    template <typename T>
    void KernelArguments::append(std::string const& name, T value)
    {
        append(name, value, true);
    }

    template <typename T>
    void KernelArguments::appendUnbound(std::string const& name)
    {
        append(name, static_cast<T>(0), false);
    }

    template <typename T>
    void KernelArguments::bind(std::string const& name, T value)
    {
        auto it = m_argRecords.find(name);
        if(it == m_argRecords.end())
        {
            throw std::runtime_error("Attempt to bind unknown argument " + name);
        }

        auto & record = it->second;

        if(std::get<ArgBound>(record))
        {
            throw std::runtime_error("Attempt to bind already bound argument " + name);
        }

        if(sizeof(T) != std::get<ArgSize>(record))
        {
            throw std::runtime_error("Size mismatch in binding argument " + name);
        }

        size_t offset = std::get<ArgOffset>(record);

        if(offset % alignof(T) != 0)
        {
            throw std::runtime_error("Alignment error in argument " + name + ": type mismatch?");
        }

        writeValue(offset, value);

        std::get<ArgString>(record) = stringForValue(value, true);
        std::get<ArgBound>(record) = true;
    }

    template <typename T>
    std::string KernelArguments::stringForValue(T value, bool bound)
    {
        if(!m_log)
            return "";

        if(!bound)
            return "<unbound>";


        std::ostringstream msg;
        msg << value;
        return std::move(msg.str());
    }

    template <typename T>
    void KernelArguments::append(std::string const& name, T value, bool bound)
    {
        std::string valueString = stringForValue(value, bound);

        alignTo(alignof(T));

        size_t offset = m_data.size();
        size_t size = sizeof(T);

        appendRecord(name, Arg(offset, size, bound, valueString));

        m_data.insert(m_data.end(), sizeof(value), 0);
        writeValue(offset, value);
    }

    template <typename T>
    void KernelArguments::writeValue(size_t offset, T value)
    {
        if(offset + sizeof(T) > m_data.size())
        {
            throw std::runtime_error("Value exceeds allocated bounds.");
        }

        std::memcpy(&m_data[offset], &value, sizeof(T));
    }
}

