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

#include <Tensile/KernelArguments.hpp>

#include <cstddef>
#include <iomanip>

namespace Tensile
{
    std::ostream& operator<<(std::ostream& stream, const KernelArguments& t)
    {
        size_t prevOffset = 0;
        for(auto const& name: t.m_names)
        {
            auto const& iter = t.m_argRecords.find(name);

            if(iter == t.m_argRecords.end())
                throw std::runtime_error("Argument " + name + " not found in record.");

            auto const& record = iter->second;

            size_t offset = std::get<KernelArguments::ArgOffset>(record);
            size_t size   = std::get<KernelArguments::ArgSize>(record);

            if(prevOffset != offset)
            {
                stream << "[" << prevOffset << ".." << offset-1 << "] <padding>" << std::endl;
            }

            stream << "["  << offset << ".." << offset + size-1 << "] " << name << ":";


            if(std::get<KernelArguments::ArgBound>(record))
            {
                auto oldFill = stream.fill();
                auto oldWidth = stream.width();
                stream << std::hex;
                for(size_t i = offset; i < offset + size; i++)
                    stream << " " << std::setfill('0') << std::setw(2) << static_cast<uint32_t>(t.m_data[i]);
                stream << std::dec;
                stream.fill(oldFill);
                stream.width(oldWidth);

                if(t.m_log)
                {
                    stream << " (" << std::get<KernelArguments::ArgString>(record) << ")";
                }
            }
            else
            {
                stream << " <unbound>";
            }

            stream << std::endl;

            prevOffset = offset + size;
        }

        return stream;
    }

    KernelArguments::KernelArguments(bool log)
        : m_log(log)
    {
    }

    KernelArguments::~KernelArguments()
    {
    }

    bool KernelArguments::isFullyBound() const
    {
        for(auto& record: m_argRecords)
        {
            if(!std::get<ArgBound>(record.second))
                return false;
        }

        return true;
    }

    void const* KernelArguments::data() const
    {
        if(!isFullyBound())
            throw std::runtime_error("Arguments not fully bound.");

        return reinterpret_cast<void const *>(m_data.data());
    }

    size_t KernelArguments::size() const
    {
        return m_data.size();
    }

    void KernelArguments::alignTo(size_t alignment)
    {
        size_t extraElements = m_data.size() % alignment;
        size_t padding = (alignment - extraElements) % alignment;

        m_data.insert(m_data.end(), padding, 0);
    }

    void KernelArguments::appendRecord(std::string const& name, KernelArguments::Arg record)
    {
        auto it = m_argRecords.find(name);
        if(it != m_argRecords.end())
        {
            throw "Duplicate argument name: " + name;
        }

        m_argRecords[name] = record;
        m_names.push_back(name);
    }
}

