/**
 * Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <Tensile/KernelArguments.hpp>

#include <cstddef>
#include <iomanip>

namespace Tensile
{
    std::ostream& operator<<(std::ostream& stream, const KernelArguments& t)
    {
        size_t prevOffset = 0;
        for(auto const& name : t.m_names)
        {
            auto const& iter = t.m_argRecords.find(name);

            if(iter == t.m_argRecords.end())
                throw std::runtime_error("Argument " + name + " not found in record.");

            auto const& record = iter->second;

            size_t offset = std::get<KernelArguments::ArgOffset>(record);
            size_t size   = std::get<KernelArguments::ArgSize>(record);

            if(prevOffset != offset)
            {
                stream << "[" << prevOffset << ".." << offset - 1 << "] <padding>" << std::endl;
            }

            stream << "[" << offset << ".." << offset + size - 1 << "] " << name << ":";

            if(std::get<KernelArguments::ArgBound>(record))
            {
                auto oldFill  = stream.fill();
                auto oldWidth = stream.width();
                stream << std::hex;
                for(size_t i = offset; i < offset + size; i++)
                    stream << " " << std::setfill('0') << std::setw(2)
                           << static_cast<uint32_t>(t.m_data[i]);
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

    KernelArguments::~KernelArguments() {}

    void KernelArguments::reserve(size_t bytes, size_t count)
    {
        m_data.reserve(bytes);
        m_names.reserve(count);
        if(m_log)
            m_argRecords.reserve(count);
    }

    bool KernelArguments::isFullyBound() const
    {
        if(!m_log)
            return true;

        for(auto const& record : m_argRecords)
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

        return reinterpret_cast<void const*>(m_data.data());
    }

    size_t KernelArguments::size() const
    {
        return m_data.size();
    }
} // namespace Tensile
