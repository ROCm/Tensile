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

#include <memory>
#include <vector>

#include <Tensile/Macros.hpp>
#include <Tensile/Singleton.hpp>

namespace Tensile
{
    template <typename Object>
    struct EmbedData;

    /**
     * Encapsulates the storage of binary data stored in the executable.
     */
    template <typename Object>
    TENSILE_API
    class EmbeddedData: public LazySingleton<EmbeddedData<Object>>
    {
    public:
        using Base = LazySingleton<EmbeddedData<Object>>;


        using Item = std::vector<uint8_t>;
        using Items = std::vector<Item>;
        static Items const& Get() { return Base::Instance().items; }

    protected:
        friend Base;
        friend class EmbedData<Object>;

        static Items & GetMutable() { return Base::Instance().items; }
        EmbeddedData() = default;

        Items items;
    };

    template <typename Object>
    TENSILE_API
    struct EmbedData
    {
        EmbedData(std::initializer_list<uint8_t> data)
        {
            EmbeddedData<Object>::GetMutable().emplace_back(data);
        }

        EmbedData(std::vector<uint8_t> const& data)
        {
            EmbeddedData<Object>::Get().push_back(data);
        }
    };

}

