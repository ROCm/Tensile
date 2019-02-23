/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <cassert>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace Tensile
{

    enum class DataType: int
    {
        Half,
        Float,
        Int32,
        Int8,
        Count
    };

    inline std::size_t TypeSize(DataType d)
    {
        switch(d)
        {
            case DataType::Int32:
            case DataType::Float: return 4;
            case DataType::Half: return 2;
            case DataType::Int8: return 1;

            case DataType::Count:
                throw std::runtime_error("Unknown data type");
        }
        throw std::runtime_error("Unknown data type");
    }

    std::string ToString(DataType d);
    std::ostream& operator<<(std::ostream& stream, const DataType& t);

    struct TensorDescriptor
    {
        TensorDescriptor();

        template <typename IterA,
                  typename IterB,
                  typename IterC>
        TensorDescriptor(DataType t,
                         IterA logicalCountsBegin,   IterA logicalCountsEnd,
                         IterB allocatedCountsBegin, IterB allocatedCountsEnd,
                         IterC dimensionOrderBegin,  IterC dimensionOrderEnd)
            : m_dataType(t),
              m_logicalCounts(logicalCountsBegin, logicalCountsEnd),
              m_allocatedCounts(allocatedCountsBegin, allocatedCountsEnd),
              m_dimensionOrder(dimensionOrderBegin, dimensionOrderEnd)
        {
            this->calculate();
        }

        template <typename IterA,
                  typename IterB>
        TensorDescriptor(DataType t,
                         IterA logicalCountsBegin, IterA logicalCountsEnd,
                         IterB allocatedCountsBegin, IterB allocatedCountsEnd)
            : m_dataType(t),
              m_logicalCounts(logicalCountsBegin, logicalCountsEnd),
              m_allocatedCounts(allocatedCountsBegin, allocatedCountsEnd)
        {
            this->calculate();
        }

        template <typename Iter>
        TensorDescriptor(DataType t,
                         Iter logicalCountsBegin, Iter logicalCountsEnd)
            : m_dataType(t),
              m_logicalCounts(logicalCountsBegin, logicalCountsEnd)
        {
            this->calculate();
        }

        template <typename T>
        TensorDescriptor(DataType t,
                         std::initializer_list<T> logicalCounts)
            : TensorDescriptor(t, logicalCounts.begin(), logicalCounts.end())
        { }

        template <typename T>
        TensorDescriptor(DataType t,
                         std::initializer_list<T> logicalCounts,
                         std::initializer_list<T> allocatedCounts)
            : TensorDescriptor(t,
                               logicalCounts.begin(), logicalCounts.end(),
                               allocatedCounts.begin(), allocatedCounts.end())
        { }

        template <typename T>
        TensorDescriptor(DataType t,
                         std::initializer_list<T> logicalCounts,
                         std::initializer_list<T> allocatedCounts,
                         std::initializer_list<T> dimensionOrder)
            : TensorDescriptor(t,
                               logicalCounts.begin(), logicalCounts.end(),
                               allocatedCounts.begin(), allocatedCounts.end(),
                               dimensionOrder.begin(), dimensionOrder.end())
        { }


        void calculate();

        const std::vector<std::size_t>& logicalCounts() const { return m_logicalCounts; }
        const std::vector<std::size_t>& allocatedCounts() const { return m_allocatedCounts; }
        const std::vector<std::size_t>& dimensionOrder() const { return m_dimensionOrder; }
        const std::vector<std::size_t>& strides() const { return m_strides; }

        size_t storedStride(size_t idx)         const { return         m_strides[m_dimensionOrder[idx]]; }
        size_t storedLogicalCount(size_t idx)   const { return   m_logicalCounts[m_dimensionOrder[idx]]; }
        size_t storedAllocatedCount(size_t idx) const { return m_allocatedCounts[m_dimensionOrder[idx]]; }

        void transpose(std::size_t dimA, std::size_t dimB);
        bool transposed() const;
        bool empty() const { return m_dimensionOrder.empty(); }

        void appendDim(size_t logicalCount);
        void appendDim(size_t logicalCount, size_t allocatedCount);

        std::size_t dimensions()             const { return m_dimensionOrder.size(); }
        std::size_t totalLogicalElements()   const { return m_totalLogicalElements; }
        std::size_t totalAllocatedElements() const { return m_totalAllocatedElements; }
        std::size_t totalAllocatedBytes()    const { return totalAllocatedElements() * elementBytes(); }

        std::size_t elementBytes() const { return TypeSize(m_dataType); }

        DataType dataType() const { return m_dataType; }

        template <typename Container>
        inline std::size_t index(Container const& indices) const
        {
            if(indices.size() != dimensions())
                throw std::runtime_error("Incorrect number of indices.");

            for(int i = 0; i < indices.size(); i++)
                if(indices[i] >= m_logicalCounts[i])
                    throw std::runtime_error("Index out of bounds.");

            return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), std::size_t(0));
        }

        template <typename T>
        inline std::size_t index(std::initializer_list<T> indices) const
        {
            if(indices.size() != dimensions())
                throw std::runtime_error("Incorrect number of indices.");

            for(auto i = std::make_pair(indices.begin(), m_logicalCounts.begin()); i.first != indices.end(); i.first++, i.second++)
                if(*i.first >= *i.second)
                    throw std::runtime_error("Index out of bounds.");

            return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), std::size_t(0));
        }


        template <class... Ts,
                    typename = typename std::enable_if
                    <
                        std::is_integral
                        <
                            typename std::common_type<Ts...>::type
                        >::value
                    >::type
                >
        inline std::size_t index(Ts... is) const
        {
            return this->index({is...});
        }

        bool operator==(const TensorDescriptor& rhs) const;
        bool operator!=(const TensorDescriptor& rhs) const;

        std::string ToString() const;

        friend std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

        private:
        std::vector<std::size_t> m_logicalCounts;
        std::vector<std::size_t> m_allocatedCounts;
        std::vector<std::size_t> m_dimensionOrder;

        std::vector<std::size_t> m_strides;

        std::size_t m_totalLogicalElements = 0;
        std::size_t m_totalAllocatedElements = 0;

        DataType m_dataType = DataType::Float;
    };

    std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

} // namespace

