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

#include <cassert>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <Tensile/DataTypes.hpp>
#include <Tensile/Macros.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    template <typename SizeIter>
    inline size_t CoordCount(SizeIter sizeBegin, SizeIter sizeEnd)
    {
        size_t rv = 1;

        while(sizeBegin != sizeEnd)
        {
            rv *= *sizeBegin;
            sizeBegin++;
        }

        return rv;
    }

    template <typename CoordIter, typename SizeIter>
    inline void CoordNumbered(size_t num,
                              CoordIter coordBegin, CoordIter coordEnd,
                              SizeIter sizeBegin, SizeIter sizeEnd)
    {
        auto coord = coordBegin;
        auto size = sizeBegin;

        while(coord != coordEnd && size != sizeEnd)
        {
            *coord = num % *size;
            num /= *size;

            coord++;
            size++;
        }

        if(coord != coordEnd || size != sizeEnd)
            throw std::runtime_error("Inconsistent size of coordinates.");
    }

    template <typename CoordIter, typename SizeIter>
    inline bool IncrementCoord(CoordIter coordBegin, CoordIter coordEnd,
                               SizeIter sizeBegin, SizeIter sizeEnd)
    {
        auto coord = coordBegin;
        auto size = sizeBegin;

        while(coord != coordEnd)
        {
            (*coord)++;
            if(*coord < *size)
                return true;

            *coord = 0;

            coord++;
            size++;
        }

        return false;
    }

    class TENSILE_API TensorDescriptor
    {
    public:

        TensorDescriptor();

        template <typename IterA,
                  typename IterB>
        TensorDescriptor(DataType t,
                         IterA sizesBegin,   IterA sizesEnd,
                         IterB stridesBegin, IterB stridesEnd)
            : m_sizes(sizesBegin, sizesEnd),
              m_strides(stridesBegin, stridesEnd),
              m_dataType(t)
        {
            this->calculate();
        }

        template <typename Iter>
        TensorDescriptor(DataType t,
                         Iter sizesBegin, Iter sizesEnd)
            : m_sizes(sizesBegin, sizesEnd),
              m_dataType(t)
        {
            this->calculate();
        }

        TensorDescriptor(DataType t,
                         std::initializer_list<size_t> sizes)
            : m_sizes(sizes),
              m_dataType(t)
        
        {
            this->calculate();
        }

        TensorDescriptor(DataType t,
                         std::initializer_list<size_t> sizes,
                         std::initializer_list<size_t> strides)
            : m_sizes(sizes),
              m_strides(strides),
              m_dataType(t)
        {
            this->calculate();
        }

        void calculate();

        const std::vector<size_t>& sizes() const { return m_sizes; }
        const std::vector<size_t>& strides() const { return m_strides; }

        bool empty() const { return m_sizes.empty(); }

        void appendDim(size_t logicalCount);
        void appendDim(size_t logicalCount, size_t allocatedCount);

        /**
         * Returns the number of elements of padding in the given dimension (0 if unpadded).
         */
        size_t dimensionPadding(size_t dim) const;

        /**
         * Collapses dimensions in the interval [begin, end).
         * 
         * preconditions:
         * - end >= begin
         * - begin < dimensions()
         * - end <= dimensions()
         * - dimensions in the interval [begin, end-1) are not padded.
         *
         * postconditions:
         * - dimensions() is diminished by end-begin
         * - total elements (allocated and logical) remain the same
         * - dimension 'begin' is the product of all the dimensions in the interval [begin, end).
         */
        void collapseDims(size_t begin, size_t end);

        size_t dimensions()             const { return m_sizes.size(); }
        size_t totalLogicalElements()   const { return m_totalLogicalElements; }
        size_t totalAllocatedElements() const { return m_totalAllocatedElements; }
        size_t totalAllocatedBytes()    const { return totalAllocatedElements() * elementBytes(); }

        size_t elementBytes() const { return DataTypeInfo::Get(m_dataType).elementSize; }

        DataType dataType() const { return m_dataType; }

        template <typename Container>
        inline size_t index(Container const& indices) const
        {
            if(indices.size() != dimensions())
                throw std::runtime_error("Incorrect number of indices.");

            for(int i = 0; i < indices.size(); i++)
                if(indices[i] >= m_sizes[i])
                    throw std::runtime_error("Index out of bounds.");

            return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), size_t(0));
        }

        template <typename T>
        inline size_t index(std::initializer_list<T> indices) const
        {
            if(indices.size() != dimensions())
                throw std::runtime_error("Incorrect number of indices.");

            for(auto i = std::make_pair(indices.begin(), m_sizes.begin()); i.first != indices.end(); i.first++, i.second++)
                if(*i.first >= *i.second)
                    throw std::runtime_error("Index out of bounds.");

            return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), size_t(0));
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
        inline size_t index(Ts... is) const
        {
            return this->index({is...});
        }

        inline bool incrementCoord(std::vector<size_t> & coord, size_t firstDimension = 0) const
        {
            if(coord.size() != dimensions())
                throw std::runtime_error(concatenate("Invalid coordinate size ", coord.size(), " for ", dimensions(), "-tensor"));

            if(firstDimension >= dimensions())
                return false;

            return IncrementCoord(coord.begin() + firstDimension, coord.end(),
                                  m_sizes.begin(), m_sizes.end());
        }

        bool operator==(const TensorDescriptor& rhs) const;
        bool operator!=(const TensorDescriptor& rhs) const;

        std::string ToString() const;

        friend std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

    private:
        std::vector<size_t> m_sizes;
        std::vector<size_t> m_strides;

        size_t m_totalLogicalElements = 0;
        size_t m_totalAllocatedElements = 0;

        DataType m_dataType = DataType::Float;
    };

    std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

    template <typename T>
    void WriteTensor(std::ostream & stream, T * data, TensorDescriptor const& desc)
    {
        if(desc.dimensions() != 3)
            throw std::runtime_error("Fix this function to work with dimensions != 3");

        std::vector<size_t> index3{0,0,0};

        stream << "Tensor("
            << desc.sizes()[0] << ", "
            << desc.sizes()[1] << ", "
            << desc.sizes()[2] << ")";

       stream << std::endl;

        for(index3[2] = 0; index3[2] < desc.sizes()[2]; index3[2]++)
        {
            stream << "[" << std::endl;
            for(index3[0] = 0; index3[0] < desc.sizes()[0]; index3[0]++)
            {
                for(index3[1] = 0; index3[1] < desc.sizes()[1]; index3[1]++)
                {
                    size_t idx = desc.index(index3);
                    stream << data[idx] << " ";
                }
                stream << std::endl;
            }
            stream << "]" << std::endl;
        }
    }

} // namespace

