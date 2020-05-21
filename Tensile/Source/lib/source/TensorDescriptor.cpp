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

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <string>

#include <Tensile/Comparison.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    const size_t TensorDescriptor::UseDefaultStride = static_cast<size_t>(-1);

    TensorDescriptor::TensorDescriptor()
    {
        this->calculate();
    }

    void TensorDescriptor::calculate()
    {
        if(m_sizes.empty())
        {
            m_strides                = m_sizes;
            m_totalLogicalElements   = 0;
            m_totalAllocatedElements = 0;
            return;
        }

        for(int i = 0; i < m_sizes.size(); i++)
        {
            TENSILE_ASSERT_EXC(m_sizes[i] > 0);
        }

        m_strides.resize(m_sizes.size(), UseDefaultStride);
        if(m_strides[0] == UseDefaultStride)
        {
            m_strides[0] = 1;
        }
        m_totalLogicalElements = m_sizes[0];

        for(int i = 1; i < m_sizes.size(); i++)
        {
            m_totalLogicalElements *= m_sizes[i];

            if(m_strides[i] == UseDefaultStride)
            {
                m_strides[i] = m_strides[i - 1] * m_sizes[i - 1];
            }
        }

        m_totalAllocatedElements = 1;
        for(int i = 0; i < m_sizes.size(); i++)
            m_totalAllocatedElements += m_strides[i] * (m_sizes[i] - 1);

        if(Debug::Instance().printTensorInfo())
        {
            std::cout << "TensorDescriptor:calculate  " << *this
                      << "totalLogicalElements=" << m_totalLogicalElements
                      << " totalAllocatedElem=" << m_totalAllocatedElements << "\n";
        }
    }

    bool TensorDescriptor::operator==(const TensorDescriptor& rhs) const
    {
        return m_dataType == rhs.m_dataType && m_sizes == rhs.m_sizes && m_strides == rhs.m_strides;
    }

    bool TensorDescriptor::operator!=(const TensorDescriptor& rhs) const
    {
        return !(*this == rhs);
    }

    void TensorDescriptor::appendDim(size_t size)
    {
        appendDim(size, m_totalAllocatedElements);
    }

    void TensorDescriptor::appendDim(size_t size, size_t stride)
    {
        m_sizes.push_back(size);
        m_strides.push_back(stride);

        calculate();
    }

    int64_t TensorDescriptor::dimensionPadding(size_t dim) const
    {
        TENSILE_ASSERT_EXC(dim < dimensions());

        if(dim == 0)
            return m_strides[0] - 1;

        return m_strides[dim] - (m_strides[dim - 1] * m_sizes[dim - 1]);
    }

    void TensorDescriptor::collapseDims(size_t begin, size_t end)
    {
        TENSILE_ASSERT_EXC(end >= begin);
        TENSILE_ASSERT_EXC(begin < dimensions());
        TENSILE_ASSERT_EXC(end <= dimensions());

        if(end <= (begin + 1))
            return;

        for(size_t i = begin + 1; i < end; i++)
            TENSILE_ASSERT_EXC(dimensionPadding(i) == 0);

        size_t newDimensionSize = 1;
        for(size_t i = begin; i < end; i++)
            newDimensionSize *= m_sizes[i];

        m_sizes.erase(m_sizes.begin() + (begin + 1), m_sizes.begin() + end);
        m_sizes[begin] = newDimensionSize;

        m_strides.erase(m_strides.begin() + (begin + 1), m_strides.begin() + end);

        calculate();
    }

    std::string TensorDescriptor::ToString() const
    {
        std::ostringstream result;

        result << dimensions() << "-tensor<" << dataType() << ">"
               << "( sizes(";
        streamJoin(result, m_sizes, ", ");

        result << "), strides(";
        streamJoin(result, m_strides, ", ");

        result << ") )";

        return result.str();
    }

    std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t)
    {
        return stream << t.ToString();
    }

} // namespace Tensile
