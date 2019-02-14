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

#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>

#include <Tensile/TensorDescriptor.hpp>

namespace Tensile {

    TensorDescriptor::TensorDescriptor()
    {
        this->calculate();
    }

    void TensorDescriptor::calculate()
    {
        if(m_allocatedCounts.empty())
            m_allocatedCounts = m_logicalCounts;

        if(m_logicalCounts.size() != m_allocatedCounts.size())
        {
            throw std::runtime_error("Size mismatch between logical and allocated counts.");
        }

        for(int i = 0; i < m_logicalCounts.size(); i++)
        {
            if(m_logicalCounts[i] <= 0)
                throw std::runtime_error("Each logical count must be > 0.");
            if(m_logicalCounts[i] > m_allocatedCounts[i])
                throw std::runtime_error("Each allocated count must be >= equivalent logical count.");
        }

        std::vector<std::size_t> modelDimensionOrder(m_logicalCounts.size(), 0);
        std::iota(modelDimensionOrder.begin(), modelDimensionOrder.end(), 0);

        if(m_dimensionOrder.empty())
        {
            m_dimensionOrder = std::move(modelDimensionOrder);
        }
        else
        {
            if(m_logicalCounts.size() != m_dimensionOrder.size())
                throw std::runtime_error("Dimension mismatch.");

            {
                std::vector<std::size_t> doCopy(m_dimensionOrder);
                std::sort(doCopy.begin(), doCopy.end());
                if(doCopy != modelDimensionOrder)
                    throw std::runtime_error("Each dimension must be < number of dimensions and each dimension must appear exactly once.");
            }
        }

        m_strides.resize(m_logicalCounts.size(), 0);

        if(m_logicalCounts.size() == 0)
        {
            m_totalLogicalElements = 0;
            m_totalAllocatedElements = 0;
            return;
        }

        auto multiply = [](size_t a, size_t b) { return a * b; };
        m_totalLogicalElements = std::accumulate(m_logicalCounts.begin(), m_logicalCounts.end(), 1, multiply);
        m_totalAllocatedElements = std::accumulate(m_allocatedCounts.begin(), m_allocatedCounts.end(), 1, multiply);

        m_strides[m_dimensionOrder[0]] = 1;

        for(std::size_t i = 1; i < m_strides.size(); i++)
        {
            m_strides[m_dimensionOrder[i]] = m_strides[m_dimensionOrder[i-1]]
                                           * m_allocatedCounts[m_dimensionOrder[i-1]];
        }
    }

    void TensorDescriptor::transpose(std::size_t dimA, std::size_t dimB)
    {
        if(dimA >= dimensions()) throw std::runtime_error("Invalid dimA.");
        if(dimB >= dimensions()) throw std::runtime_error("Invalid dimB.");

        std::swap(m_logicalCounts[dimA],   m_logicalCounts[dimB]);
        std::swap(m_allocatedCounts[dimA], m_allocatedCounts[dimB]);
        std::swap(m_dimensionOrder[dimA],  m_dimensionOrder[dimB]);
        std::swap(m_strides[dimA],         m_strides[dimB]);
    }

    bool TensorDescriptor::operator==(const TensorDescriptor& rhs) const
    {
        return m_logicalCounts   == rhs.m_logicalCounts
            && m_allocatedCounts == rhs.m_allocatedCounts
            && m_dimensionOrder  == rhs.m_dimensionOrder;
    }

    bool TensorDescriptor::operator!=(const TensorDescriptor& rhs) const { return !(*this == rhs); }


    std::string ToString(DataType d)
    {
        switch(d)
        {
            case DataType::Int32: return "Int32";
            case DataType::Float: return "Float";
            case DataType::Half: return  "Half";
            case DataType::Int8: return  "Int8";

            case DataType::Count:;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, const DataType& t)
    {
        return stream << ToString(t);
    }

    template <typename Container, typename Joiner>
    void streamJoin(std::ostream & stream, Container const& c, Joiner const& j)
    {
        bool first = true;
        for(auto const& item: c)
        {
            if(!first) stream << j;
            stream << item;
            first = false;
        }
    }

    std::string TensorDescriptor::ToString() const
    {
        std::ostringstream result;

        result << dimensions()
               << "-tensor<" << dataType() << ">"
               << "( logical(";
        streamJoin(result, m_logicalCounts, ", ");

        result << "), allocated(";
        streamJoin(result, m_allocatedCounts, ", ");

        result << "), dimensions(";
        streamJoin(result, m_dimensionOrder, ", ");
        result << ") )";

        return result.str();
    }

    std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t)
    {
        return stream << t.ToString();
    }

    //std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t)
    //{
    //    return LogRange(stream, t.lens, ", ");
    //}

}

#if 0
int main()
{
    Tensile::TensorDescriptor t(Tensile::DataType::Float, {2,3,4});

    std::cout << t << std::endl;
    std::cout << t.index(1,2,3) << std::endl;

    return 0;
}

#endif

