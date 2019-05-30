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

#include <Tensile/DataTypes.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>

namespace Tensile
{
    std::string ToString(DataType d)
    {
        switch(d)
        {
        case DataType::Float:
            return "Float";
        case DataType::Double:
            return "Double";
        case DataType::ComplexFloat:
            return "ComplexFloat";
        case DataType::ComplexDouble:
            return "ComplexDouble";
        case DataType::Half:
            return "Half";
        case DataType::Int8:
            return "Int8";
        case DataType::Int32:
            return "Int32";

        case DataType::Count:;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, const DataType& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, DataType& t)
    {
        std::string strValue;
        stream >> strValue;

        if(strValue == ToString(DataType::Float))
            t = DataType::Float;
        else if(strValue == ToString(DataType::Double))
            t = DataType::Double;
        else if(strValue == ToString(DataType::ComplexFloat))
            t = DataType::ComplexFloat;
        else if(strValue == ToString(DataType::ComplexDouble))
            t = DataType::ComplexDouble;
        else if(strValue == ToString(DataType::Half))
            t = DataType::Half;
        else if(strValue == ToString(DataType::Int8))
            t = DataType::Int8;
        else if(strValue == ToString(DataType::Int32))
            t = DataType::Int32;
        else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
        {
            int value = atoi(strValue.c_str());
            if(value >= 0 && value < static_cast<int>(DataType::Count))
                t = static_cast<DataType>(value);
            else
                throw std::runtime_error(concatenate("Can't convert ", strValue, " to DataType."));
        }
        else
        {
            throw std::runtime_error(concatenate("Can't convert ", strValue, " to DataType."));
        }

        return stream;
    }
}
