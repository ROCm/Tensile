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

#include <Tensile/DataTypes.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>

namespace Tensile
{
    std::map<DataType, DataTypeInfo> DataTypeInfo::data;
    std::map<std::string, DataType>  DataTypeInfo::typeNames;

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
        case DataType::Int8x4:
            return "Int8x4";
        case DataType::Int32:
            return "Int32";
        case DataType::BFloat16:
            return "BFloat16";

        case DataType::Count:;
        }
        return "Invalid";
    }

    std::string TypeAbbrev(DataType d)
    {
        switch(d)
        {
        case DataType::Float:
            return "S";
        case DataType::Double:
            return "D";
        case DataType::ComplexFloat:
            return "C";
        case DataType::ComplexDouble:
            return "Z";
        case DataType::Half:
            return "H";
        case DataType::Int8x4:
            return "4xi8";
        case DataType::Int32:
            return "I";
        case DataType::BFloat16:
            return "B";

        case DataType::Count:;
        }
        return "Invalid";
    }

    template <typename T>
    void DataTypeInfo::registerTypeInfo()
    {
        using T_Info = TypeInfo<T>;

        DataTypeInfo info;

        info.dataType = T_Info::Enum;
        info.name     = T_Info::Name();
        info.abbrev   = T_Info::Abbrev();

        info.packing     = T_Info::Packing;
        info.elementSize = T_Info::ElementSize;
        info.segmentSize = T_Info::SegmentSize;

        info.isComplex  = T_Info::IsComplex;
        info.isIntegral = T_Info::IsIntegral;

        addInfoObject(info);
    }

    void DataTypeInfo::registerAllTypeInfo()
    {
        registerTypeInfo<float>();
        registerTypeInfo<double>();
        registerTypeInfo<std::complex<float>>();
        registerTypeInfo<std::complex<double>>();
        registerTypeInfo<Half>();
        registerTypeInfo<Int8x4>();
        registerTypeInfo<int32_t>();
        registerTypeInfo<BFloat16>();
    }

    void DataTypeInfo::registerAllTypeInfoOnce()
    {
        static int call_once = (registerAllTypeInfo(), 0);
    }

    void DataTypeInfo::addInfoObject(DataTypeInfo const& info)
    {
        data[info.dataType]  = info;
        typeNames[info.name] = info.dataType;
    }

    DataTypeInfo const& DataTypeInfo::Get(int index)
    {
        return Get(static_cast<DataType>(index));
    }

    DataTypeInfo const& DataTypeInfo::Get(DataType t)
    {
        registerAllTypeInfoOnce();

        auto iter = data.find(t);
        if(iter == data.end())
            throw std::runtime_error(concatenate("Invalid data type: ", static_cast<int>(t)));

        return iter->second;
    }

    DataTypeInfo const& DataTypeInfo::Get(std::string const& str)
    {
        registerAllTypeInfoOnce();

        auto iter = typeNames.find(str);
        if(iter == typeNames.end())
            throw std::runtime_error(concatenate("Invalid data type: ", str));

        return Get(iter->second);
    }

    std::ostream& operator<<(std::ostream& stream, const DataType& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, DataType& t)
    {
        std::string strValue;
        stream >> strValue;

#if 1
        t = DataTypeInfo::Get(strValue).dataType;

#else

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
        else if(strValue == ToString(DataType::Int8x4))
            t = DataType::Int8x4;
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
#endif

        return stream;
    }
} // namespace Tensile
