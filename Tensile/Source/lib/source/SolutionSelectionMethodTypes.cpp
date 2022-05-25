/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
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

#include <Tensile/SolutionSelectionMethodTypes.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>

namespace Tensile
{
    std::map<SolutionSelectionMethod, SolutionSelectionMethodTypeInfo> SolutionSelectionMethodTypeInfo::data;
    std::map<std::string, SolutionSelectionMethod>                     SolutionSelectionMethodTypeInfo::typeNames;

    std::string ToString(SolutionSelectionMethod ssm)
    {
        switch(ssm)
        {
        case SolutionSelectionMethod::Auto:
            return "Auto";
        case SolutionSelectionMethod::CUEfficiency:
            return "CUEfficiency";
        case SolutionSelectionMethod::DeviceEfficiency:
            return "DeviceEfficiency";
        case SolutionSelectionMethod::Experimental:
            return "Experimental";

        case SolutionSelectionMethod::Count:
        default:
            return "Invalid";
        }
    }

    std::string TypeAbbrev(SolutionSelectionMethod ssm)
    {
        switch(ssm)
        {
        case SolutionSelectionMethod::Auto:
            return "Auto";
        case SolutionSelectionMethod::CUEfficiency:
            return "CUEff";
        case SolutionSelectionMethod::DeviceEfficiency:
            return "DvEff";
        case SolutionSelectionMethod::Experimental:
            return "Test";
        case SolutionSelectionMethod::Count:
        default:
            return "Invalid";
        }
    }

    template <SolutionSelectionMethod T>
    void SolutionSelectionMethodTypeInfo::registerTypeInfo()
    {
        using T_Info = SolutionSelectionMethodInfo<T>;

        SolutionSelectionMethodTypeInfo info;

        info.m_solutionSelectionMethod = T_Info::Enum;
        info.name   = T_Info::Name();
        info.abbrev = T_Info::Abbrev();

        addInfoObject(info);
    }

    void SolutionSelectionMethodTypeInfo::registerAllTypeInfo()
    {
        registerTypeInfo<SolutionSelectionMethod::Auto>();
        registerTypeInfo<SolutionSelectionMethod::CUEfficiency>();
        registerTypeInfo<SolutionSelectionMethod::DeviceEfficiency>();
        registerTypeInfo<SolutionSelectionMethod::Experimental>();
    }

    void SolutionSelectionMethodTypeInfo::registerAllTypeInfoOnce()
    {
        static int call_once = (registerAllTypeInfo(), 0);

        // Use the variable to quiet the compiler.
        if(call_once)
            return;
    }

    void SolutionSelectionMethodTypeInfo::addInfoObject(SolutionSelectionMethodTypeInfo const& info)
    {
        auto toLower = [](std::string tmp) {
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
            return tmp;
        };

        data[info.m_solutionSelectionMethod] = info;

        // Add some flexibility to names registry. Accept abbreviations and
        // lower case versions of the strings
        typeNames[info.name]            = info.m_solutionSelectionMethod;
        typeNames[toLower(info.name)]   = info.m_solutionSelectionMethod;
        typeNames[info.abbrev]          = info.m_solutionSelectionMethod;
        typeNames[toLower(info.abbrev)] = info.m_solutionSelectionMethod;
    }

    SolutionSelectionMethodTypeInfo const& SolutionSelectionMethodTypeInfo::Get(int index)
    {
        return Get(static_cast<SolutionSelectionMethod>(index));
    }

    SolutionSelectionMethodTypeInfo const& SolutionSelectionMethodTypeInfo::Get(SolutionSelectionMethod ssm)
    {
        registerAllTypeInfoOnce();

        auto iter = data.find(ssm);
        if(iter == data.end())
            throw std::runtime_error(
                concatenate("Invalid solution selection method: ", static_cast<int>(ssm)));

        return iter->second;
    }

    SolutionSelectionMethodTypeInfo const& SolutionSelectionMethodTypeInfo::Get(std::string const& str)
    {
        registerAllTypeInfoOnce();

        auto iter = typeNames.find(str);
        if(iter == typeNames.end())
            throw std::runtime_error(concatenate("Invalid solution selection method: ", str));

        return Get(iter->second);
    }

    std::ostream& operator<<(std::ostream& stream, const SolutionSelectionMethod& ssm)
    {
        return stream << ToString(ssm);
    }

    std::istream& operator>>(std::istream& stream, SolutionSelectionMethod& ssm)
    {
        std::string strValue;
        stream >> strValue;

        ssm = SolutionSelectionMethodTypeInfo::Get(strValue).m_solutionSelectionMethod;

        return stream;
    }
} // namespace Tensile
