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

#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>

namespace Tensile
{
    std::map<KernelLanguage, KernelLanguageTypeInfo> KernelLanguageTypeInfo::data;
    std::map<std::string, KernelLanguage>            KernelLanguageTypeInfo::typeNames;

    std::string ToString(KernelLanguage d)
    {
        switch(d)
        {
        case KernelLanguage::Any:
            return "Any";
        case KernelLanguage::Assembly:
            return "Assembly";
        case KernelLanguage::Source:
            return "Source";

        case KernelLanguage::Count:
        default:;
        }
        return "Invalid";
    }

    std::string TypeAbbrev(KernelLanguage d)
    {
        switch(d)
        {
        case KernelLanguage::Any:
            return "Any";
        case KernelLanguage::Assembly:
            return "Asm";
        case KernelLanguage::Source:
            return "Src";

        case KernelLanguage::Count:
        default:;
        }
        return "Invalid";
    }

    template <KernelLanguage T>
    void KernelLanguageTypeInfo::registerTypeInfo()
    {
        using T_Info = KernelLanguageInfo<T>;

        KernelLanguageTypeInfo info;

        info.m_kernelLanguage = T_Info::Enum;
        info.name             = T_Info::Name();
        info.abbrev           = T_Info::Abbrev();

        addInfoObject(info);
    }

    void KernelLanguageTypeInfo::registerAllTypeInfo()
    {
        registerTypeInfo<KernelLanguage::Any>();
        registerTypeInfo<KernelLanguage::Assembly>();
        registerTypeInfo<KernelLanguage::Source>();
    }

    void KernelLanguageTypeInfo::registerAllTypeInfoOnce()
    {
        static int call_once = (registerAllTypeInfo(), 0);
    }

    void KernelLanguageTypeInfo::addInfoObject(KernelLanguageTypeInfo const& info)
    {
        auto toLower = [](std::string tmp) {
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
            return tmp;
        };

        data[info.m_kernelLanguage] = info;

        // Add some flexibility to names registry. Accept abbreviations and
        // lower case versions of the strings
        typeNames[info.name]            = info.m_kernelLanguage;
        typeNames[toLower(info.name)]   = info.m_kernelLanguage;
        typeNames[info.abbrev]          = info.m_kernelLanguage;
        typeNames[toLower(info.abbrev)] = info.m_kernelLanguage;
    }

    KernelLanguageTypeInfo const& KernelLanguageTypeInfo::Get(int index)
    {
        return Get(static_cast<KernelLanguage>(index));
    }

    KernelLanguageTypeInfo const& KernelLanguageTypeInfo::Get(KernelLanguage t)
    {
        registerAllTypeInfoOnce();

        auto iter = data.find(t);
        if(iter == data.end())
            throw std::runtime_error(concatenate("Invalid kernel language: ", static_cast<int>(t)));

        return iter->second;
    }

    KernelLanguageTypeInfo const& KernelLanguageTypeInfo::Get(std::string const& str)
    {
        registerAllTypeInfoOnce();

        auto iter = typeNames.find(str);
        if(iter == typeNames.end())
            throw std::runtime_error(concatenate("Invalid kernel language: ", str));

        return Get(iter->second);
    }

    std::ostream& operator<<(std::ostream& stream, const KernelLanguage& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, KernelLanguage& t)
    {
        std::string strValue;
        stream >> strValue;

        t = KernelLanguageTypeInfo::Get(strValue).m_kernelLanguage;

        return stream;
    }
} // namespace Tensile
