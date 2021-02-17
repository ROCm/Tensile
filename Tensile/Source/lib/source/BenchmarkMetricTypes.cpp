/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
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

#include <Tensile/BenchmarkMetricTypes.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>

namespace Tensile
{
    std::map<BenchmarkMetric, BenchmarkMetricTypeInfo> BenchmarkMetricTypeInfo::data;
    std::map<std::string, BenchmarkMetric>             BenchmarkMetricTypeInfo::typeNames;

    std::string ToString(BenchmarkMetric d)
    {
        switch(d)
        {
        case BenchmarkMetric::Best:
            return "Best";
        case BenchmarkMetric::CUEfficiency:
            return "CUEfficiency";
        case BenchmarkMetric::Overall:
            return "Overall";

        case BenchmarkMetric::Count:
        default:;
        }
        return "Invalid";
    }

    std::string TypeAbbrev(BenchmarkMetric d)
    {
        switch(d)
        {
        case BenchmarkMetric::Best:
            return "Best";
        case BenchmarkMetric::CUEfficiency:
            return "CUEff";
        case BenchmarkMetric::Overall:
            return "Ovrl";

        case BenchmarkMetric::Count:
        default:;
        }
        return "Invalid";
    }

    template <BenchmarkMetric T>
    void BenchmarkMetricTypeInfo::registerTypeInfo()
    {
        using T_Info = BenchmarkMetricInfo<T>;

        BenchmarkMetricTypeInfo info;

        info.m_benchmarkMetric = T_Info::Enum;
        info.name              = T_Info::Name();
        info.abbrev            = T_Info::Abbrev();

        addInfoObject(info);
    }

    void BenchmarkMetricTypeInfo::registerAllTypeInfo()
    {
        registerTypeInfo<BenchmarkMetric::Best>();
        registerTypeInfo<BenchmarkMetric::CUEfficiency>();
        registerTypeInfo<BenchmarkMetric::Overall>();
    }

    void BenchmarkMetricTypeInfo::registerAllTypeInfoOnce()
    {
        static int call_once = (registerAllTypeInfo(), 0);
    }

    void BenchmarkMetricTypeInfo::addInfoObject(BenchmarkMetricTypeInfo const& info)
    {
        auto toLower = [](std::string tmp) {
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
            return tmp;
        };

        data[info.m_benchmarkMetric] = info;

        // Add some flexibility to names registry. Accept abbreviations and
        // lower case versions of the strings
        typeNames[info.name]            = info.m_benchmarkMetric;
        typeNames[toLower(info.name)]   = info.m_benchmarkMetric;
        typeNames[info.abbrev]          = info.m_benchmarkMetric;
        typeNames[toLower(info.abbrev)] = info.m_benchmarkMetric;
    }

    BenchmarkMetricTypeInfo const& BenchmarkMetricTypeInfo::Get(int index)
    {
        return Get(static_cast<BenchmarkMetric>(index));
    }

    BenchmarkMetricTypeInfo const& BenchmarkMetricTypeInfo::Get(BenchmarkMetric t)
    {
        registerAllTypeInfoOnce();

        auto iter = data.find(t);
        if(iter == data.end())
            throw std::runtime_error(
                concatenate("Invalid benchmark metric: ", static_cast<int>(t)));

        return iter->second;
    }

    BenchmarkMetricTypeInfo const& BenchmarkMetricTypeInfo::Get(std::string const& str)
    {
        registerAllTypeInfoOnce();

        auto iter = typeNames.find(str);
        if(iter == typeNames.end())
            throw std::runtime_error(concatenate("Invalid benchmark metric: ", str));

        return Get(iter->second);
    }

    std::ostream& operator<<(std::ostream& stream, const BenchmarkMetric& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, BenchmarkMetric& t)
    {
        std::string strValue;
        stream >> strValue;

        t = BenchmarkMetricTypeInfo::Get(strValue).m_benchmarkMetric;

        return stream;
    }
} // namespace Tensile
