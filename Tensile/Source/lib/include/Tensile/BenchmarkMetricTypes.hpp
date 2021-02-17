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

#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace Tensile
{
    /**
 * \ingroup Tensile
 * \defgroup BenchmarkMetrics Benchmark metric type Info
 *
 * @brief Definitions and metadata on supported benchmark metric types.
 */

    /**
 * \ingroup BenchmarkMetrics
 * @{
 */

    /**
 * Benchmark Metric
 */
    enum class BenchmarkMetric : int
    {
        Best,
        CUEfficiency,
        Overall,
        Count
    };

    std::string   ToString(BenchmarkMetric d);
    std::string   TypeAbbrev(BenchmarkMetric d);
    std::ostream& operator<<(std::ostream& stream, BenchmarkMetric const& t);
    std::istream& operator>>(std::istream& stream, BenchmarkMetric& t);

    /**
 * \ingroup BenchmarkMetrics
 * \brief Runtime accessible benchmark metric type metadata
 */
    struct BenchmarkMetricTypeInfo
    {
        static BenchmarkMetricTypeInfo const& Get(int index);
        static BenchmarkMetricTypeInfo const& Get(BenchmarkMetric t);
        static BenchmarkMetricTypeInfo const& Get(std::string const& str);

        BenchmarkMetric m_benchmarkMetric;
        std::string     name;
        std::string     abbrev;

    private:
        static void registerAllTypeInfo();
        static void registerAllTypeInfoOnce();

        template <BenchmarkMetric T_Enum>
        static void registerTypeInfo();

        static void addInfoObject(BenchmarkMetricTypeInfo const& info);

        static std::map<BenchmarkMetric, BenchmarkMetricTypeInfo> data;
        static std::map<std::string, BenchmarkMetric>             typeNames;
    };

    /**
 * \ingroup BenchmarkMetrics
 * \brief Compile-time accessible benchmark metric type metadata.
 */
    template <BenchmarkMetric T_Enum>
    struct BenchmarkMetricInfo
    {
    };

    template <BenchmarkMetric T_Enum>
    struct BaseBenchmarkMetricInfo
    {
        constexpr static BenchmarkMetric Enum = T_Enum;

        static inline std::string Name()
        {
            return ToString(Enum);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Enum);
        }
    };

    template <BenchmarkMetric T_Enum>
    constexpr BenchmarkMetric BaseBenchmarkMetricInfo<T_Enum>::Enum;

    template <>
    struct BenchmarkMetricInfo<BenchmarkMetric::Best>
        : public BaseBenchmarkMetricInfo<BenchmarkMetric::Best>
    {
    };
    template <>
    struct BenchmarkMetricInfo<BenchmarkMetric::CUEfficiency>
        : public BaseBenchmarkMetricInfo<BenchmarkMetric::CUEfficiency>
    {
    };
    template <>
    struct BenchmarkMetricInfo<BenchmarkMetric::Overall>
        : public BaseBenchmarkMetricInfo<BenchmarkMetric::Overall>
    {
    };

    /**
 * @}
 */
} // namespace Tensile

namespace std
{
    template <>
    struct hash<Tensile::BenchmarkMetric>
    {
        inline size_t operator()(Tensile::BenchmarkMetric const& val) const
        {
            return hash<int>()(static_cast<int>(val));
        }
    };
} // namespace std
