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
 * \defgroup SolutionSelectionMethods Solution selection method type Info
 *
 * @brief Definitions and metadata on supported solution selection method types.
 */

    /**
 * \ingroup SolutionSelectionMethods
 * @{
 */

    /**
 * Solution Selection Method
 */
    enum class SolutionSelectionMethod : int
    {
        Auto,
        CUEfficiency,
        DeviceEfficiency,
        Experimental,
        Count
    };

    std::string   ToString(SolutionSelectionMethod ssm);
    std::string   TypeAbbrev(SolutionSelectionMethod ssm);
    std::ostream& operator<<(std::ostream& stream, SolutionSelectionMethod const& ssm);
    std::istream& operator>>(std::istream& stream, SolutionSelectionMethod& ssm);

    /**
 * \ingroup SolutionSelectionMethods
 * \brief Runtime accessible performance metric type metadata
 */
    struct SolutionSelectionMethodTypeInfo
    {
        static SolutionSelectionMethodTypeInfo const& Get(int index);
        static SolutionSelectionMethodTypeInfo const& Get(SolutionSelectionMethod t);
        static SolutionSelectionMethodTypeInfo const& Get(std::string const& str);

        SolutionSelectionMethod m_solutionSelectionMethod;
        std::string             name;
        std::string             abbrev;

    private:
        static void registerAllTypeInfo();
        static void registerAllTypeInfoOnce();

        template <SolutionSelectionMethod T_Enum>
        static void registerTypeInfo();

        static void addInfoObject(SolutionSelectionMethodTypeInfo const& info);

        static std::map<SolutionSelectionMethod, SolutionSelectionMethodTypeInfo> data;
        static std::map<std::string, SolutionSelectionMethod>                     typeNames;
    };

    /**
 * \ingroup SolutionSelectionMethods
 * \brief Compile-time accessible solution selection method type metadata.
 */
    template <SolutionSelectionMethod T_Enum>
    struct SolutionSelectionMethodInfo;

    template <SolutionSelectionMethod T_Enum>
    struct BaseSolutionSelectionMethodInfo
    {
        constexpr static SolutionSelectionMethod Enum = T_Enum;

        static inline std::string Name()
        {
            return ToString(Enum);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Enum);
        }
    };

    template <SolutionSelectionMethod T_Enum>
    constexpr SolutionSelectionMethod BaseSolutionSelectionMethodInfo<T_Enum>::Enum;

    template <>
    struct SolutionSelectionMethodInfo<SolutionSelectionMethod::Auto>
        : public BaseSolutionSelectionMethodInfo<SolutionSelectionMethod::Auto>
    {
    };
    template <>
    struct SolutionSelectionMethodInfo<SolutionSelectionMethod::CUEfficiency>
        : public BaseSolutionSelectionMethodInfo<SolutionSelectionMethod::CUEfficiency>
    {
    };
    template <>
    struct SolutionSelectionMethodInfo<SolutionSelectionMethod::DeviceEfficiency>
        : public BaseSolutionSelectionMethodInfo<SolutionSelectionMethod::DeviceEfficiency>
    {
    };
    template <>
    struct SolutionSelectionMethodInfo<SolutionSelectionMethod::Experimental>
        : public BaseSolutionSelectionMethodInfo<SolutionSelectionMethod::Experimental>
    {
    };


    /**
 * @}
 */
} // namespace Tensile

namespace std
{
    template <>
    struct hash<Tensile::SolutionSelectionMethod>
    {
        inline size_t operator()(Tensile::SolutionSelectionMethod const& val) const
        {
            return hash<int>()(static_cast<int>(val));
        }
    };
} // namespace std
