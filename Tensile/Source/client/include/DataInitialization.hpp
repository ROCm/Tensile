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

#include <boost/program_options.hpp>

#include <Tensile/ContractionProblem.hpp>

#include "ClientProblemFactory.hpp"

#include <cstddef>

namespace po = boost::program_options;

namespace Tensile
{
    namespace Client
    {
        enum class InitMode
        {
            Zero = 0, One, Two, Random, NaN, SerialIdx, /*SerialDim0, SerialDim1,*/ Count
        };

        std::string ToString(InitMode mode);

        std::ostream & operator<<(std::ostream & stream, InitMode const& mode);
        std::istream & operator>>(std::istream & stream, InitMode      & mode);

        template <typename TypedInputs>
        class TypedDataInitialization;

        class DataInitialization
        {
        public:
            static double GetRepresentativeBetaValue(po::variables_map const& args);

            static std::shared_ptr<DataInitialization> Get(
                    po::variables_map const& args, ClientProblemFactory const& problemFactory);

            template <typename TypedInputs>
            static std::shared_ptr<TypedDataInitialization<TypedInputs>> GetTyped(
                    po::variables_map const& args, ClientProblemFactory const& problemFactory);

            DataInitialization(po::variables_map const& args, ClientProblemFactory const& problemFactory);
            ~DataInitialization();

            virtual std::shared_ptr<ContractionInputs> prepareCPUInputs() = 0;
            virtual std::shared_ptr<ContractionInputs> cpuConvInputs() const = 0;
            virtual std::shared_ptr<ContractionInputs> prepareGPUInputs() = 0;

            template <typename T>
            static T getValue(InitMode mode, int idx)
            {
                switch(mode)
                {
                    case InitMode::Zero:   return getValue<T, InitMode::Zero>(idx);
                    case InitMode::One:    return getValue<T, InitMode::One>(idx);
                    case InitMode::Two:    return getValue<T, InitMode::Two>(idx);
                    case InitMode::Random: return getValue<T, InitMode::Random>(idx);
                    case InitMode::NaN:    return getValue<T, InitMode::NaN>(idx);
                    case InitMode::SerialIdx: return getValue<T, InitMode::SerialIdx>(idx);
                    // case InitMode::SerialDim0: throw std::runtime_error("Invalid InitMode."); // return getValue<T, InitMode::SerialDim0>(idx);
                    // case InitMode::SerialDim1: throw std::runtime_error("Invalid InitMode."); // return getValue<T, InitMode::SerialDim1>(idx);
                    case InitMode::Count:  throw std::runtime_error("Invalid InitMode.");
                }
            }

            template <typename T, InitMode Mode>
            static inline T getValue(int idx);

            template <typename T>
            void initArray(InitMode mode, T * array, size_t elements)
            {
                switch(mode)
                {
                    case InitMode::Zero:   initArray<T, InitMode::Zero  >(array, elements); break;
                    case InitMode::One:    initArray<T, InitMode::One   >(array, elements); break;
                    case InitMode::Two:    initArray<T, InitMode::Two   >(array, elements); break;
                    case InitMode::Random: initArray<T, InitMode::Random>(array, elements); break;
                    case InitMode::NaN:    initArray<T, InitMode::NaN   >(array, elements); break;
                    case InitMode::SerialIdx: initArray<T, InitMode::SerialIdx>(array, elements); break;
                    // case InitMode::SerialDim0: initArray<T, InitMode::SerialDim0>(array, elements); break;
                    // case InitMode::SerialDim1: initArray<T, InitMode::SerialDim1>(array, elements); break;
                    case InitMode::Count:  throw std::runtime_error("Invalid InitMode.");
                }
            }

            template <typename T, InitMode Mode>
            void initArray(T * array, size_t elements)
            {
                for(int i = 0; i < elements; i++)
                {
                    array[i] = getValue<T, Mode>(i);
                }
            }

        protected:
            InitMode m_aInit, m_bInit, m_cInit, m_dInit;
            InitMode m_alphaInit, m_betaInit;

            size_t m_aMaxElements;
            size_t m_bMaxElements;
            size_t m_cMaxElements;
            size_t m_dMaxElements;

            bool m_cEqualsD;
            bool m_convolutionVsContraction;


            /// If true, we will allocate an extra copy of the inputs on the GPU.
            /// This will improve performance as we don't have to copy from the CPU
            /// with each kernel launch, but it will use extra memory.
            bool m_keepPristineCopyOnGPU = true;
        };

        template <> inline float  DataInitialization::getValue<float,  InitMode::Zero>(int idx) { return 0.0f; }
        template <> inline float  DataInitialization::getValue<float,  InitMode::One>(int idx)  { return 1.0f; }
        template <> inline float  DataInitialization::getValue<float,  InitMode::Two>(int idx)  { return 2.0f; }
        template <> inline float  DataInitialization::getValue<float,  InitMode::NaN>(int idx)  { return std::numeric_limits<float>::quiet_NaN(); }

        template <> inline float  DataInitialization::getValue<float, InitMode::Random>(int idx)
        {
            return static_cast<float>((rand() % 201) - 100);
        }

        template <> inline float DataInitialization::getValue<float, InitMode::SerialIdx>(int idx)
        {
            return idx;
        }

        // template <> inline float DataInitialization::getValue<float, InitMode::SerialDim0>()
        // {
        //     return TODO;
        // }

        // template <> inline float DataInitialization::getValue<float, InitMode::SerialDim1>()
        // {
        //     return TODO;
        // }

        template <> inline double DataInitialization::getValue<double, InitMode::Zero>(int idx) { return 0.0; }
        template <> inline double DataInitialization::getValue<double, InitMode::One>(int idx)  { return 1.0; }
        template <> inline double DataInitialization::getValue<double, InitMode::Two>(int idx)  { return 2.0; }
        template <> inline double DataInitialization::getValue<double, InitMode::NaN>(int idx)  { return std::numeric_limits<double>::quiet_NaN(); }

        template <> inline double DataInitialization::getValue<double, InitMode::Random>(int idx)
        {
            return static_cast<double>((rand() % 2001) - 1000);
        }

        template <> inline double DataInitialization::getValue<double, InitMode::SerialIdx>(int idx)
        {
            return idx;
        }

        // template <> inline double DataInitialization::getValue<float, InitMode::SerialDim0>()
        // {
        //     return TODO;
        // }

        // template <> inline double DataInitialization::getValue<float, InitMode::SerialDim1>()
        // {
        //     return TODO;
        // }

        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::Zero>(int idx) { return std::complex<float>(0.0f, 0.0f); }
        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::One>(int idx)  { return std::complex<float>(1.0f, 0.0f); }
        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::Two>(int idx)  { return std::complex<float>(2.0f, 0.0f); }
        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::NaN>(int idx)  { return std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                                                                                                                                  std::numeric_limits<float>::quiet_NaN()); }

        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>, InitMode::Random>(int idx)
        {
            return std::complex<float>(getValue<float, InitMode::Random>(idx),
                                       getValue<float, InitMode::Random>(idx));
        }

        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>, InitMode::SerialIdx>(int idx)
        {
            return std::complex<float>(idx, 0.0f);
        }

        // template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>, InitMode::SerialDim0>()
        // {
        //     return TODO;
        // }

        // template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>, InitMode::SerialDim1>()
        // {
        //     return TODO;
        // }

        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::Zero>(int idx) { return std::complex<double>(0.0, 0.0); }
        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::One>(int idx)  { return std::complex<double>(1.0, 0.0); }
        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::Two>(int idx)  { return std::complex<double>(2.0, 0.0); }
        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::NaN>(int idx)  { return std::complex<double>(std::numeric_limits<double>::quiet_NaN(),
                                                                                                                                                     std::numeric_limits<double>::quiet_NaN()); }

        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>, InitMode::Random>(int idx)
        {
            return std::complex<double>(getValue<double, InitMode::Random>(idx),
                                        getValue<double, InitMode::Random>(idx));
        }

        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>, InitMode::SerialIdx>(int idx)
        {
            return std::complex<double>(idx, 0.0);
        }

        // template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>, InitMode::SerialDim0>()
        // {
        //     return TODO;
        // }

        // template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>, InitMode::SerialDim1>()
        // {
        //     return TODO;
        // }

        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::Zero>(int idx) { return 0; }
        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::One>(int idx)  { return 1; }
        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::Two>(int idx)  { return 2; }
        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::NaN>(int idx)  { throw std::runtime_error("NaN not available for int32_t."); }

        template <> inline int32_t  DataInitialization::getValue<int32_t, InitMode::Random>(int idx)
        {
            return rand()%7 - 3;
        }

        template <> inline int32_t  DataInitialization::getValue<int32_t, InitMode::SerialIdx>(int idx)
        {
            return idx;
        }

        // template <> inline int32_t  DataInitialization::getValue<int32_t, InitMode::SerialDim0>()
        // {
        //     return TODO;
        // }

        // template <> inline int32_t  DataInitialization::getValue<int32_t, InitMode::SerialDim1>()
        // {
        //     return TODO;
        // }

        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::Zero>(int idx) { return Int8x4{0,0,0,0}; }
        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::One>(int idx)  { return Int8x4{1,1,1,1}; }
        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::Two>(int idx)  { return Int8x4{2,2,2,2}; }
        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::NaN>(int idx)  { throw std::runtime_error("NaN not available for Int8x4."); }

        template <> inline Int8x4  DataInitialization::getValue<Int8x4, InitMode::Random>(int idx)
        {
            return Int8x4{static_cast<int8_t>((rand()%7) - 3),
                          static_cast<int8_t>((rand()%7) - 3),
                          static_cast<int8_t>((rand()%7) - 3),
                          static_cast<int8_t>((rand()%7) - 3)};
        }

        template <> inline Int8x4  DataInitialization::getValue<Int8x4, InitMode::SerialIdx>(int idx)
        {
            throw std::runtime_error("NaN not available for Int8x4.");
        }

        // template <> inline Int8x4  DataInitialization::getValue<Int8x4, InitMode::SerialDim0>()
        // {
        //     return TODO;
        // }

        // template <> inline Int8x4  DataInitialization::getValue<Int8x4, InitMode::SerialDim1>()
        // {
        //     return TODO;
        // }

        template <> inline Half  DataInitialization::getValue<Half,  InitMode::Zero>(int idx) { return static_cast<Half>(0); }
        template <> inline Half  DataInitialization::getValue<Half,  InitMode::One>(int idx)  { return static_cast<Half>(1); }
        template <> inline Half  DataInitialization::getValue<Half,  InitMode::Two>(int idx)  { return static_cast<Half>(2); }
        template <> inline Half  DataInitialization::getValue<Half,  InitMode::NaN>(int idx)  { return std::numeric_limits<Half>::quiet_NaN(); }

        template <> inline Half  DataInitialization::getValue<Half, InitMode::Random>(int idx)
        {
            return static_cast<Half>((rand()%7) - 3);
        }

        template <> inline Half  DataInitialization::getValue<Half, InitMode::SerialIdx>(int idx)
        {
            return static_cast<Half>(idx);
        }

        // template <> inline Half  DataInitialization::getValue<Half, InitMode::SerialDim0>()
        // {
        //     return TODO;
        // }

        // template <> inline Half  DataInitialization::getValue<Half, InitMode::SerialDim1>()
        // {
        //     return TODO;
        // }

        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::Zero>(int idx) { return static_cast<BFloat16>(0); }
        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::One>(int idx)  { return static_cast<BFloat16>(1); }
        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::Two>(int idx)  { return static_cast<BFloat16>(2); }
        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::NaN>(int idx)  { return static_cast<BFloat16>(std::numeric_limits<float>::quiet_NaN()); }

        template <> inline BFloat16  DataInitialization::getValue<BFloat16, InitMode::Random>(int idx)
        {
            return static_cast<BFloat16>((rand()%7) - 3);
        }

        template <> inline BFloat16  DataInitialization::getValue<BFloat16, InitMode::SerialIdx>(int idx)
        {
            return static_cast<BFloat16>(idx);
        }

        // template <> inline BFloat16  DataInitialization::getValue<BFloat16, InitMode::SerialDim0>()
        // {
        //     return TODO;
        // }

        // template <> inline BFloat16  DataInitialization::getValue<BFloat16, InitMode::SerialDim1>()
        // {
        //     return TODO;
        // }
    }
}

