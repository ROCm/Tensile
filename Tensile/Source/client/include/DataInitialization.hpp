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
            Zero = 0, One, Two, Random, NaN, Inf, BadInput, BadOutput, SerialIdx, SerialDim0, SerialDim1, Count
        };

        static bool IsProblemDependent(InitMode const& mode)
        {
            return mode == InitMode::SerialIdx || mode == InitMode::SerialDim0 || mode == InitMode::SerialDim1;
        }

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

            virtual std::shared_ptr<ContractionInputs> prepareCPUInputs(ContractionProblem const& problem) = 0;
            virtual std::shared_ptr<ContractionInputs> cpuConvInputs() const = 0;
            virtual std::shared_ptr<ContractionInputs> prepareGPUInputs(ContractionProblem const& problem) = 0;

            template <typename T>
            static T getValue(InitMode mode)
            {
                switch(mode)
                {
                    case InitMode::Zero:      return getValue<T, InitMode::Zero>();
                    case InitMode::One:       return getValue<T, InitMode::One>();
                    case InitMode::Two:       return getValue<T, InitMode::Two>();
                    case InitMode::Random:    return getValue<T, InitMode::Random>();
                    case InitMode::NaN:       return getValue<T, InitMode::NaN>();
                    case InitMode::Inf:       return getValue<T, InitMode::Inf>();
                    case InitMode::BadInput:  return getValue<T, InitMode::BadInput>();
                    case InitMode::BadOutput: return getValue<T, InitMode::BadOutput>();
                    case InitMode::SerialIdx:
                    case InitMode::SerialDim0:
                    case InitMode::SerialDim1:
                    case InitMode::Count:  throw std::runtime_error("Invalid InitMode.");
                }
            }

            template <typename T, InitMode Mode>
            static inline T getValue();

            template <typename T>
            static bool isBadInput(T value);

            template <typename T>
            static bool isBadOutput(T value);

            // Fills max buffer size
            template <typename T>
            void initArray(InitMode mode, T * array, size_t elements)
            {
                switch(mode)
                {
                    case InitMode::Zero:      initArray<T, InitMode::Zero  >(array, elements); break;
                    case InitMode::One:       initArray<T, InitMode::One   >(array, elements); break;
                    case InitMode::Two:       initArray<T, InitMode::Two   >(array, elements); break;
                    case InitMode::Random:    initArray<T, InitMode::Random>(array, elements); break;
                    case InitMode::NaN:       initArray<T, InitMode::NaN   >(array, elements); break;
                    case InitMode::Inf:       initArray<T, InitMode::Inf   >(array, elements); break;
                    case InitMode::BadInput:  initArray<T, InitMode::BadInput>(array, elements); break;
                    case InitMode::BadOutput: initArray<T, InitMode::BadOutput>(array, elements); break;
                    case InitMode::SerialIdx:
                    case InitMode::SerialDim0:
                    case InitMode::SerialDim1:
                    case InitMode::Count:  throw std::runtime_error("Invalid InitMode.");
                }
            }

            // For problem dependent data initialization
            template <typename T>
            void initArray(InitMode mode, T * array, TensorDescriptor const& tensor)
            {
                switch(mode)
                {
                    case InitMode::Zero:      initArray<T, InitMode::Zero  >(array, tensor); break;
                    case InitMode::One:       initArray<T, InitMode::One   >(array, tensor); break;
                    case InitMode::Two:       initArray<T, InitMode::Two   >(array, tensor); break;
                    case InitMode::Random:    initArray<T, InitMode::Random>(array, tensor); break;
                    case InitMode::NaN:       initArray<T, InitMode::NaN   >(array, tensor); break;
                    case InitMode::Inf:       initArray<T, InitMode::Inf   >(array, tensor); break;
                    case InitMode::BadInput:  initArray<T, InitMode::BadInput>(array, tensor); break;
                    case InitMode::BadOutput: initArray<T, InitMode::BadOutput>(array, tensor); break;
                    case InitMode::SerialIdx:  initArraySerialIdx<T>(array, tensor); break;
                    case InitMode::SerialDim0: initArraySerialDim0<T>(array, tensor); break;
                    case InitMode::SerialDim1: initArraySerialDim1<T>(array, tensor); break;
                    case InitMode::Count:  throw std::runtime_error("Invalid InitMode.");
                }
            }

            template <typename T, InitMode Mode>
            void initArray(T * array, size_t elements)
            {
                for(size_t i = 0; i < elements; i++)
                {
                    array[i] = getValue<T, Mode>();
                }
            }

            template <typename T, InitMode Mode>
            void initArray(T * array, TensorDescriptor const& tensor)
            {
                size_t elements = tensor.totalAllocatedElements();
                initArray<T, Mode>(array, elements);
            }

            template <typename T>
            void initArraySerialIdx(T * array, TensorDescriptor const& tensor)
            {
                auto const& sizes = tensor.sizes();
                auto count = CoordCount(sizes.begin(), sizes.end());
                std::vector<size_t> coord(tensor.dimensions(), 0);
                for(size_t idx = 0; idx < count; idx++)
                {
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    array[tensor.index(coord)] = static_cast<T>(idx);
                }
            }

            template <typename T>
            void initArraySerialDim0(T * array, TensorDescriptor const& tensor)
            {
                auto const& sizes = tensor.sizes();
                auto count = CoordCount(sizes.begin(), sizes.end());
                std::vector<size_t> coord(tensor.dimensions(), 0);
                for(size_t idx = 0; idx < count; idx++)
                {
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    array[tensor.index(coord)] = static_cast<T>(coord[0]);
                }
            }

            template <typename T>
            void initArraySerialDim1(T * array, TensorDescriptor const& tensor)
            {
                auto const& sizes = tensor.sizes();
                auto count = CoordCount(sizes.begin(), sizes.end());
                std::vector<size_t> coord(tensor.dimensions(), 0);
                for(size_t idx = 0; idx < count; idx++)
                {
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    array[tensor.index(coord)] = static_cast<T>(coord[1]);
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

            /// If true, we will initialize all out-of-bounds inputs to NaN, and
            /// all out-of-bounds outputs to a known value. This allows us to
            /// verify that out-of-bounds values are not used or written to.
            bool m_boundsCheck = false;

            /// If true, the data is dependent on the problem size (e.g. serial)
            /// and must be reinitialized for each problem. Pristine copy on GPU
            /// cannot be used with problem dependent data.
            bool m_problemDependentData = false;
        };

        template <> inline float  DataInitialization::getValue<float,  InitMode::Zero>() { return 0.0f; }
        template <> inline float  DataInitialization::getValue<float,  InitMode::One>()  { return 1.0f; }
        template <> inline float  DataInitialization::getValue<float,  InitMode::Two>()  { return 2.0f; }
        template <> inline float  DataInitialization::getValue<float,  InitMode::NaN>()  { return std::numeric_limits<float>::quiet_NaN(); }
        template <> inline float  DataInitialization::getValue<float,  InitMode::Inf>()  { return std::numeric_limits<float>::infinity(); }

        template <> inline float  DataInitialization::getValue<float, InitMode::Random>()
        {
            return static_cast<float>((rand() % 201) - 100);
        }

        template <> inline float  DataInitialization::getValue<float,  InitMode::BadInput>()  { return getValue<float, InitMode::NaN>(); }
        template <> inline float  DataInitialization::getValue<float,  InitMode::BadOutput>()  { return getValue<float, InitMode::Inf>(); }

        template <> inline double DataInitialization::getValue<double, InitMode::Zero>() { return 0.0; }
        template <> inline double DataInitialization::getValue<double, InitMode::One>()  { return 1.0; }
        template <> inline double DataInitialization::getValue<double, InitMode::Two>()  { return 2.0; }
        template <> inline double DataInitialization::getValue<double, InitMode::NaN>()  { return std::numeric_limits<double>::quiet_NaN(); }
        template <> inline double DataInitialization::getValue<double, InitMode::Inf>()  { return std::numeric_limits<double>::infinity(); }

        template <> inline double DataInitialization::getValue<double, InitMode::Random>()
        {
            return static_cast<double>((rand() % 2001) - 1000);
        }

        template <> inline double  DataInitialization::getValue<double,  InitMode::BadInput>()  { return getValue<double, InitMode::NaN>(); }
        template <> inline double  DataInitialization::getValue<double,  InitMode::BadOutput>()  { return getValue<double, InitMode::Inf>(); }

        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::Zero>() { return std::complex<float>(0.0f, 0.0f); }
        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::One>()  { return std::complex<float>(1.0f, 0.0f); }
        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::Two>()  { return std::complex<float>(2.0f, 0.0f); }
        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::NaN>()  { return std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                                                                                                                                  std::numeric_limits<float>::quiet_NaN()); }

        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::Inf>()  { return std::complex<float>(std::numeric_limits<float>::infinity(),
                                                                                                                                                  std::numeric_limits<float>::infinity()); }

        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>, InitMode::Random>()
        {
            return std::complex<float>(getValue<float, InitMode::Random>(),
                                       getValue<float, InitMode::Random>());
        }

        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::BadInput>()  { return getValue<std::complex<float>, InitMode::NaN>(); }
        template <> inline std::complex<float>  DataInitialization::getValue<std::complex<float>,  InitMode::BadOutput>()  { return getValue<std::complex<float>, InitMode::Inf>(); }

        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::Zero>() { return std::complex<double>(0.0, 0.0); }
        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::One>()  { return std::complex<double>(1.0, 0.0); }
        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::Two>()  { return std::complex<double>(2.0, 0.0); }
        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::NaN>()  { return std::complex<double>(std::numeric_limits<double>::quiet_NaN(),
                                                                                                                                                     std::numeric_limits<double>::quiet_NaN()); }
        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::Inf>()  { return std::complex<double>(std::numeric_limits<double>::infinity(),
                                                                                                                                                     std::numeric_limits<double>::infinity()); }

        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>, InitMode::Random>()
        {
            return std::complex<double>(getValue<double, InitMode::Random>(),
                                        getValue<double, InitMode::Random>());
        }

        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::BadInput>()  { return getValue<std::complex<double>, InitMode::NaN>(); }
        template <> inline std::complex<double>  DataInitialization::getValue<std::complex<double>,  InitMode::BadOutput>()  { return getValue<std::complex<double>, InitMode::Inf>(); }

        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::Zero>() { return 0; }
        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::One>()  { return 1; }
        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::Two>()  { return 2; }
        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::NaN>()  { throw std::runtime_error("NaN not available for int32_t."); }
        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::Inf>()  { throw std::runtime_error("Inf not available for int32_t."); }

        template <> inline int32_t  DataInitialization::getValue<int32_t, InitMode::Random>()
        {
            return rand()%7 - 3;
        }

        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::BadInput>() { return std::numeric_limits<int32_t>::max(); }
        template <> inline int32_t  DataInitialization::getValue<int32_t,  InitMode::BadOutput>() { return std::numeric_limits<int32_t>::min(); }

        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::Zero>() { return Int8x4{0,0,0,0}; }
        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::One>()  { return Int8x4{1,1,1,1}; }
        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::Two>()  { return Int8x4{2,2,2,2}; }
        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::NaN>()  { throw std::runtime_error("NaN not available for Int8x4."); }
        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::Inf>()  { throw std::runtime_error("Inf not available for Int8x4."); }

        template <> inline Int8x4  DataInitialization::getValue<Int8x4, InitMode::Random>()
        {
            return Int8x4{static_cast<int8_t>((rand()%7) - 3),
                          static_cast<int8_t>((rand()%7) - 3),
                          static_cast<int8_t>((rand()%7) - 3),
                          static_cast<int8_t>((rand()%7) - 3)};
        }

        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::BadInput>()
        {
            auto val = std::numeric_limits<int8_t>::max();
            return Int8x4{val, val, val, val};
        }

        template <> inline Int8x4  DataInitialization::getValue<Int8x4,  InitMode::BadOutput>()
        {
            auto val = std::numeric_limits<int8_t>::min();
            return Int8x4{val, val, val, val};
        }

        template <> inline Half  DataInitialization::getValue<Half,  InitMode::Zero>() { return static_cast<Half>(0); }
        template <> inline Half  DataInitialization::getValue<Half,  InitMode::One>()  { return static_cast<Half>(1); }
        template <> inline Half  DataInitialization::getValue<Half,  InitMode::Two>()  { return static_cast<Half>(2); }
        template <> inline Half  DataInitialization::getValue<Half,  InitMode::NaN>()
        {
            union
            {
                uint16_t bits;
                Half value;
            } x;

            x.bits = 0xFFFF;
            return x.value;
        }
        template <> inline Half  DataInitialization::getValue<Half,  InitMode::Inf>()
        {
            union
            {
                uint16_t bits;
                Half value;
            } x;
            
            x.bits = 0x7C00;
            return x.value;
        }

        template <> inline Half  DataInitialization::getValue<Half, InitMode::Random>()
        {
            return static_cast<Half>((rand()%7) - 3);
        }

        template <> inline Half  DataInitialization::getValue<Half,  InitMode::BadInput>()  { return getValue<Half, InitMode::NaN>(); }
        template <> inline Half  DataInitialization::getValue<Half,  InitMode::BadOutput>()  { return getValue<Half, InitMode::Inf>(); }

        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::Zero>() { return static_cast<BFloat16>(0); }
        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::One>()  { return static_cast<BFloat16>(1); }
        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::Two>()  { return static_cast<BFloat16>(2); }
        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::NaN>()  { return static_cast<BFloat16>(std::numeric_limits<float>::quiet_NaN()); }
        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::Inf>()  { return static_cast<BFloat16>(std::numeric_limits<float>::infinity()); }

        template <> inline BFloat16  DataInitialization::getValue<BFloat16, InitMode::Random>()
        {
            return static_cast<BFloat16>((rand()%7) - 3);
        }

        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::BadInput>()  { return getValue<BFloat16, InitMode::NaN>(); }
        template <> inline BFloat16  DataInitialization::getValue<BFloat16,  InitMode::BadOutput>()  { return getValue<BFloat16, InitMode::Inf>(); }

        template <> inline bool DataInitialization::isBadInput<float>(float value)
        {
            return std::isnan(value);
        }

        template <> inline bool DataInitialization::isBadInput<double>(double value)
        {
            return std::isnan(value);
        }

        template <> inline bool DataInitialization::isBadInput<std::complex<float>>(
            std::complex<float> value)
        {
            return std::isnan(value.real()) && std::isnan(value.imag());
        }

        template <> inline bool DataInitialization::isBadInput<std::complex<double>>(
            std::complex<double> value)
        {
            return std::isnan(value.real()) && std::isnan(value.imag());
        }

        template <> inline bool DataInitialization::isBadInput<int32_t>(int32_t value)
        {
            return value == DataInitialization::getValue<int32_t, InitMode::BadInput>();
        }

        template <> inline bool DataInitialization::isBadInput<Int8x4>(Int8x4 value)
        {
            return value == DataInitialization::getValue<Int8x4, InitMode::BadInput>();
        }

        template <> inline bool DataInitialization::isBadInput<Half>(Half value)
        {
            return std::isnan(static_cast<float>(value));
        }

        template <> inline bool DataInitialization::isBadInput<BFloat16>(BFloat16 value)
        {
            return std::isnan(value);
        }

        template <> inline bool DataInitialization::isBadOutput<float>(float value)
        {
            return std::isinf(value);
        }

        template <> inline bool DataInitialization::isBadOutput<double>(double value)
        {
            return std::isinf(value);
        }

        template <> inline bool DataInitialization::isBadOutput<std::complex<float>>(
            std::complex<float> value)
        {
            return std::isinf(value.real()) && std::isinf(value.imag());
        }

        template <> inline bool DataInitialization::isBadOutput<std::complex<double>>(
            std::complex<double> value)
        {
            return std::isinf(value.real()) && std::isinf(value.imag());
        }

        template <> inline bool DataInitialization::isBadOutput<int32_t>(int32_t value)
        {
            return value == DataInitialization::getValue<int32_t, InitMode::BadOutput>();
        }

        template <> inline bool DataInitialization::isBadOutput<Int8x4>(Int8x4 value)
        {
            return value == DataInitialization::getValue<Int8x4, InitMode::BadOutput>();
        }

        template <> inline bool DataInitialization::isBadOutput<Half>(Half value)
        {
            return std::isinf(static_cast<float>(value));
        }

        template <> inline bool DataInitialization::isBadOutput<BFloat16>(BFloat16 value)
        {
            return std::isinf(value);
        }
    }
}

