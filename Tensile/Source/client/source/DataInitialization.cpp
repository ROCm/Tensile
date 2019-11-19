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

#include "DataInitialization.hpp"
#include "DataInitializationTyped.hpp"

#include <Tensile/Utils.hpp>

#include <hip/hip_runtime.h>

namespace Tensile
{
    namespace Client
    {
        std::string ToString(InitMode mode)
        {
            switch(mode)
            {
                case InitMode::Zero:    return "Zero";
                case InitMode::One:     return "One";
                case InitMode::Two:     return "Two";
                case InitMode::Random:  return "Random";
                case InitMode::NaN:     return "NaN";
                case InitMode::SerialIdx: return "SerialIdx";
                // case InitMode::SerialDim0: return "SerialDim0";
                // case InitMode::SerialDim1: return "SerialDim1";
                case InitMode::Count:   break;
            }

            throw std::runtime_error(concatenate("Invalid InitMode value: ", static_cast<int>(mode)));
        }

        std::ostream & operator<<(std::ostream & stream, InitMode const& mode)
        {
            return stream << ToString(mode);
        }

        std::istream & operator>>(std::istream & stream, InitMode      & mode)
        {
            std::string strValue;
            stream >> strValue;

            if(     strValue == ToString(InitMode::Zero))      mode = InitMode::Zero;
            else if(strValue == ToString(InitMode::One))       mode = InitMode::One;
            else if(strValue == ToString(InitMode::Two))       mode = InitMode::Two;
            else if(strValue == ToString(InitMode::Random))    mode = InitMode::Random;
            else if(strValue == ToString(InitMode::NaN))       mode = InitMode::NaN;
            else if(strValue == ToString(InitMode::SerialIdx)) mode = InitMode::SerialIdx;
            // else if (strValue == ToString(InitMode::SerialDim0)) mode = InitMode::SerialDim0;
            // else if (strValue == ToString(InitMode::SerialDim1)) mode = InitMode::SerialDim1;
            else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
            {
                int value = atoi(strValue.c_str());
                if(value >= 0 && value < static_cast<int>(InitMode::Count))
                    mode = static_cast<InitMode>(value);
                else
                    throw std::runtime_error(concatenate("Can't convert ", strValue, " to InitMode."));
            }
            else
            {
                throw std::runtime_error(concatenate("Can't convert ", strValue, " to InitMode."));
            }

            return stream;
        }

        double DataInitialization::GetRepresentativeBetaValue(po::variables_map const& args)
        {
            auto argValue = args["init-beta"].as<int>();

            if(argValue == 0)
                return 0.0;

            if(argValue == 1)
                return 1.0;

            return 1.5;
        }

        template <typename TypedInputs>
        std::shared_ptr<TypedDataInitialization<TypedInputs>> DataInitialization::GetTyped(
                po::variables_map const& args, ClientProblemFactory const& problemFactory)
        {
            auto * ptr = new TypedDataInitialization<TypedInputs>(args, problemFactory);

            return std::shared_ptr<TypedDataInitialization<TypedInputs>>(ptr);
        }

        std::shared_ptr<DataInitialization> DataInitialization::Get(
                po::variables_map const& args, ClientProblemFactory const& problemFactory)
        {
            auto aType     = args["a-type"].as<DataType>();
            auto bType     = args["b-type"].as<DataType>();
            auto cType     = args["c-type"].as<DataType>();
            auto dType     = args["d-type"].as<DataType>();
            auto alphaType = args["alpha-type"].as<DataType>();
            auto betaType  = args["beta-type"].as<DataType>();

            if(aType == DataType::Float && bType == DataType::Float
            && cType == DataType::Float && dType == DataType::Float
            && alphaType == DataType::Float && betaType == DataType::Float)
            {
                return GetTyped<TypedContractionInputs<float>>(args, problemFactory);
            }
            else if(aType == DataType::Double && bType == DataType::Double
                 && cType == DataType::Double && dType == DataType::Double
                 && alphaType == DataType::Double && betaType == DataType::Double)
            {
                return GetTyped<TypedContractionInputs<double>>(args, problemFactory);
            }
            else if(aType == DataType::ComplexFloat && bType == DataType::ComplexFloat
                 && cType == DataType::ComplexFloat && dType == DataType::ComplexFloat
                 && alphaType == DataType::ComplexFloat && betaType == DataType::ComplexFloat)
            {
                return GetTyped<TypedContractionInputs<std::complex<float>>>(args, problemFactory);
            }
            else if(aType == DataType::ComplexDouble && bType == DataType::ComplexDouble
                 && cType == DataType::ComplexDouble && dType == DataType::ComplexDouble
                 && alphaType == DataType::ComplexDouble && betaType == DataType::ComplexDouble)
            {
                return GetTyped<TypedContractionInputs<std::complex<double>>>(args, problemFactory);
            }
            else if(aType == DataType::Half && bType == DataType::Half
                 && cType == DataType::Half && dType == DataType::Half
                 && alphaType == DataType::Half && betaType == DataType::Half)
            {
                return GetTyped<TypedContractionInputs<Half>>(args, problemFactory);
            }
            else if(aType == DataType::Int8x4 && bType == DataType::Int8x4
                 && cType == DataType::Int32 && dType == DataType::Int32
                 && alphaType == DataType::Int32 && betaType == DataType::Int32)
            {
                return GetTyped<TypedContractionInputs<Int8x4, Int8x4, int32_t, int32_t>>(args, problemFactory);
            }
            else if(aType == DataType::Int32 && bType == DataType::Int32
                 && cType == DataType::Int32 && dType == DataType::Int32
                 && alphaType == DataType::Int32 && betaType == DataType::Int32)
            {
                return GetTyped<TypedContractionInputs<int32_t>>(args, problemFactory);
            }
            else if(aType == DataType::BFloat16 && bType == DataType::BFloat16
                 && cType == DataType::BFloat16 && dType == DataType::BFloat16
                 && alphaType == DataType::Float && betaType == DataType::Float)
            {
                return GetTyped<BFloat16ContractionInputs>(args, problemFactory);
            }

            throw std::runtime_error(concatenate("Invalid combination of data types: ",
                                                 "a: ", aType, ", b: ", bType, ", c: ", cType, ", d: ", dType,
                                                 ", alpha: ", alphaType, ", beta: ", betaType));
        }

        DataInitialization::DataInitialization(po::variables_map const& args, ClientProblemFactory const& problemFactory)
            : m_aInit(    args["init-a"].as<InitMode>()),
              m_bInit(    args["init-b"].as<InitMode>()),
              m_cInit(    args["init-c"].as<InitMode>()),
              m_dInit(    args["init-d"].as<InitMode>()),
              m_alphaInit(args["init-alpha"].as<InitMode>()),
              m_betaInit( args["init-beta"].as<InitMode>()),
              m_aMaxElements(0),
              m_bMaxElements(0),
              m_cMaxElements(0),
              m_dMaxElements(0),
              m_cEqualsD(args["c-equal-d"].as<bool>()),
              m_keepPristineCopyOnGPU(args["pristine-on-gpu"].as<bool>())
        {
            if (args.count("convolution-vs-contraction"))
                m_convolutionVsContraction = args["convolution-vs-contraction"].as<bool>();

            for(auto const& problem: problemFactory.problems())
            {
                m_aMaxElements = std::max(m_aMaxElements, problem.a().totalLogicalElements());
                m_bMaxElements = std::max(m_bMaxElements, problem.b().totalLogicalElements());
                m_cMaxElements = std::max(m_cMaxElements, problem.c().totalLogicalElements());
                m_dMaxElements = std::max(m_dMaxElements, problem.d().totalLogicalElements());
            }
        }

        DataInitialization::~DataInitialization()
        {
        }
    }
}

