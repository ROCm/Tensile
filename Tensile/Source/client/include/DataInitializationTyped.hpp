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

#include "DataInitialization.hpp"

#include <hip/hip_runtime.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/hip/HipUtils.hpp>

namespace Tensile
{
    namespace Client
    {
        template <typename A, typename B = A, typename C = B, typename D = C, typename Alpha = D, typename Beta = Alpha>
        struct ManagedContractionInputs: public TypedContractionInputs<A, B, C, D, Alpha, Beta>
        {
            using Base = TypedContractionInputs<A, B, C, D, Alpha, Beta>;
            using AType = A;
            using BType = B;
            using CType = C;
            using DType = D;
            using AlphaType = Alpha;
            using BetaType = Beta;

            ManagedContractionInputs(std::shared_ptr<A> _a, std::shared_ptr<B> _b, std::shared_ptr<C> _c, std::shared_ptr<D> _d,
                                     Alpha _alpha, Beta _beta, bool _gpu)
                : Base(_a.get(), _b.get(), _c.get(), _d.get(), _alpha, _beta),
                  managedA(_a),
                  managedB(_b),
                  managedC(_c),
                  managedD(_d),
                  gpu(_gpu)
            {
            }

            ~ManagedContractionInputs() = default;

            std::shared_ptr<A> managedA;
            std::shared_ptr<B> managedB;
            std::shared_ptr<C> managedC;
            std::shared_ptr<D> managedD;

            bool gpu;

        };

        template <typename TypedInputs>
        class TypedDataInitialization: public DataInitialization
        {
        public:
            using AType     = typename TypedInputs::AType;
            using BType     = typename TypedInputs::BType;
            using CType     = typename TypedInputs::CType;
            using DType     = typename TypedInputs::DType;
            using AlphaType = typename TypedInputs::AlphaType;
            using BetaType  = typename TypedInputs::BetaType;
            using ManagedInputs = ManagedContractionInputs<AType, BType, CType, DType, AlphaType, BetaType>;

            TypedDataInitialization(po::variables_map const& args, ClientProblemFactory const& problemFactory)
                : DataInitialization(args, problemFactory)
            {
            }

            virtual std::shared_ptr<ContractionInputs> prepareCPUInputs()
            {
                return prepareCPUInputsTyped();
            }

            virtual std::shared_ptr<ContractionInputs> prepareGPUInputs()
            {
                return prepareGPUInputsTyped();
            }

            std::shared_ptr<ManagedInputs> prepareCPUInputsTyped()
            {
                if(!m_cpuInputsPristine)
                    m_cpuInputsPristine = createNewCPUInputs();

                if(m_cpuInputs)
                {
                    copyD(m_cpuInputs, m_cpuInputsPristine);
                }
                else
                {
                    m_cpuInputs = allocNewCPUInputs();
                    copyInputs(m_cpuInputs, m_cpuInputsPristine);
                }

                if (m_convolutionVsContraction and !m_cpuConvInputs) {
                  m_cpuConvInputs = allocNewCPUInputs();
                  copyInputs(m_cpuConvInputs, m_cpuInputsPristine);
                }

                return m_cpuInputs;
            }

            virtual std::shared_ptr<ContractionInputs> cpuConvInputs() const {
              return m_cpuConvInputs;
            };

            std::shared_ptr<ManagedInputs> prepareGPUInputsTyped()
            {
                std::shared_ptr<ManagedInputs> pristine;

                if(m_keepPristineCopyOnGPU)
                {
                    if(!m_gpuInputsPristine)
                        m_gpuInputsPristine = createNewGPUInputs();

                    pristine = m_gpuInputsPristine;
                }
                else
                {
                    if(!m_cpuInputsPristine)
                        m_cpuInputsPristine = createNewCPUInputs();

                    pristine = m_cpuInputsPristine;
                }

                if(m_gpuInputs)
                {
                    copyD(m_gpuInputs, pristine);
                }
                else
                {
                    m_gpuInputs = allocNewGPUInputs(pristine);
                    copyInputs(m_gpuInputs, pristine);
                }

                return m_gpuInputs;
            }

            std::shared_ptr<ManagedInputs> createNewCPUInputs()
            {
                auto rv = allocNewCPUInputs();
                initializeCPUInputs(*rv);

                return rv;
            }

            std::shared_ptr<ManagedInputs> createNewGPUInputs()
            {
                auto rv = allocNewGPUInputs();
                std::shared_ptr<ManagedInputs> source;
                if(!m_cpuInputsPristine)
                    m_cpuInputsPristine = createNewCPUInputs();
                copyInputs(rv, m_cpuInputsPristine);

                return rv;
            }

            std::shared_ptr<ManagedInputs> allocNewCPUInputs(std::shared_ptr<ManagedInputs> pristine = nullptr)
            {
                std::shared_ptr<AType> a;
                std::shared_ptr<BType> b;
                std::shared_ptr<CType> c;
                std::shared_ptr<DType> d;

                if(pristine)
                {
                    a = pristine->managedA;
                    b = pristine->managedB;
                }
                else
                {
                    a = std::shared_ptr<AType>((AType *)std::malloc(TypeInfo<AType>::ElementSize * m_aMaxElements), std::free);
                    if (a==nullptr)
                        throw std::runtime_error("out of host memory allocating a");
                    b = std::shared_ptr<BType>((BType *)std::malloc(TypeInfo<BType>::ElementSize * m_bMaxElements), std::free);
                    if (a==nullptr)
                        throw std::runtime_error("out of host memory allocating b");
                }

                if(m_cEqualsD || !pristine)
                {
                    c = std::shared_ptr<CType>((CType *)std::malloc(TypeInfo<CType>::ElementSize * m_cMaxElements), std::free);
                    if (c==nullptr)
                        throw std::runtime_error("out of host memory allocating c");
                }
                else
                {
                    c = pristine->managedC;
                }

                if(m_cEqualsD)
                {
                    d = c;
                }
                else if(pristine)
                {
                    d = pristine->managedD;
                }
                else
                {
                    d = std::shared_ptr<DType>((DType *)std::malloc(TypeInfo<DType>::ElementSize * m_dMaxElements), std::free);
                    if (d==nullptr)
                        throw std::runtime_error("out of host memory allocating d");
                }

                auto alpha = static_cast<AlphaType>(0);
                auto beta  = static_cast<BetaType>(0);

                auto rv = std::make_shared<ManagedInputs>(a, b, c, d, alpha, beta, false);

                return rv;
            }

            std::shared_ptr<ManagedInputs> allocNewGPUInputs(std::shared_ptr<ManagedInputs> pristine = nullptr)
            {
                if(pristine && !pristine->gpu)
                    pristine = nullptr;

                std::shared_ptr<AType> a;
                std::shared_ptr<BType> b;
                std::shared_ptr<CType> c;
                std::shared_ptr<DType> d;

                if(pristine)
                {
                    a = pristine->managedA;
                    b = pristine->managedB;
                }
                else
                {
                    AType * aPtr = nullptr;
                    HIP_CHECK_EXC(hipMalloc(&aPtr, TypeInfo<AType>::ElementSize * m_aMaxElements));
                    a = std::shared_ptr<AType>(aPtr, hipFree);

                    BType * bPtr = nullptr;
                    HIP_CHECK_EXC(hipMalloc(&bPtr, TypeInfo<BType>::ElementSize * m_bMaxElements));
                    b = std::shared_ptr<BType>(bPtr, hipFree);
                }

                if(m_cEqualsD || !pristine)
                {
                    CType * cPtr = nullptr;
                    HIP_CHECK_EXC(hipMalloc(&cPtr, TypeInfo<CType>::ElementSize * m_cMaxElements));
                    c = std::shared_ptr<CType>(cPtr, hipFree);
                }
                else
                {
                    c = pristine->managedC;
                }

                if(m_cEqualsD)
                {
                    d = c;
                }
                else if(pristine)
                {
                    d = pristine->managedD;
                }
                else
                {
                    DType * dPtr = nullptr;
                    HIP_CHECK_EXC(hipMalloc(&dPtr, TypeInfo<DType>::ElementSize * m_dMaxElements));
                    d = std::shared_ptr<DType>(dPtr, hipFree);
                }

                auto alpha = static_cast<AlphaType>(0);
                auto beta  = static_cast<BetaType>(0);

                auto rv = std::make_shared<ManagedInputs>(a, b, c, d, alpha, beta, true);
                return rv;
            }

            void initializeCPUInputs(ManagedInputs & inputs)
            {
                if(inputs.gpu)
                    throw std::runtime_error("Initializing GPU inputs as CPU.");

                initArray(m_aInit, inputs.managedA.get(), m_aMaxElements);
                initArray(m_bInit, inputs.managedB.get(), m_bMaxElements);
                initArray(m_cInit, inputs.managedC.get(), m_cMaxElements);
                if(!m_cEqualsD)
                    initArray(m_dInit, inputs.managedD.get(), m_dMaxElements);

                inputs.alpha = getValue<AlphaType>(m_alphaInit);
                inputs.beta = getValue<BetaType>(m_betaInit);
            }

            hipMemcpyKind copyKind(std::shared_ptr<ManagedInputs> dst, std::shared_ptr<ManagedInputs> src)
            {
                if(src->gpu)
                {
                    if(dst->gpu)
                    {
                        return hipMemcpyDeviceToDevice;
                    }
                    else
                    {
                        return hipMemcpyDeviceToHost;
                    }
                }
                else
                {
                    if(dst->gpu)
                    {
                        return hipMemcpyHostToDevice;
                    }
                    else
                    {
                        return hipMemcpyHostToHost;
                    }
                }
            }

            void copyInputs(std::shared_ptr<ManagedInputs> dst, std::shared_ptr<ManagedInputs> src)
            {
                hipMemcpyKind kind = copyKind(dst, src);

                if(dst->managedA != src->managedA)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedA.get(), src->managedA.get(), TypeInfo<AType>::ElementSize * m_aMaxElements, kind));

                if(dst->managedB != src->managedB)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedB.get(), src->managedB.get(), TypeInfo<BType>::ElementSize * m_bMaxElements, kind));

                if(dst->managedC != src->managedC)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedC.get(), src->managedC.get(), TypeInfo<CType>::ElementSize * m_cMaxElements, kind));

                if(!m_cEqualsD && dst->managedD != src->managedD)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedD.get(), src->managedD.get(), TypeInfo<DType>::ElementSize * m_dMaxElements, kind));

                //HIP_CHECK_EXC(hipDeviceSynchronize());

                dst->alpha = src->alpha;
                dst->beta = src->beta;
            }

            void copyD(std::shared_ptr<ManagedInputs> dst, std::shared_ptr<ManagedInputs> src)
            {
                hipMemcpyKind kind = copyKind(dst, src);

                HIP_CHECK_EXC(hipMemcpy(dst->managedD.get(), src->managedD.get(), TypeInfo<DType>::ElementSize * m_dMaxElements, kind));
            }

        private:

            std::shared_ptr<ManagedInputs> m_cpuConvInputs;
            std::shared_ptr<ManagedInputs> m_cpuInputs, m_cpuInputsPristine;
            std::shared_ptr<ManagedInputs> m_gpuInputs, m_gpuInputsPristine;

        };
    }
}

