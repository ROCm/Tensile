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

            ManagedContractionInputs(std::shared_ptr<A> _a,
                                     std::shared_ptr<B> _b,
                                     std::shared_ptr<C> _c,
                                     std::shared_ptr<D> _d,
                                     size_t _aElements,
                                     size_t _bElements,
                                     size_t _cElements,
                                     size_t _dElements,
                                     Alpha _alpha, Beta _beta, bool _gpu)
                : Base(_a.get(), _b.get(), _c.get(), _d.get(), _alpha, _beta),
                  managedA(_a),
                  managedB(_b),
                  managedC(_c),
                  managedD(_d),
                  aElements(_aElements),
                  bElements(_bElements),
                  cElements(_cElements),
                  dElements(_dElements),
                  gpu(_gpu)
            {
            }

            ~ManagedContractionInputs() = default;

            std::shared_ptr<A> managedA;
            std::shared_ptr<B> managedB;
            std::shared_ptr<C> managedC;
            std::shared_ptr<D> managedD;

            size_t aElements;
            size_t bElements;
            size_t cElements;
            size_t dElements;

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

            virtual std::shared_ptr<ContractionInputs> prepareCPUInputs(ContractionProblem const& problem)
            {
                return prepareCPUInputsTyped(problem);
            }

            virtual std::shared_ptr<ContractionInputs> prepareGPUInputs(ContractionProblem const& problem)
            {
                return prepareGPUInputsTyped(problem);
            }

            std::shared_ptr<ManagedInputs> prepareCPUInputsTyped(ContractionProblem const& problem)
            {
                if(!m_cpuInputsPristine)
                    m_cpuInputsPristine = createNewCPUInputs();

                if(m_cpuInputs && !m_boundsCheck)
                {
                    copyD(m_cpuInputs, m_cpuInputsPristine);
                }
                else
                {
                    if(!m_cpuInputs)
                        m_cpuInputs = allocNewCPUInputs();

                    if(m_boundsCheck && !m_cpuBadInputs)
                    {
                        m_cpuBadInputs = createNewCPUBadInputs();
                    }

                    copyInputs(m_cpuInputs, m_cpuInputsPristine, m_cpuBadInputs, problem);
                }

                if (m_convolutionVsContraction)
                {
                    bool allocated = false;
                    if(!m_cpuConvInputs)
                    {
                        allocated = true;
                        m_cpuConvInputs = allocNewCPUInputs();
                    }

                    if(allocated || m_boundsCheck)
                        copyInputs(m_cpuConvInputs, m_cpuInputsPristine, m_cpuBadInputs, problem);
                }

                return m_cpuInputs;
            }

            virtual std::shared_ptr<ContractionInputs> cpuConvInputs() const {
              return m_cpuConvInputs;
            };

            std::shared_ptr<ManagedInputs> prepareGPUInputsTyped(ContractionProblem const& problem)
            {
                std::shared_ptr<ManagedInputs> pristine;
                std::shared_ptr<ManagedInputs> bad;

                if(m_keepPristineCopyOnGPU)
                {
                    if(!m_gpuInputsPristine)
                        m_gpuInputsPristine = createNewGPUInputs();

                    pristine = m_gpuInputsPristine;

                    if(m_boundsCheck)
                    {
                        if(!m_gpuBadInputs)
                            m_gpuBadInputs = createNewGPUBadInputs();

                        bad = m_gpuBadInputs;
                    }
                }
                else
                {
                    if(!m_cpuInputsPristine)
                        m_cpuInputsPristine = createNewCPUInputs();

                    pristine = m_cpuInputsPristine;

                    if(m_boundsCheck)
                    {
                        if(!m_cpuBadInputs)
                            m_cpuBadInputs = createNewCPUBadInputs();

                        bad = m_cpuBadInputs;
                    }
                }

                if(m_gpuInputs && !m_boundsCheck)
                {
                    copyD(m_gpuInputs, pristine);
                }
                else
                {
                    if(!m_gpuInputs)
                        m_gpuInputs = allocNewGPUInputs(pristine);

                    copyInputs(m_gpuInputs, pristine, bad, problem);
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
                if(!m_cpuInputsPristine)
                    m_cpuInputsPristine = createNewCPUInputs();
                copyInputBuffers(rv, m_cpuInputsPristine);

                return rv;
            }

            std::shared_ptr<ManagedInputs> createNewCPUBadInputs()
            {
                auto rv = allocNewCPUInputs();
                initializeCPUBadInputs(*rv);

                return rv;
            }

            std::shared_ptr<ManagedInputs> createNewGPUBadInputs()
            {
                auto rv = allocNewGPUInputs();
                if(!m_cpuBadInputs)
                    m_cpuBadInputs = createNewCPUBadInputs();
                copyInputBuffers(rv, m_cpuBadInputs);

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

                auto rv = std::make_shared<ManagedInputs>(a, b, c, d,
                                                          m_aMaxElements,
                                                          m_bMaxElements,
                                                          m_cMaxElements,
                                                          m_dMaxElements,
                                                          alpha, beta, false);

                return rv;
            }

            std::shared_ptr<ManagedInputs> allocNewGPUInputs(std::shared_ptr<ManagedInputs> pristine = nullptr)
            {
                if(m_boundsCheck || (pristine && !pristine->gpu))
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

                auto rv = std::make_shared<ManagedInputs>(a, b, c, d,
                                                          m_aMaxElements,
                                                          m_bMaxElements,
                                                          m_cMaxElements,
                                                          m_dMaxElements,
                                                          alpha, beta, true);
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

            void initializeCPUBadInputs(ManagedInputs & inputs)
            {
                if(inputs.gpu)
                    throw std::runtime_error("Initializing GPU inputs as CPU.");

                initArray(InitMode::BadInput, inputs.managedA.get(), m_aMaxElements);
                initArray(InitMode::BadInput, inputs.managedB.get(), m_bMaxElements);

                initArray(InitMode::BadOutput, inputs.managedD.get(), m_dMaxElements);
                
                if(!m_cEqualsD)
                    initArray(InitMode::BadInput, inputs.managedC.get(), m_cMaxElements);

                inputs.alpha = getValue<AlphaType>(m_alphaInit);
                inputs.beta = getValue<BetaType>(m_betaInit);
            }

            hipMemcpyKind getCopyKind(std::shared_ptr<ManagedInputs> dst, std::shared_ptr<ManagedInputs> src)
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

            void copyInputBuffers(std::shared_ptr<ManagedInputs> dst,
                                  std::shared_ptr<ManagedInputs> src)
            {
                hipMemcpyKind kind = getCopyKind(dst, src);

                if(dst->managedA != src->managedA)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedA.get(), src->managedA.get(), TypeInfo<AType>::ElementSize * m_aMaxElements, kind));

                if(dst->managedB != src->managedB)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedB.get(), src->managedB.get(), TypeInfo<BType>::ElementSize * m_bMaxElements, kind));

                if(dst->managedC != src->managedC)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedC.get(), src->managedC.get(), TypeInfo<CType>::ElementSize * m_cMaxElements, kind));

                if(!m_cEqualsD && dst->managedD != src->managedD)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedD.get(), src->managedD.get(), TypeInfo<DType>::ElementSize * m_dMaxElements, kind));

                dst->alpha = src->alpha;
                dst->beta = src->beta;
            }

            void copyInputs(std::shared_ptr<ManagedInputs> dst,
                            std::shared_ptr<ManagedInputs> src,
                            std::shared_ptr<ManagedInputs> bad,
                            ContractionProblem const& problem)
            {
                hipMemcpyKind kind = getCopyKind(dst, src);

                if(m_boundsCheck)
                {
                    if(!bad)
                        throw std::runtime_error("bad inputs must be initialized for bounds check!");
                    if(bad->gpu != src->gpu)
                        throw std::runtime_error("bad inputs must be in the same location as the source");
                    if(dst->managedA == src->managedA)
                        throw std::runtime_error("A pointers are equal for bounds check!");
                    if(dst->managedB == src->managedB)
                        throw std::runtime_error("B pointers are equal for bounds check!");
                    if(dst->managedC == src->managedC)
                        throw std::runtime_error("C pointers are equal for bounds check!");
                    if(dst->managedD == src->managedD)
                        throw std::runtime_error("D pointers are equal for bounds check!");

                    copyInputBuffers(dst, bad);

                    {
                        ptrdiff_t aPadding = dst->aElements - problem.a().totalAllocatedElements();
                        dst->a = dst->managedA.get() + aPadding/2;
                    }

                    {
                        ptrdiff_t bPadding = dst->bElements - problem.b().totalAllocatedElements();
                        dst->b = dst->managedB.get() + bPadding/2;
                    }

                    {
                        ptrdiff_t cPadding = dst->cElements - problem.c().totalAllocatedElements();
                        dst->c = dst->managedC.get() + cPadding/2;
                    }

                    {
                        ptrdiff_t dPadding = dst->dElements - problem.d().totalAllocatedElements();
                        dst->d = dst->managedD.get() + dPadding/2;
                    }

                    Tensile::hip::CopyTensor(const_cast<AType *>(dst->a), src->a, problem.a(), kind);
                    Tensile::hip::CopyTensor(const_cast<BType *>(dst->b), src->b, problem.b(), kind);

                    Tensile::hip::CopyTensor(const_cast<CType *>(dst->c), src->c, problem.c(), kind);

                    if(!m_cEqualsD)
                        Tensile::hip::CopyTensor(dst->d, src->d, problem.d(), kind);

                    dst->alpha = src->alpha;
                    dst->beta  = src->beta;
                }
                else
                {
                    copyInputBuffers(dst, src);
                }

            }

            void copyD(std::shared_ptr<ManagedInputs> dst, std::shared_ptr<ManagedInputs> src)
            {
                hipMemcpyKind kind = getCopyKind(dst, src);

                HIP_CHECK_EXC(hipMemcpy(dst->managedD.get(), src->managedD.get(), TypeInfo<DType>::ElementSize * m_dMaxElements, kind));
            }

        private:

            std::shared_ptr<ManagedInputs> m_cpuConvInputs;
            std::shared_ptr<ManagedInputs> m_cpuInputs, m_cpuInputsPristine;
            std::shared_ptr<ManagedInputs> m_cpuBadInputs;
            std::shared_ptr<ManagedInputs> m_gpuInputs, m_gpuInputsPristine;
            std::shared_ptr<ManagedInputs> m_gpuBadInputs;

        };

        using ManagedBFloat16ContractionInputs =
        ManagedContractionInputs<BFloat16, BFloat16, BFloat16, BFloat16, float, float>;
    }
}

