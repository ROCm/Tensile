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

#include <Tensile/Predicates.hpp>
#include <Tensile/GEMMProblem.hpp>

#include <vector>

namespace Tensile
{
    namespace Predicates
    {
        namespace GEMM
        {
            using namespace Tensile::Predicates;

            struct ADimensionOrder: public Predicate<GEMMProblem>
            {
                enum { HasValue = true };
                std::vector<size_t> value;

                ADimensionOrder() = default;
                template <typename T>
                ADimensionOrder(std::initializer_list<size_t> init)
                    : value(init)
                {
                }
                ADimensionOrder(std::vector<size_t> const& init)
                    : value(init)
                {
                }

                static std::string Key() { return "ADimensionOrder"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return value == problem.a.dimensionOrder();
                }
            };

            struct BDimensionOrder: public Predicate<GEMMProblem>
            {
                enum { HasValue = true };
                std::vector<size_t> value;

                BDimensionOrder() = default;
                BDimensionOrder(std::initializer_list<size_t> init)
                    : value(init)
                {
                }
                BDimensionOrder(std::vector<size_t> const& init)
                    : value(init)
                {
                }

                static std::string Key() { return "BDimensionOrder"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return value == problem.b.dimensionOrder();
                }
            };

            struct CDimensionOrder: public Predicate<GEMMProblem>
            {
                enum { HasValue = true };
                std::vector<size_t> value;

                CDimensionOrder() = default;
                CDimensionOrder(std::initializer_list<size_t> init)
                    : value(init)
                {
                }
                CDimensionOrder(std::vector<size_t> const& init)
                    : value(init)
                {
                }

                static std::string Key() { return "CDimensionOrder"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return value == problem.c.dimensionOrder();
                }
            };

            struct DDimensionOrder: public Predicate<GEMMProblem>
            {
                enum { HasValue = true };
                std::vector<size_t> value;

                DDimensionOrder() = default;
                DDimensionOrder(std::initializer_list<size_t> init)
                    : value(init)
                {
                }
                DDimensionOrder(std::vector<size_t> const& init)
                    : value(init)
                {
                }

                static std::string Key() { return "DDimensionOrder"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return value == problem.d.dimensionOrder();
                }
            };

            struct IDivisible: public Predicate<GEMMProblem>
            {
                enum { HasValue = true };
                size_t value;

                IDivisible() = default;
                IDivisible(size_t init): value(init) {}

                static std::string Key() { return "IDivisible"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_I() % value == 0;
                }
            };

            struct JDivisible: public Predicate<GEMMProblem>
            {
                enum { HasValue = true };
                size_t value;

                JDivisible() = default;
                JDivisible(size_t init): value(init) {}

                static std::string Key() { return "JDivisible"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_J() % value == 0;
                }
            };

            struct KDivisible: public Predicate<GEMMProblem>
            {
                enum { HasValue = true };
                size_t value;

                KDivisible() = default;
                KDivisible(size_t init): value(init) {}

                static std::string Key() { return "KDivisible"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_K() % value == 0;
                }
            };

            struct LDivisible: public Predicate<GEMMProblem>
            {
                enum { HasValue = true };
                size_t value;

                LDivisible() = default;
                LDivisible(size_t init): value(init) {}

                static std::string Key() { return "LDivisible"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_L() % value == 0;
                }
            };

            struct CDStridesEqual: public Predicate<GEMMProblem>
            {
                enum { HasValue = false };
                static std::string Key() { return "CDStridesEqual"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return problem.c.strides() == problem.d.strides();
                }
            };

            struct LDCEqualsLDD: public Predicate<GEMMProblem>
            {
                enum { HasValue = false };
                static std::string Key() { return "LDCEqualsLDD"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_strideC1() == problem.tensile_strideD1();
                }
            };

            struct UseBeta: public Predicate<GEMMProblem>
            {
                enum { HasValue = true };
                bool value;

                UseBeta() = default;
                UseBeta(bool init): value(init) {}

                static std::string Key() { return "UseBeta"; }
                virtual std::string key() const { return Key(); }

                virtual bool operator()(GEMMProblem const& problem) const
                {
                    return problem.useBeta == value;
                }
            };
        }
    }
}

