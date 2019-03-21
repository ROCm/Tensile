/**
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <Tensile/PropertyMatching.hpp>

#include <cmath>

namespace Tensile
{
    namespace Matching
    {
        template <typename Key>
        struct RatioDistance: public Distance<Key>
        {
            enum { HasIndex = false, HasValue = false };
            static std::string  Type() { return "Ratio"; }
            virtual std::string type() const override { return Type(); }

            double operator() (Key const& p1, Key const& p2) const override
            {
                double distance = 1.0;
                for (int i=0; i<p1.size(); i++)
                {
                    distance += std::abs(std::log(double(p1[i])/double(p2[i])));
                }
              return distance;
            }
        };
        
        template <typename Key>
        struct ManhattanDistance: public Distance<Key>
        {
            enum { HasIndex = false, HasValue = false };
            static std::string  Type() { return "Manhattan"; }
            virtual std::string type() const override { return Type(); }

            double operator() (Key const& p1, Key const& p2) const override
            {
                double distance = 0;
                for (int i=0; i<p1.size(); i++)
                {
                    distance += std::abs(double(p1[i]) - double(p2[i]));
            }
            return distance;
          }
        };
        
        
        template <typename Key>
        struct EuclideanDistance: public Distance<Key>
        {
            enum { HasIndex = false, HasValue = false };

            static std::string  Type() { return "Euclidean"; }
            virtual std::string type() const override { return Type(); }

            double operator() (Key const& p1, Key const& p2) const override
            {
                double distance = 0;
                for (int i=0; i<p1.size(); i++)
                {
                    distance += std::pow(double(p1[i])-double(p2[i]),2);
                }
                return distance;
            }
        };
        
        template <typename Key>
        struct RandomDistance: public Distance<Key>
        {
            enum { HasIndex = false, HasValue = false };

            static std::string  Type() { return "Random"; }
            virtual std::string type() const override { return Type(); }

            double operator() (Key const& p1, Key const& p2) const override
            {
                return double(rand());
            }
        };

    }
}

