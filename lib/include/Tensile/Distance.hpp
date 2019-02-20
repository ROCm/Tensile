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

namespace Tensile
{
    struct Distance
    {
        virtual std::string key() const = 0;

        virtual double operator()(std::vector<size_t> const& a, std::vector<size_t> const& b) const;
    };

    struct RatioDistance: public Distance
    {
        static std::string  Key() { return "RatioDistance"; }
        virtual std::string key() { return Key(); }

        double operator() (std::vector<size_t> const& p1, std::vector<size_t> const& p2) const override
        {
            double distance = 1.0;
            for (int i=0; i<p1.size(); i++)
            {
                distance += std::abs(::log(double(p1[i])/double(p2[i])));
            }
          return distance;
        }
    };
    
    struct ManhattanDistance: public Distance
    {
        static std::string  Key() { return "ManhattanDistance"; }
        virtual std::string key() { return Key(); }

        double operator() (std::vector<size_t> const& p1, std::vector<size_t> const& p2) const override
        {
            double distance = 0;
            for (int i=0; i<p1.size(); i++)
            {
                distance += std::abs(double(p1[i]) - double(p2[i]));
        }
        return distance;
      }
    };
    
    
    struct EuclideanDistance: public Distance
    {
        static std::string  Key() { return "EuclideanDistance"; }
        virtual std::string key() { return Key(); }

        double operator() (std::vector<size_t> const& p1, std::vector<size_t> const& p2) const override
        {
            double distance = 0;
            for (int i=0; i<p1.size(); i++)
            {
                distance += std::pow(std::abs(p1[i]-p2[i]),2);
            }
            return distance;
        }
    };
    
    template <class ProblemKeyType>
    struct RandomDistance: public Distance
    {
        static std::string  Key() { return "EuclideanDistance"; }
        virtual std::string key() { return Key(); }

        double operator() (std::vector<size_t> const& p1, std::vector<size_t> const& p2) const override
        {
            return double(rand());
        }
    };

}

