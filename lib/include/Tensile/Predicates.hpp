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

#include <algorithm>
#include <string>
#include <vector>

namespace Tensile
{
    namespace Predicates
    {
        template <typename Object>
        class Predicate
        {
        public:
            virtual ~Predicate() = default;
            virtual std::string type() const = 0;
            virtual bool operator()(Object const&) const = 0;
        };

        template <typename Object>
        class True: public Predicate<Object>
        {
        public:
            enum { HasIndex = false, HasValue = false };
            static std::string Type() { return "TruePred"; }
            virtual std::string type() const { return Type(); } 

            virtual bool operator()(Object const&) const
            {
                return true;
            }
        };

        template <typename Object>
        class False: public Predicate<Object>
        {
        public:
            enum { HasIndex = false, HasValue = false };
            static std::string Type() { return "FalsePred"; }
            virtual std::string type() const { return Type(); } 

            virtual bool operator()(Object const&) const
            {
                return false;
            }
        };

        template <typename Object>
        class And: public Predicate<Object>
        {
        public:
            enum { HasIndex = false, HasValue = true };
            std::vector<std::shared_ptr<Predicate<Object>>> value;

            And() = default;
            And(std::initializer_list<std::shared_ptr<Predicate<Object>>> init) : value(init) { }
            And(std::vector<std::shared_ptr<Predicate<Object>>>    const& init) : value(init) { }

            static std::string Type() { return "And"; }
            virtual std::string type() const { return Type(); } 

            virtual bool operator()(Object const& obj) const
            {
                return std::all_of(value.begin(), value.end(),
                        [&obj](std::shared_ptr<Predicate<Object>> pred) { return (*pred)(obj); });
            }
        };

        template <typename Object>
        class Or: public Predicate<Object>
        {
        public:
            enum { HasIndex = false, HasValue = true };
            std::vector<std::shared_ptr<Predicate<Object>>> value;

            Or() = default;
            Or(std::initializer_list<std::shared_ptr<Predicate<Object>>> init) : value(init) { }
            Or(std::vector<std::shared_ptr<Predicate<Object>>>    const& init) : value(init) { }

            static std::string Type() { return "Or"; }
            virtual std::string type() const { return Type(); } 

            virtual bool operator()(Object const& obj) const
            {
                return std::any_of(value.begin(), value.end(),
                        [&obj](std::shared_ptr<Predicate<Object>> pred) { return (*pred)(obj); });
            }
        };

        template <typename Object>
        class Not: public Predicate<Object>
        {
        public:
            enum { HasIndex = false, HasValue = true };
            std::shared_ptr<Predicate<Object>> value;
            Not() = default;
            Not(std::shared_ptr<Predicate<Object>> init) : value(init) { }

            static std::string Type() { return "Not"; }
            virtual std::string type() const { return Type(); } 

            virtual bool operator()(Object const& obj) const
            {
                return !(*value)(obj);
            }
        };

        template <typename Object, typename Subclass>
        class IsSubclass: public Predicate<Object>
        {
        public:
            enum { HasIndex = false, HasValue = true };
            static_assert(std::is_base_of<Object, Subclass>::value, "Subclass must be derived from Object.");

            std::shared_ptr<Predicate<Subclass>> value;

            IsSubclass() = default;
            IsSubclass(std::shared_ptr<Predicate<Subclass>> init) : value(init) { }

            static std::string Type() { return Subclass::Type(); }
            virtual std::string type() const { return Type(); }

            virtual bool operator()(Object const& obj) const
            {
                auto const* sc = dynamic_cast<Subclass const*>(&obj);
                if(!sc) return false;

                return (*value)(*sc);
            }
        };
    }
}

