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

#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/DecisionTree.hpp>
#include <Tensile/Properties.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    /**
     * \ingroup SolutionLibrary
     *
     * TODO: documentation
     */
    template <typename Key>
    struct KeyFactory
    {
    };

    template <typename T>
    struct KeyFactory<std::vector<T>>
    {
        static std::vector<T> MakeKey(size_t size)
        {
            return std::vector<T>(size);
        }
    };

    template <typename T, size_t N>
    struct KeyFactory<std::array<T, N>>
    {
        static std::array<T, N> MakeKey(size_t size)
        {
            return std::array<T, N>();
        }
    };

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct DecisionTreeLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using Element    = std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>;
        using Key        = std::array<int64_t, 3>;
        using Tree       = DecisionTree::Tree<Key, Element, std::shared_ptr<MySolution>>;
        using Properties = std::vector<std::shared_ptr<Property<MyProblem>>>;

        Properties        properties;
        std::vector<Tree> trees;

        static std::string Type()
        {
            return "DecisionTree";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            return "TODO: description";
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            typename Tree::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->findBestSolution(problem, hardware);
            };

            Key key = keyForProblem(problem);
            for(Tree const& tree : trees)
            {
                float result = tree.predict(key);
                if(result > 0)
                    return tree.getSolution(transform);
            }
            return std::shared_ptr<MySolution>();
        }

        virtual SolutionSet<MySolution> findAllSolutions(MyProblem const& problem,
                                                         Hardware const&  hardware) const override
        {
            typename Tree::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->findBestSolution(problem, hardware);
            };

            bool debug = Debug::Instance().printPropertyEvaluation();

            SolutionSet<MySolution> rv;

            for(Tree const& tree : trees)
            {
                rv.insert(tree.getSolution(transform));
            }

            return rv;
        }

        Key keyForProblem(MyProblem const& object) const
        {
            bool debug = Debug::Instance().printPropertyEvaluation();

            Key myKey = KeyFactory<Key>::MakeKey(this->properties.size());

            for(int i = 0; i < this->properties.size(); i++)
                myKey[i] = (*this->properties[i])(object);

            if(debug)
            {
                std::cout << "Object key: ";
                streamJoin(std::cout, myKey, ", ");
                std::cout << std::endl;
            }

            return myKey;
        }
    };

} // namespace Tensile
