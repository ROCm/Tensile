/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
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

#include "CSVStackFile.hpp"
#include "ResultReporter.hpp"

#include <boost/program_options.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        namespace po = boost::program_options;

        class ResultFileReporter : public ResultReporter
        {
        public:
            static std::shared_ptr<ResultFileReporter> Default(po::variables_map const& args);

            ResultFileReporter(std::string const& filename);

            virtual void reportValue_string(std::string const& key,
                                            std::string const& value) override;
            virtual void reportValue_uint(std::string const& key, uint64_t value) override;
            virtual void reportValue_int(std::string const& key, int64_t value) override;
            virtual void reportValue_double(std::string const& key, double value) override;
            virtual void reportValue_sizes(std::string const&         key,
                                           std::vector<size_t> const& value) override;

            virtual void postProblem() override;
            virtual void postSolution() override;

            void finalizeReport() override;

        private:
            template <typename T>
            void reportValue(std::string const& key, T const& value);

            CSVStackFile m_output;
            std::string  m_solutionName;
            bool         m_invalidSolution = false;
        };
    } // namespace Client
} // namespace Tensile
