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

#include "RunListener.hpp"

#include <boost/program_options.hpp>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include "DataInitialization.hpp"
#include "RunListener.hpp"

namespace Tensile
{
    namespace Client
    {
        namespace po = boost::program_options;

        class ReferenceValidator: public RunListener
        {
        public:
            ReferenceValidator(po::variables_map const& args, std::shared_ptr<DataInitialization> dataInit);

            virtual void setUpProblem(ContractionProblem const& problem);
            virtual void setUpSolution(ContractionSolution const& solution);

            virtual bool needsMoreRunsInSolution();
            virtual bool isWarmupRun();

            virtual void setUpRun(bool isWarmup);
            virtual void tearDownRun();

            virtual void tearDownSolution();
            virtual void tearDownProblem();

            virtual void validate(std::shared_ptr<ContractionInputs> inputs);
            template <typename TypedInputs>
            void validateTyped(TypedInputs const& reference, TypedInputs const& result);

            template <typename TypedInputs>
            void checkResultsTyped(TypedInputs const& reference, TypedInputs const& result);

            template <typename TypedInputs>
            void printTensorsTyped(TypedInputs const& reference, TypedInputs const& result);

            virtual void report();
            virtual int error();

        private:
            std::shared_ptr<DataInitialization> m_dataInit;
            std::shared_ptr<ContractionInputs> m_referenceInputs;

            std::vector<uint8_t> m_cpuResultBuffer;

            ContractionProblem m_problem;

            bool m_enabled;
            int  m_elementsToValidate;
            bool m_printValids;
            int  m_printMax;

            bool m_printTensorA;
            bool m_printTensorB;
            bool m_printTensorC;
            bool m_printTensorD;

            bool m_validatedSolution = false;
            bool m_errorInSolution = false;
            bool m_error = false;
        };
    }
}

