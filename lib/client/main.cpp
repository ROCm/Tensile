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

#include <Tensile/Tensile.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "BenchmarkTimer.hpp"
#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "MetaRunListener.hpp"
#include "ReferenceValidator.hpp"
#include "ResultReporter.hpp"
#include "TimingEvents.hpp"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

namespace po = boost::program_options;

namespace Tensile
{
    namespace Client
    {

        template <typename T>
        po::typed_value<T> * value_default(std::string const& desc)
        {
            return po::value<T>()->default_value(T(), desc);
        }

        template <typename T>
        po::typed_value<T> * value_default()
        {
            return po::value<T>()->default_value(T());
        }

        template <typename T>
        po::typed_value<std::vector<T>> * vector_default_empty()
        {
            return value_default<std::vector<T>>("[]");
        }

        po::options_description all_options()
        {
            po::options_description options("Tensile client options");

            options.add_options()
                ("help,h", "Show help message.")

                ("library-file,l",             po::value<std::string>(), "Load a (YAML) solution library.  If not specified, we will use "
                                                                       "the embedded library, if available.")
                ("code-object,c",            vector_default_empty<std::string>(), "Code object file with kernel(s).  If none are "
                                                                                  "specified, we will use the embedded code "
                                                                                  "object(s) if available.")

                ("problem-identifier",       po::value<std::string>(), "Problem identifer (Einstein notation). Either "
                                                                       "this or free/batch/bound must be specified.")
                ("free",                     value_default<ContractionProblem::FreeIndices>("[]"),  "Free index. Order: a,b,ca,cb,da,db")
                ("batch",                    value_default<ContractionProblem::BatchIndices>("[]"), "Batch index. Order: a,b,c,d")
                ("bound",                    value_default<ContractionProblem::BoundIndices>("[]"), "Bound/summation index. Order: a,b")

                ("type",                     po::value<DataType>()->default_value(DataType::Count), "Data type")
                ("a-type",                   po::value<DataType>()->default_value(DataType::Count), "A data type")
                ("b-type",                   po::value<DataType>()->default_value(DataType::Count), "B data type")
                ("c-type",                   po::value<DataType>()->default_value(DataType::Count), "C data type")
                ("d-type",                   po::value<DataType>()->default_value(DataType::Count), "D data type")
                ("alpha-type",               po::value<DataType>()->default_value(DataType::Count), "alpha data type")
                ("beta-type",                po::value<DataType>()->default_value(DataType::Count), "beta data type")

                ("init-a",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for A")
                ("init-b",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for B")
                ("init-c",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for C")
                ("init-d",                   po::value<InitMode>()->default_value(InitMode::Zero), "Initialization for D")
                ("init-alpha",               po::value<InitMode>()->default_value(InitMode::Two), "Initialization for alpha")
                ("init-beta",                po::value<InitMode>()->default_value(InitMode::Two), "Initialization for beta")
                ("c-equal-d",                po::value<bool>()->default_value(false), "C equals D")

                ("print-valids",             po::value<bool>()->default_value(false), "Print values that pass validation")
                ("print-max",                po::value<int>()->default_value(-1), "Max number of values to print")
                ("num-elements-to-validate", po::value<int>()->default_value(0), "Number of elements to validate")

                ("print-tensor-a",           po::value<bool>()->default_value(false), "Print tensor A.")
                ("print-tensor-b",           po::value<bool>()->default_value(false), "Print tensor B.")
                ("print-tensor-c",           po::value<bool>()->default_value(false), "Print tensor C.")
                ("print-tensor-d",           po::value<bool>()->default_value(false), "Print tensor D.")

                ("device-idx",               po::value<int>()->default_value(0), "Device index")
                ("use-default-stream",       po::value<bool>()->default_value(false), "Use default Hip stream to run kernels.")
                ("platform-idx",             po::value<int>()->default_value(0), "OpenCL Platform Index")

                ("num-warmups",              po::value<int>()->default_value(1), "Number of benchmarks to run") 
                ("num-benchmarks",           po::value<int>()->default_value(1), "Number of benchmarks to run") 
                ("num-enqueues-per-sync",    po::value<int>()->default_value(1), "Enqueues per sync")
                ("num-syncs-per-benchmark",  po::value<int>()->default_value(1), "Syncs per benchmark")
                ("use-gpu-timer",            po::value<bool>()->default_value(true), "Use GPU timer")
                ("sleep-percent",            po::value<int>()->default_value(0), "Sleep percentage")

                ("problem-size,p",           vector_default_empty<std::string>(), "Specify a problem size.  Comma-separated list of "
                                                                                  "sizes, in the order of the Einstein notation.")

                ("a-strides",                vector_default_empty<std::string>(), "A strides. Unspecified means no padding, "
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("b-strides",                vector_default_empty<std::string>(), "B strides. Unspecified means no padding, "
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("c-strides",                vector_default_empty<std::string>(), "C strides. Unspecified means no padding, "
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("d-strides",                vector_default_empty<std::string>(), "D strides. Unspecified means no padding, "
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("solution-start-idx",       po::value<int>()->default_value(0),  "First solution to run")
                ("num-solutions",            po::value<int>()->default_value(-1), "Number of solutions to run");

            return options;
        }

        std::shared_ptr<Hardware> GetHardware(po::variables_map const& args)
        {
            HIP_CHECK_EXC(hipSetDevice(args["device-idx"].as<int>()));

            return hip::GetCurrentDevice();
        }

        hipStream_t GetStream(po::variables_map const& args)
        {
            if(args["use-default-stream"].as<bool>())
                return 0;

            hipStream_t stream;
            HIP_CHECK_EXC(hipStreamCreate(&stream));
            return stream;
        }

        std::shared_ptr<SolutionLibrary<ContractionProblem>>
        LoadSolutionLibrary(po::variables_map const& args)
        {
            auto filename = args["library-file"];
            if(!filename.empty())
            {
                return LoadLibraryFile<ContractionProblem>(filename.as<std::string>());
            }

            auto embeddedLibrary = EmbeddedLibrary<ContractionProblem>::Get();

            if(embeddedLibrary != nullptr)
                return embeddedLibrary;

            throw std::runtime_error("Client must be linked with an embedded library or a library must be specified at runtime.");

        }

        void LoadCodeObjects(po::variables_map const& args, hip::SolutionAdapter & adapter)
        {
            auto const& filenames = args["code-object"].as<std::vector<std::string>>();

            if(filenames.empty())
            {
                adapter.loadEmbeddedCodeObjects();
            }
            else
            {
                for(auto const& filename: filenames)
                {
                    std::cout << "Loading " << filename << std::endl;
                    adapter.loadCodeObjectFile(filename);
                }
            }
        }

        std::vector<size_t> split_ints(std::string const& value)
        {
            std::vector<std::string> parts;
            boost::split(parts, value, boost::algorithm::is_any_of(","));

            std::vector<size_t> rv;
            rv.reserve(parts.size());

            for(auto const& part: parts)
                rv.push_back(boost::lexical_cast<size_t>(part));

            return rv;
        }

        void parse_arg_ints(po::variables_map & args, std::string const& name)
        {
            //if(args[name].empty())
            //    return;

            auto inValue = args[name].as<std::vector<std::string>>();

            std::vector<std::vector<size_t>> outValue;
            outValue.reserve(inValue.size());
            for(auto const& str: inValue)
                outValue.push_back(split_ints(str));

            boost::any v(outValue);

            args.at(name).value() = v;
        }

        void fix_data_types(po::variables_map & args)
        {
            auto type = args["type"].as<DataType>();

            // These types use the same data type for all inputs/outputs, so we allow using the overarching 'type' parameter.
            if(type == DataType::Float
            || type == DataType::Double
            || type == DataType::ComplexFloat
            || type == DataType::ComplexDouble
            || type == DataType::Int32)
            {
                args.at("a-type").value()     = boost::any(type);
                args.at("b-type").value()     = boost::any(type);
                args.at("c-type").value()     = boost::any(type);
                args.at("d-type").value()     = boost::any(type);
                args.at("alpha-type").value() = boost::any(type);
                args.at("beta-type").value()  = boost::any(type);
            }
        }

        po::variables_map parse_args(int argc, const char * argv[])
        {
            auto options = all_options();

            po::variables_map args;
            po::store(po::parse_command_line(argc, argv, options), args);
            po::notify(args);

            if(args.count("help"))
            {
                std::cout << options << std::endl;
                exit(1);
            }

            fix_data_types(args);

            parse_arg_ints(args, "problem-size");
            parse_arg_ints(args, "a-strides");
            parse_arg_ints(args, "b-strides");
            parse_arg_ints(args, "c-strides");
            parse_arg_ints(args, "d-strides");

            return args;
        }

    }
}

int main(int argc, const char * argv[])
{
    using namespace Tensile;
    using namespace Tensile::Client;

    auto args = parse_args(argc, argv);

    ClientProblemFactory problemFactory(args);

    auto hardware = GetHardware(args);
    hipStream_t stream = GetStream(args);

    auto library = LoadSolutionLibrary(args);
    Tensile::hip::SolutionAdapter adapter;
    LoadCodeObjects(args, adapter);

    auto dataInit = DataInitialization::Get(args, problemFactory);

    MetaRunListener listeners;
    listeners.addListener(std::make_shared<ReferenceValidator>(args, dataInit));
    listeners.addListener(std::make_shared<BenchmarkTimer>(args));

    auto reporter = std::make_shared<MetaResultReporter>();
    reporter->addReporter(
            std::shared_ptr<LogReporter>(
                new LogReporter(LogLevel::Debug,
                                {"operation", "solution", "validation", "time_ns", "gflops"},
                                std::cout)));
    
    listeners.setReporter(reporter);

    //ReferenceValidator validator(args, dataInit);
    //BenchmarkTimer timer(args);

    while(listeners.needMoreBenchmarkRuns())
    {
        listeners.preBenchmarkRun();

        for(auto const& problem: problemFactory.problems())
        {
            //std::cout << "Problem: " << problem.operationDescription() << std::endl;
            //std::cout << "a: " << problem.a() << std::endl;
            //std::cout << "b: " << problem.b() << std::endl;
            //std::cout << "c: " << problem.c() << std::endl;
            //std::cout << "d: " << problem.d() << std::endl;

            listeners.preProblem(problem);
            
            auto solutions = library->findAllSolutions(problem, *hardware);
            for(auto solution: solutions)
            {
                listeners.preSolution(*solution);

                while(listeners.needMoreRunsInSolution())
                {
                    auto inputs = dataInit->prepareGPUInputs();

                    auto kernels = solution->solve(problem, *inputs, *hardware);

                    size_t warmupInvocations = listeners.numWarmupRuns();
                    TimingEvents warmupStartEvents(warmupInvocations, kernels.size());
                    TimingEvents warmupStopEvents(warmupInvocations, kernels.size());

                    for(int i = 0; i < warmupInvocations; i++)
                    {
                        listeners.preWarmup();
                        adapter.launchKernels(kernels, stream, warmupStartEvents[i], warmupStopEvents[i]);
                        listeners.postWarmup();
                    }

                    listeners.validateWarmups(inputs, warmupStartEvents, warmupStopEvents);

                    size_t syncs = listeners.numSyncs();
                    size_t enq   = listeners.numEnqueuesPerSync();

                    for(int i = 0; i < syncs; i++)
                    {
                        listeners.preSyncs();

                        TimingEvents startEvents(enq, kernels.size());
                        TimingEvents  stopEvents(enq, kernels.size());

                        listeners.preEnqueues();

                        for(int j = 0; j < enq; j++)
                        {
                            adapter.launchKernels(kernels, stream, startEvents[j], stopEvents[j]);
                            std::cout << ".";
                            std::cout.flush();
                        }

                        std::cout << std::endl;

                        listeners.postEnqueues();
                        listeners.validateEnqueues(inputs, startEvents, stopEvents);

                        listeners.postSyncs();
                    }
                }

                listeners.postSolution();
            }

            listeners.postProblem();
        }

        listeners.postBenchmarkRun();
    }

    listeners.finalizeReport();

    return listeners.error();
}

