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

#include <Tensile/ArithmeticUnitTypes.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "BenchmarkTimer.hpp"
#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "HardwareMonitorListener.hpp"
#include "MetaRunListener.hpp"
#include "ProgressListener.hpp"
#include "ReferenceValidator.hpp"
#include "SolutionIterator.hpp"
#include "TimingEvents.hpp"

#include "LogReporter.hpp"
#include "MetaResultReporter.hpp"
#include "PerformanceReporter.hpp"
#include "ResultFileReporter.hpp"
#include "ResultReporter.hpp"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/program_options.hpp>

#include <cstddef>

namespace po = boost::program_options;

namespace Tensile
{
    namespace Client
    {

        template <typename T>
        po::typed_value<T>* value_default(std::string const& desc)
        {
            return po::value<T>()->default_value(T(), desc);
        }

        template <typename T>
        po::typed_value<T>* value_default()
        {
            return po::value<T>()->default_value(T());
        }

        template <typename T>
        po::typed_value<std::vector<T>>* vector_default_empty()
        {
            return value_default<std::vector<T>>("[]");
        }

        po::options_description all_options()
        {
            po::options_description options("Tensile client options");

            // clang-format off
            options.add_options()
                ("help,h", "Show help message.")

                ("config-file",              vector_default_empty<std::string>(), "INI config file(s) to read.")

                ("library-file,l",           po::value<std::string>(), "Load a (YAML) solution library.  If not specified, we will use "
                                                                       "the embedded library, if available.")
                ("code-object,c",            vector_default_empty<std::string>(), "Code object file with kernel(s).  If none are "
                                                                                  "specified, we will use the embedded code "
                                                                                  "object(s) if available.")

                ("problem-identifier",       po::value<std::string>(), "Problem identifer (Einstein notation). Either "
                                                                       "this or free/batch/bound must be specified.")
                ("free",                     value_default<ContractionProblem::FreeIndices>("[]"),  "Free index. Order: a,b,ca,cb,da,db")
                ("batch",                    value_default<ContractionProblem::BatchIndices>("[]"), "Batch index. Order: a,b,c,d")
                ("bound",                    value_default<ContractionProblem::BoundIndices>("[]"), "Bound/summation index. Order: a,b")

                ("type",                     po::value<DataType>()->default_value(DataType::None), "Data type")
                ("a-type",                   po::value<DataType>()->default_value(DataType::None), "A data type")
                ("b-type",                   po::value<DataType>()->default_value(DataType::None), "B data type")
                ("c-type",                   po::value<DataType>()->default_value(DataType::None), "C data type")
                ("d-type",                   po::value<DataType>()->default_value(DataType::None), "D data type")
                ("alpha-type",               po::value<DataType>()->default_value(DataType::None), "alpha data type")
                ("beta-type",                po::value<DataType>()->default_value(DataType::None), "beta data type")
                ("high-precision-accumulate", po::value<bool>()->default_value(false), "Use high-precision accumulate.")
                ("strided-batched",          po::value<bool>()->default_value(true), "Use strided-batched or general batched")
                ("kernel-language",          po::value<KernelLanguage>()->default_value(KernelLanguage::Any), "Select kernel language.")
                ("deterministic-mode",       po::value<bool>()->default_value(false), "Enforce deterministic summation patterns"
                                                                                      "by not splitting U among workgroups")
                ("arithmetic-unit",          po::value<ArithmeticUnit>()->default_value(ArithmeticUnit::Any), "Select arithmetic unit.")

                ("init-a",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for A")
                ("init-b",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for B")
                ("init-c",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for C")
                ("init-d",                   po::value<InitMode>()->default_value(InitMode::Zero), "Initialization for D")
                ("init-alpha",               po::value<InitMode>()->default_value(InitMode::Two), "Initialization for alpha")
                ("init-beta",                po::value<InitMode>()->default_value(InitMode::Two), "Initialization for beta")
                ("pristine-on-gpu",          po::value<bool>()->default_value(false), "Keep a pristine copy of inputs on GPU for performance")
                ("c-equal-d",                po::value<bool>()->default_value(false), "C equals D")
                ("offset-a",                 po::value<size_t>()->default_value(0), "buffer a start offset")
                ("offset-b",                 po::value<size_t>()->default_value(0), "buffer b start offset")
                ("offset-c",                 po::value<size_t>()->default_value(0), "buffer c start offset")
                ("offset-d",                 po::value<size_t>()->default_value(0), "buffer d start offset")
                ("print-valids",             po::value<bool>()->default_value(false), "Print values that pass validation")
                ("print-max",                po::value<int>()->default_value(-1), "Max number of values to print")
                ("num-elements-to-validate", po::value<int>()->default_value(0), "Number of elements to validate")
                ("bounds-check",             po::value<BoundsCheckMode>()->default_value(BoundsCheckMode::Disable),
                "1:Use sentinel values to check memory boundaries."
                "2:Memory bound check by front guard page"
                "3:Memory bound check by back guard page"
                "4:Memory bound check by both side guard page")

                ("print-tensor-a",           po::value<bool>()->default_value(false), "Print tensor A.")
                ("print-tensor-b",           po::value<bool>()->default_value(false), "Print tensor B.")
                ("print-tensor-c",           po::value<bool>()->default_value(false), "Print tensor C.")
                ("print-tensor-d",           po::value<bool>()->default_value(false), "Print tensor D.")
                ("print-tensor-ref",         po::value<bool>()->default_value(false), "Print reference tensor D.")

                ("dump-tensors",             po::value<bool>()->default_value(false), "Binary dump tensors instead of printing.")

                ("convolution-identifier",   po::value<std::string>(), "Convolution problem identifer:  ConvolutionType_ActFormat_FilterFormat_Filter_Stride_Dilation_Groups.  Example: ConvolutionBackwardWeights_NCHW_filter:3x3_stride:1x1_dilation:1x1_groups:1.  Batch count, spacial dimensions (H,W,D), Cin and Cout filters are determined by the problem dimensions.")
                ("convolution-vs-contraction",  po::value<bool>()->default_value(false), "Compare reference convolution against contraction.")

                ("device-idx",               po::value<int>()->default_value(0), "Device index")
                ("use-default-stream",       po::value<bool>()->default_value(false), "Use default Hip stream to run kernels.")
                ("platform-idx",             po::value<int>()->default_value(0), "OpenCL Platform Index")

                ("num-warmups",              po::value<int>()->default_value(0), "Number of warmups to run")
                ("num-benchmarks",           po::value<int>()->default_value(1), "Number of benchmarks to run")
                ("num-enqueues-per-sync",    po::value<int>()->default_value(1), "Enqueues per sync")
                ("num-syncs-per-benchmark",  po::value<int>()->default_value(1), "Syncs per benchmark")
                ("use-gpu-timer",            po::value<bool>()->default_value(true), "Use GPU timer")
                ("sleep-percent",            po::value<int>()->default_value(0), "Sleep percentage")
                ("hardware-monitor",         po::value<bool>()->default_value(true), "Use hardware monitor.")

                ("perf-l2-read-hits",        po::value<double>()->default_value(0.0), "L2 read hits")
                ("perf-l2-write-hits",       po::value<double>()->default_value(0.5), "L2 write hits")
                ("perf-l2-read-bw-mul",      po::value<double>()->default_value(2.0), "L2 read bandwidth multiplier")
                ("perf-read-efficiency",     po::value<double>()->default_value(0.85), "Read efficiency")
                ("perf-ops-per-cycle",       po::value<int>()->default_value(64), "Ops per cycle")
                ("csv-export-extra-cols",    po::value<bool>()->default_value(false), "CSV exports winner information")
                ("csv-merge-same-problems",  po::value<bool>()->default_value(false), "CSV merge rows of same problem id")

                ("problem-size,p",           vector_default_empty<std::string>(), "Specify a problem size.  Comma-separated list of "
                                                                                  "sizes, in the order of the Einstein notation.")

                ("a-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("b-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("c-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("d-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("a-zero-pads",                vector_default_empty<std::string>(), "Comma-separated tuple(s) of anchor dim,"
                                                                                  "summation dim, leading pad, trailing pad."
                                                                                  "Each tuple must be separated with a semi-colon.")

                ("b-zero-pads",                vector_default_empty<std::string>(), "Comma-separated tuple(s) of anchor dim,"
                                                                                  "summation dim, leading pad, trailing pad."
                                                                                  "Each tuple must be separated with a semi-colon.")

                ("a-ops",                    vector_default_empty<TensorOp>(), "Operations applied to A.")
                ("b-ops",                    vector_default_empty<TensorOp>(), "Operations applied to B.")
                ("c-ops",                    vector_default_empty<TensorOp>(), "Operations applied to C.")
                ("d-ops",                    vector_default_empty<TensorOp>(), "Operations applied to D.")

                ("problem-start-idx",        po::value<int>()->default_value(0),  "First problem to run")
                ("num-problems",             po::value<int>()->default_value(-1), "Number of problems to run")

                ("solution-start-idx",       po::value<int>()->default_value(-1), "First solution to run")
                ("num-solutions",            po::value<int>()->default_value(-1), "Number of solutions to run")
                ("best-solution",            po::value<bool>()->default_value(false), "Best solution benchmark mode")

                ("results-file",             po::value<std::string>()->default_value("results.csv"), "File name to write results.")
                ("log-file",                 po::value<std::string>(),                               "File name for output log.")
                ("log-file-append",          po::value<bool>()->default_value(false),                "Append to log file.")
                ("log-level",                po::value<LogLevel>()->default_value(LogLevel::Debug),  "Log level")
                ("exit-on-failure",          po::value<bool>()->default_value(false), "Exit run early on failed kernels.")
                ("selection-only",           po::value<bool>()->default_value(false), "Don't run any solutions, only print kernel selections.")
                ("max-workspace-size",       po::value<size_t>()->default_value(32*1024*1024), "Max workspace for training")
                ("granularity-threshold",    po::value<double>()->default_value(0.0), "Don't run a solution if total granularity is below")
                ;
            // clang-format on

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

        std::shared_ptr<MasterSolutionLibrary<ContractionProblem>>
            LoadSolutionLibrary(po::variables_map const& args)
        {
            auto filename = args["library-file"];
            if(!filename.empty())
            {
                return std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblem>>(
                    LoadLibraryFile<ContractionProblem>(filename.as<std::string>()));
            }

            auto embeddedLibrary
                = std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblem>>(
                    EmbeddedLibrary<ContractionProblem>::Get());

            if(embeddedLibrary != nullptr)
                return embeddedLibrary;

            throw std::runtime_error("Client must be linked with an embedded library or "
                                     "a library must be specified at runtime.");
        }

        void LoadCodeObjects(po::variables_map const& args, hip::SolutionAdapter& adapter)
        {
            auto const& filenames = args["code-object"].as<std::vector<std::string>>();
            auto        logLevel  = args["log-level"].as<LogLevel>();

            if(filenames.empty())
            {
                adapter.loadEmbeddedCodeObjects();
            }
            else
            {
                for(auto const& filename : filenames)
                {
                    if(logLevel >= LogLevel::Verbose)
                        std::cout << "Loading " << filename << std::endl;
                    adapter.loadCodeObjectFile(filename);
                }
            }
        }

        std::vector<size_t> split_ints(std::string const& value)
        {
            std::vector<std::string> parts;
            boost::split(parts, value, boost::algorithm::is_any_of(",;"));

            std::vector<size_t> rv;
            rv.reserve(parts.size());

            for(auto const& part : parts)
                if(part != "")
                    rv.push_back(boost::lexical_cast<size_t>(part));

            return rv;
        }

        void parse_arg_ints(po::variables_map& args, std::string const& name)
        {
            auto inValue = args[name].as<std::vector<std::string>>();

            std::vector<std::vector<size_t>> outValue;
            outValue.reserve(inValue.size());
            for(auto const& str : inValue)
                outValue.push_back(split_ints(str));

            boost::any v(outValue);

            args.at(name).value() = v;
        }

        void fix_data_types(po::variables_map& args)
        {
            auto type = args["type"].as<DataType>();

            // These types use the same data type for all inputs/outputs, so we allow
            // using the overarching 'type' parameter.
            if(type == DataType::Float || type == DataType::Double || type == DataType::ComplexFloat
               || type == DataType::ComplexDouble || type == DataType::Int32)
            {
                args.at("a-type").value()     = boost::any(type);
                args.at("b-type").value()     = boost::any(type);
                args.at("c-type").value()     = boost::any(type);
                args.at("d-type").value()     = boost::any(type);
                args.at("alpha-type").value() = boost::any(type);
                args.at("beta-type").value()  = boost::any(type);
            }
        }

        po::variables_map parse_args(int argc, const char* argv[])
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

            if(args.count("config-file"))
            {
                auto configFiles = args["config-file"].as<std::vector<std::string>>();
                for(auto filename : configFiles)
                {
                    std::ifstream file(filename.c_str());
                    po::store(po::parse_config_file(file, options), args);
                }
            }

            fix_data_types(args);

            parse_arg_ints(args, "problem-size");
            parse_arg_ints(args, "a-strides");
            parse_arg_ints(args, "b-strides");
            parse_arg_ints(args, "c-strides");
            parse_arg_ints(args, "d-strides");
            parse_arg_ints(args, "a-zero-pads");
            parse_arg_ints(args, "b-zero-pads");

            return args;
        }

        size_t getMaxWorkspace(std::shared_ptr<MasterSolutionLibrary<ContractionProblem>>& library,
                               std::shared_ptr<Hardware>&                                  hardware,
                               po::variables_map&                                          args,
                               std::vector<ContractionProblem>&                            problems,
                               int firstProblemIdx,
                               int lastProblemIdx)
        {
            // get max workspace size
            size_t maxWorkspaceSize = 0;

            auto            solutionIterator = SolutionIterator::Default(library, hardware, args);
            MetaRunListener listeners;
            listeners.addListener(solutionIterator);
            auto reporters = std::make_shared<MetaResultReporter>();
            listeners.setReporter(reporters);

            listeners.preBenchmarkRun();

            for(int problemIdx = firstProblemIdx; problemIdx <= lastProblemIdx; problemIdx++)
            {
                auto& problem = problems[problemIdx];

                problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

                listeners.preProblem(problem);

                while(solutionIterator->moreSolutionsInProblem())
                {
                    auto solution = solutionIterator->getSolution();

                    listeners.preSolution(*solution);

                    if(solutionIterator->runCurrentSolution())
                    {
                        maxWorkspaceSize
                            = std::max(maxWorkspaceSize,
                                       solution->requiredWorkspaceSize(problems[problemIdx]));
                    }

                    listeners.postSolution();
                }

                listeners.postProblem();
            }

            listeners.postBenchmarkRun();

            return maxWorkspaceSize;
        }

    } // namespace Client
} // namespace Tensile

int main(int argc, const char* argv[])
{
    using namespace Tensile;
    using namespace Tensile::Client;

    auto args = parse_args(argc, argv);

    ClientProblemFactory problemFactory(args);

    auto        hardware = GetHardware(args);
    hipStream_t stream   = GetStream(args);

    auto                          library = LoadSolutionLibrary(args);
    Tensile::hip::SolutionAdapter adapter;
    LoadCodeObjects(args, adapter);

    auto problems        = problemFactory.problems();
    int  firstProblemIdx = args["problem-start-idx"].as<int>();
    int  numProblems     = args["num-problems"].as<int>();
    if(numProblems < 0)
        numProblems = problems.size();
    int lastProblemIdx = firstProblemIdx + numProblems - 1;

    int firstSolutionIdx = args["solution-start-idx"].as<int>();
    int numSolutions     = args["num-solutions"].as<int>();

    bool gpuTimer = args["use-gpu-timer"].as<bool>();

    bool runKernels = !args["selection-only"].as<bool>();

    if(firstSolutionIdx < 0)
        firstSolutionIdx = library->solutions.begin()->first;

    int lastSolutionIdx;
    if(numSolutions < 0)
    {
        auto iter = library->solutions.end();
        iter--;
        lastSolutionIdx = iter->first;
    }
    else
    {
        lastSolutionIdx = firstSolutionIdx + numSolutions - 1;
    }

    size_t maxWorkspaceSizeLimit = args["max-workspace-size"].as<size_t>();
    size_t maxWorkspaceSize
        = getMaxWorkspace(library, hardware, args, problems, firstProblemIdx, lastProblemIdx);
    maxWorkspaceSize = std::min(maxWorkspaceSize, maxWorkspaceSizeLimit);

    auto dataInit = DataInitialization::Get(args, problemFactory, maxWorkspaceSize);

    auto solutionIterator = SolutionIterator::Default(library, hardware, args);

    MetaRunListener listeners;

    listeners.addListener(dataInit);
    listeners.addListener(solutionIterator);
    listeners.addListener(std::make_shared<ProgressListener>(args));
    if(runKernels)
    {
        listeners.addListener(std::make_shared<ReferenceValidator>(args, dataInit));
        listeners.addListener(std::make_shared<BenchmarkTimer>(args, *hardware));
        listeners.addListener(std::make_shared<HardwareMonitorListener>(args));
    }

    auto reporters = std::make_shared<MetaResultReporter>();
    reporters->addReporter(PerformanceReporter::Default(args));

    // PerformanceReporter needs to be called before these two, or else values
    // will be missing
    reporters->addReporter(LogReporter::Default(args));
    reporters->addReporter(ResultFileReporter::Default(args));

    if(args.count("log-file"))
    {
        std::string filename = args["log-file"].as<std::string>();
        auto        logFile  = std::make_shared<std::ofstream>(
            filename.c_str(), args["log-file-append"].as<bool>() ? std::ios::app : std::ios::out);

        reporters->addReporter(LogReporter::Default(args, logFile));
    }

    listeners.setReporter(reporters);

    // ReferenceValidator validator(args, dataInit);
    // BenchmarkTimer timer(args);

    reporters->report(ResultKey::ProblemCount, problemFactory.problems().size());

    while(listeners.needMoreBenchmarkRuns())
    {
        listeners.preBenchmarkRun();

        for(int problemIdx = firstProblemIdx; problemIdx <= lastProblemIdx; problemIdx++)
        {
            auto& problem = problems[problemIdx];
            problem.setWorkspaceSize(dataInit->workspaceSize());

            reporters->report(ResultKey::ProblemIndex, problemIdx);
            reporters->report(ResultKey::ProblemProgress,
                              concatenate(problemIdx, "/", lastProblemIdx));

            // std::cout << "Problem: " << problem.operationDescription() <<
            // std::endl; std::cout << "a: " << problem.a() << std::endl; std::cout <<
            // "b: " << problem.b() << std::endl; std::cout << "c: " << problem.c() <<
            // std::endl; std::cout << "d: " << problem.d() << std::endl;

            listeners.preProblem(problem);

            while(solutionIterator->moreSolutionsInProblem())
            {
                auto solution = solutionIterator->getSolution();

                listeners.preSolution(*solution);

                if(solutionIterator->runCurrentSolution() && runKernels)
                {
                    try
                    {
                        while(listeners.needMoreRunsInSolution())
                        {
                            auto inputs = dataInit->prepareGPUInputs(problem);

                            auto kernels = solution->solve(problem, *inputs, *hardware);

                            size_t       warmupInvocations = listeners.numWarmupRuns();
                            size_t       eventCount        = kernels.size();
                            TimingEvents warmupStartEvents(warmupInvocations, eventCount);
                            TimingEvents warmupStopEvents(warmupInvocations, eventCount);

                            for(int i = 0; i < warmupInvocations; i++)
                            {
                                listeners.preWarmup();
                                adapter.launchKernels(
                                    kernels, stream, warmupStartEvents[i], warmupStopEvents[i]);
                                listeners.postWarmup();
                            }

                            listeners.validateWarmups(inputs, warmupStartEvents, warmupStopEvents);

                            size_t syncs = listeners.numSyncs();
                            size_t enq   = listeners.numEnqueuesPerSync();

                            for(int i = 0; i < syncs; i++)
                            {
                                listeners.preSyncs();

                                TimingEvents startEvents(enq, eventCount);
                                TimingEvents stopEvents(enq, eventCount);

                                listeners.preEnqueues();

                                for(int j = 0; j < enq; j++)
                                {
                                    if(gpuTimer)
                                        adapter.launchKernels(
                                            kernels, stream, startEvents[j], stopEvents[j]);
                                    else
                                        adapter.launchKernels(kernels, stream, nullptr, nullptr);
                                }

                                listeners.postEnqueues(startEvents, stopEvents);
                                listeners.validateEnqueues(inputs, startEvents, stopEvents);

                                listeners.postSyncs();
                            }
                        }
                    }
                    catch(std::runtime_error const& err)
                    {
                        reporters->report(ResultKey::Validation, "INVALID");
                        reporters->log(LogLevel::Error,
                                       concatenate("Exception occurred: ", err.what(), "\n"));
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
