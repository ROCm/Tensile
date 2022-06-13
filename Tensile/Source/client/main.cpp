/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "LibraryUpdateReporter.hpp"
#include "LogReporter.hpp"
#include "MetaResultReporter.hpp"
#include "PerformanceReporter.hpp"
#include "ResultFileReporter.hpp"
#include "ResultReporter.hpp"

#include "program_options.hpp"

#include <cstddef>

namespace pomain = roc;

namespace Tensile
{
    namespace Client
    {

        template <typename T>
        pomain::value<T>* value_default(std::string const& desc)
        {
            return pomain::value<T>()->default_value(T());
        }

        template <typename T>
        pomain::value<T>* value_default()
        {
            return pomain::value<T>()->default_value(T());
        }

        template <typename T>
        pomain::value<std::vector<T>>* vector_default_empty()
        {
            return value_default<std::vector<T>>("[]");
        }

        pomain::options_description all_options()
        {
            pomain::options_description options("Tensile client options");

            // clang-format off
            options.add_options()
                ("help,h", "Show help message.")

                ("config-file",              pomain::value<std::vector<std::string>>(), "INI config file(s) to read.")

                ("library-file,l",           pomain::value<std::string>(), "Load a (YAML) solution library.  If not specified, we will use "
                                                                       "the embedded library, if available.")
                ("code-object,c",            pomain::value<std::vector<std::string>>(), "Code object file with kernel(s).  If none are "
                                                                                  "specified, we will use the embedded code "
                                                                                  "object(s) if available.")

                ("performance-metric",       pomain::value<PerformanceMetric>()->default_value(PerformanceMetric::DeviceEfficiency), "Metric for benchmarking results")

                ("problem-identifier",       pomain::value<std::string>(), "Problem identifer (Einstein notation). Either "
                                                                       "this or free/batch/bound must be specified.")
                ("free",                     pomain::value<ContractionProblem::FreeIndices>()->default_value(ContractionProblem::FreeIndices(0)),  "Free index. Order: a,b,ca,cb,da,db")
                ("batch",                    pomain::value<ContractionProblem::BatchIndices>()->default_value(ContractionProblem::BatchIndices(0)), "Batch index. Order: a,b,c,d")
                ("bound",                    pomain::value<ContractionProblem::BoundIndices>()->default_value(ContractionProblem::BoundIndices(0)), "Bound/summation index. Order: a,b")
                
                ("type",                     pomain::value<DataType>()->default_value(DataType::None), "Data type")
                ("a-type",                   pomain::value<DataType>()->default_value(DataType::None), "A data type")
                ("b-type",                   pomain::value<DataType>()->default_value(DataType::None), "B data type")
                ("c-type",                   pomain::value<DataType>()->default_value(DataType::None), "C data type")
                ("d-type",                   pomain::value<DataType>()->default_value(DataType::None), "D data type")
                ("alpha-type",               pomain::value<DataType>()->default_value(DataType::None), "alpha data type")
                ("beta-type",                pomain::value<DataType>()->default_value(DataType::None), "beta data type")
                ("high-precision-accumulate", pomain::value<bool>()->default_value(false), "Use high-precision accumulate.")
                ("strided-batched",          pomain::value<bool>()->default_value(true), "Use strided-batched or general batched")
                ("kernel-language",          pomain::value<KernelLanguage>()->default_value(KernelLanguage::Any), "Select kernel language.")
                ("deterministic-mode",       pomain::value<bool>()->default_value(false), "Enforce deterministic summation patterns"
                                                                                      "by not splitting U among workgroups")
                ("arithmetic-unit",          pomain::value<ArithmeticUnit>()->default_value(ArithmeticUnit::Any), "Select arithmetic unit.")

                ("init-a",                   pomain::value<InitMode>()->default_value(InitMode::Random), "Initialization for A")
                ("init-b",                   pomain::value<InitMode>()->default_value(InitMode::Random), "Initialization for B")
                ("init-c",                   pomain::value<InitMode>()->default_value(InitMode::Random), "Initialization for C")
                ("init-d",                   pomain::value<InitMode>()->default_value(InitMode::Zero), "Initialization for D")
                ("init-alpha",               pomain::value<InitMode>()->default_value(InitMode::Two), "Initialization for alpha")
                ("init-beta",                pomain::value<InitMode>()->default_value(InitMode::Two), "Initialization for beta")
                ("pristine-on-gpu",          pomain::value<bool>()->default_value(true), "Keep a pristine copy of inputs on GPU for performance")
                ("c-equal-d",                pomain::value<bool>()->default_value(false), "C equals D")
                ("offset-a",                 pomain::value<size_t>()->default_value(0), "buffer a start offset")
                ("offset-b",                 pomain::value<size_t>()->default_value(0), "buffer b start offset")
                ("offset-c",                 pomain::value<size_t>()->default_value(0), "buffer c start offset")
                ("offset-d",                 pomain::value<size_t>()->default_value(0), "buffer d start offset")
                ("print-valids",             pomain::value<bool>()->default_value(false), "Print values that pass validation")
                ("print-max",                pomain::value<int>()->default_value(-1), "Max number of values to print")
                ("num-elements-to-validate", pomain::value<int>()->default_value(0), "Number of elements to validate")
                ("bounds-check",             pomain::value<BoundsCheckMode>()->default_value(BoundsCheckMode::Disable),
                "1:Use sentinel values to check memory boundaries."
                "2:Memory bound check by front guard page"
                "3:Memory bound check by back guard page"
                "4:Memory bound check by both side guard page")

                ("print-tensor-a",           pomain::value<bool>()->default_value(false), "Print tensor A.")
                ("print-tensor-b",           pomain::value<bool>()->default_value(false), "Print tensor B.")
                ("print-tensor-c",           pomain::value<bool>()->default_value(false), "Print tensor C.")
                ("print-tensor-d",           pomain::value<bool>()->default_value(false), "Print tensor D.")
                ("print-tensor-ref",         pomain::value<bool>()->default_value(false), "Print reference tensor D.")

                ("dump-tensors",             pomain::value<bool>()->default_value(false), "Binary dump tensors instead of printing.")

                ("convolution-identifier",   pomain::value<std::string>(), "Convolution problem identifer:  ConvolutionType_ActFormat_FilterFormat_Filter_Stride_Dilation_Groups.  Example: ConvolutionBackwardWeights_NCHW_filter:3x3_stride:1x1_dilation:1x1_groups:1.  Batch count, spacial dimensions (H,W,D), Cin and Cout filters are determined by the problem dimensions.")
                ("convolution-vs-contraction",  pomain::value<bool>()->default_value(false), "Compare reference convolution against contraction.")

                ("device-idx",               pomain::value<int>()->default_value(0), "Device index")
                ("use-default-stream",       pomain::value<bool>()->default_value(false), "Use default Hip stream to run kernels.")
                ("platform-idx",             pomain::value<int>()->default_value(0), "OpenCL Platform Index")

                ("num-warmups",              pomain::value<int>()->default_value(0), "Number of warmups to run")
                ("sync-after-warmups",       pomain::value<bool>()->default_value(true), "Synchronize GPU after warmup kernel runs")
                ("num-benchmarks",           pomain::value<int>()->default_value(1), "Number of benchmarks to run")
                ("num-enqueues-per-sync",    pomain::value<int>()->default_value(1), "Enqueues per sync, will affect by min-flops-per-sync")
                ("num-syncs-per-benchmark",  pomain::value<int>()->default_value(1), "Syncs per benchmark")
                ("min-flops-per-sync",       pomain::value<size_t>()->default_value(0), "Minimum number of flops per sync to increase stability for small problems.")
                ("use-gpu-timer",            pomain::value<bool>()->default_value(true), "Use GPU timer")
                ("sleep-percent",            pomain::value<int>()->default_value(0), "Sleep percentage")
                ("hardware-monitor",         pomain::value<bool>()->default_value(true), "Use hardware monitor.")

                ("perf-l2-read-hits",        pomain::value<double>()->default_value(0.0), "L2 read hits")
                ("perf-l2-write-hits",       pomain::value<double>()->default_value(0.5), "L2 write hits")
                ("perf-l2-read-bw-mul",      pomain::value<double>()->default_value(2.0), "L2 read bandwidth multiplier")
                ("perf-read-efficiency",     pomain::value<double>()->default_value(0.85), "Read efficiency")
                ("perf-ops-per-cycle",       pomain::value<int>()->default_value(64), "Ops per cycle")
                ("csv-export-extra-cols",    pomain::value<bool>()->default_value(false), "CSV exports winner information")
                ("csv-merge-same-problems",  pomain::value<bool>()->default_value(false), "CSV merge rows of same problem id")

                ("problem-size,p",           pomain::value<std::vector<std::vector<size_t>>>()->default_value(std::vector<std::vector<size_t>>(0)), "Specify a problem size.  Comma-separated list of "
                                                                                  "sizes, in the order of the Einstein notation.")

                ("a-strides",                pomain::value<std::vector<std::vector<size_t>>>()->default_value(std::vector<std::vector<size_t>>(0)), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("b-strides",                pomain::value<std::vector<std::vector<size_t>>>()->default_value(std::vector<std::vector<size_t>>(0)), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("c-strides",                pomain::value<std::vector<std::vector<size_t>>>()->default_value(std::vector<std::vector<size_t>>(0)), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("d-strides",                pomain::value<std::vector<std::vector<size_t>>>()->default_value(std::vector<std::vector<size_t>>(0)), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("convolution-problem",      pomain::value<std::vector<std::vector<size_t>>>()->default_value(std::vector<std::vector<size_t>>(0)), "Specify a Convolution problem size. Comma-separated list of sizes:"
                                                                                  "Spatial(w,h,d),Filter(x,y,z),Stride(v,u,#),"
                                                                                  "Dilation(j,l,^),Pad start(q,p,$),Pad end(q_,p_,$_)")

                ("a-zero-pads",                pomain::value<std::vector<std::vector<size_t>>>()->default_value(std::vector<std::vector<size_t>>(0)), "Comma-separated tuple(s) of anchor dim,"
                                                                                  "summation dim, leading pad, trailing pad."
                                                                                  "Each tuple must be separated with a semi-colon.")

                ("b-zero-pads",                pomain::value<std::vector<std::vector<size_t>>>()->default_value(std::vector<std::vector<size_t>>(0)), "Comma-separated tuple(s) of anchor dim,"
                                                                                  "summation dim, leading pad, trailing pad."
                                                                                  "Each tuple must be separated with a semi-colon.")

                ("a-ops",                    pomain::value<std::vector<TensorOp>>()->default_value(std::vector<TensorOp>(0)), "Operations applied to A.")
                ("b-ops",                    pomain::value<std::vector<TensorOp>>()->default_value(std::vector<TensorOp>(0)), "Operations applied to B.")
                ("c-ops",                    pomain::value<std::vector<TensorOp>>()->default_value(std::vector<TensorOp>(0)), "Operations applied to C.")
                ("d-ops",                    pomain::value<std::vector<TensorOp>>()->default_value(std::vector<TensorOp>(0)), "Operations applied to D.")

                ("problem-start-idx",        pomain::value<int>()->default_value(0),  "First problem to run")
                ("num-problems",             pomain::value<int>()->default_value(-1), "Number of problems to run")

                ("solution-start-idx",       pomain::value<int>()->default_value(-1), "First solution to run")
                ("num-solutions",            pomain::value<int>()->default_value(-1), "Number of solutions to run")
                ("best-solution",            pomain::value<bool>()->default_value(false), "Best solution benchmark mode")

                ("results-file",             pomain::value<std::string>()->default_value("results.csv"), "File name to write results.")
                ("log-file",                 pomain::value<std::string>(),                               "File name for output log.")
                ("log-file-append",          pomain::value<bool>()->default_value(false),                "Append to log file.")
                ("log-level",                pomain::value<LogLevel>()->default_value(LogLevel::Debug),  "Log level")

                ("library-update-file",      pomain::value<std::string>()->default_value(""), "File name for writing indices "
                                                                                          "and speeds suitable for updating "
                                                                                          "an existing library logic file.")
                ("library-update-comment",   pomain::value<bool>()->default_value(false), "Include solution name as a "
                                                                                      "comment in library update "
                                                                                      "file.")


                ("exit-on-error",            pomain::value<bool>()->default_value(false), "Exit run early on failed kernels or other errors.")
                ("selection-only",           pomain::value<bool>()->default_value(false), "Don't run any solutions, only print kernel selections.")
                ("max-workspace-size",       pomain::value<size_t>()->default_value(32*1024*1024), "Max workspace for training")
                ("granularity-threshold",    pomain::value<double>()->default_value(0.0), "Don't run a solution if total granularity is below")
                ;
            // clang-format on

            return options;
        }

        std::shared_ptr<Hardware> GetHardware(pomain::variables_map& args)
        {
            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));

            int deviceIdx = args["device-idx"].as<int>();

            if(deviceIdx >= deviceCount)
                throw std::runtime_error(concatenate(
                    "Invalid device index ", deviceIdx, " (", deviceCount, " total found.)"));

            HIP_CHECK_EXC(hipSetDevice(deviceIdx));

            return hip::GetCurrentDevice();
        }

        hipStream_t GetStream(pomain::variables_map& args)
        {
            if(args["use-default-stream"].as<bool>())
                return 0;

            hipStream_t stream;
            HIP_CHECK_EXC(hipStreamCreate(&stream));
            return stream;
        }

        std::shared_ptr<MasterSolutionLibrary<ContractionProblem>>
            LoadSolutionLibrary(pomain::variables_map& args)
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

        void LoadCodeObjects(pomain::variables_map& args, hip::SolutionAdapter& adapter)
        {
            auto const& filenames = args["code-object"].as<std::vector<std::string>>();
            auto        logLevel  = args["log-level"].as<LogLevel>();

            if(filenames.empty())
            {
                adapter.loadEmbeddedCodeObjects();
            }
            else
            {
                //only trigger exception when failed to load all code objects.
                bool       loaded   = false;
                hipError_t retError = hipSuccess;

                for(auto const& filename : filenames)
                {
                    hipError_t ret;

                    if(logLevel >= LogLevel::Verbose)
                        std::cout << "Loading " << filename << std::endl;
                    ret = adapter.loadCodeObjectFile(filename);

                    if(ret == hipSuccess)
                        loaded = true;
                    else
                        retError = ret;
                }

                if(!loaded)
                    HIP_CHECK_EXC(retError);
            }
        }

        std::vector<size_t> split_ints(std::string const& value)
        {
            std::vector<std::string> parts;
            pomain::split(parts, value, pomain::algorithm::is_any_of(",;"));

            std::vector<size_t> rv;
            rv.reserve(parts.size());

            for(auto const& part : parts)
                if(part != "")
                    rv.push_back(pomain::lexical_cast<size_t>(part));

            return rv;
        }

        void parse_unconfig_arg(pomain::variables_map& args, std::string opt, std::string strValue)
        {
            if(opt.compare(0, 5, "init-") == 0)
            {
                InitMode mode;
                if(strValue == ToString(InitMode::Zero))
                    mode = InitMode::Zero;
                else if(strValue == ToString(InitMode::One))
                    mode = InitMode::One;
                else if(strValue == ToString(InitMode::Two))
                    mode = InitMode::Two;
                else if(strValue == ToString(InitMode::Random))
                    mode = InitMode::Random;
                else if(strValue == ToString(InitMode::NaN))
                    mode = InitMode::NaN;
                else if(strValue == ToString(InitMode::Inf))
                    mode = InitMode::Inf;
                else if(strValue == ToString(InitMode::BadInput))
                    mode = InitMode::BadInput;
                else if(strValue == ToString(InitMode::BadOutput))
                    mode = InitMode::BadOutput;
                else if(strValue == ToString(InitMode::SerialIdx))
                    mode = InitMode::SerialIdx;
                else if(strValue == ToString(InitMode::SerialDim0))
                    mode = InitMode::SerialDim0;
                else if(strValue == ToString(InitMode::SerialDim1))
                    mode = InitMode::SerialDim1;
                else if(strValue == ToString(InitMode::Identity))
                    mode = InitMode::Identity;
                else if(strValue == ToString(InitMode::TrigSin))
                    mode = InitMode::TrigSin;
                else if(strValue == ToString(InitMode::TrigCos))
                    mode = InitMode::TrigCos;
                else if(strValue == ToString(InitMode::TrigAbsSin))
                    mode = InitMode::TrigAbsSin;
                else if(strValue == ToString(InitMode::TrigAbsCos))
                    mode = InitMode::TrigAbsCos;
                else if(strValue == ToString(InitMode::RandomNarrow))
                    mode = InitMode::RandomNarrow;
                else if(strValue == ToString(InitMode::NegOne))
                    mode = InitMode::NegOne;
                else if(strValue == ToString(InitMode::Max))
                    mode = InitMode::Max;
                else if(strValue == ToString(InitMode::DenormMin))
                    mode = InitMode::DenormMin;
                else if(strValue == ToString(InitMode::DenormMax))
                    mode = InitMode::DenormMax;
                auto type = args[opt].as<InitMode>();
                args.at(opt).set(mode);
            }
            else if(opt.compare(0, 12, "bounds-check") == 0)
            {
                BoundsCheckMode mode;
                if(strValue == "Disable")
                    mode = BoundsCheckMode::Disable;
                else if(strValue == "NaN")
                    mode = BoundsCheckMode::NaN;
                else if(strValue == "GuardPageFront")
                    mode = BoundsCheckMode::GuardPageFront;
                else if(strValue == "GuardPageBack")
                    mode = BoundsCheckMode::GuardPageBack;
                else if(strValue == "GuardPageAll")
                    mode = BoundsCheckMode::GuardPageAll;
                else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
                {
                    int value = atoi(strValue.c_str());
                    if(value >= 0 && value < static_cast<int>(BoundsCheckMode::MaxMode))
                        mode = static_cast<BoundsCheckMode>(value);
                    else
                        throw std::runtime_error(
                            concatenate("Can't convert ", strValue, " to BoundsCheckMode."));
                }
                else
                {
                    throw std::runtime_error(
                        concatenate("Can't convert ", opt, " to BoundsCheckMode."));
                }
                args.at(opt).set(mode);
            }
            else
            {
                throw std::runtime_error(concatenate("Can't config ", strValue, " option."));
            }
        }

        void fix_data_types(pomain::variables_map& args)
        {
            auto type = args["type"].as<DataType>();

            // These types use the same data type for all inputs/outputs, so we allow
            // using the overarching 'type' parameter.
            if(type == DataType::Float || type == DataType::Double || type == DataType::ComplexFloat
               || type == DataType::ComplexDouble || type == DataType::Int32)
            {
                args.at("a-type").set(type);
                args.at("b-type").set(type);
                args.at("c-type").set(type);
                args.at("d-type").set(type);
                args.at("alpha-type").set(type);
                args.at("beta-type").set(type);
            }
        }

        pomain::variables_map parse_args(int argc, const char* argv[])
        {
            auto                  options = all_options();
            pomain::variables_map args;
            pomain::store(pomain::parse_command_line(argc, argv, options), args);
            pomain::notify(args);
            if(args.count("help"))
            {
                std::cout << options << std::endl;
                exit(1);
            }

            std::unordered_map<std::string, std::string> unconfig;
            if(args.count("config-file"))
            {
                auto configFiles = args["config-file"].as<std::vector<std::string>>();
                for(auto filename : configFiles)
                {
                    std::cout << "loading config file " << filename << std::endl;
                    std::ifstream file(filename.c_str());
                    if(file.bad())
                        throw std::runtime_error(concatenate("Could not open ", filename));
                    pomain::store(pomain::parse_config_file(file, options), args, &unconfig);
                }
            }

            if(!unconfig.empty())
            {
                for(auto iter = unconfig.begin(); iter != unconfig.end(); iter++)
                {
                    std::string opt = iter->first;
                    std::string val = iter->second;
                    parse_unconfig_arg(args, opt, val);
                }
            }

            fix_data_types(args);

            return args;
        }

        size_t getMaxWorkspace(std::shared_ptr<MasterSolutionLibrary<ContractionProblem>>& library,
                               std::shared_ptr<Hardware>&                                  hardware,
                               pomain::variables_map const&                                args,
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

    int  firstSolutionIdx = args["solution-start-idx"].as<int>();
    int  numSolutions     = args["num-solutions"].as<int>();
    bool gpuTimer         = args["use-gpu-timer"].as<bool>();
    bool runKernels       = !args["selection-only"].as<bool>();
    bool exitOnError      = args["exit-on-error"].as<bool>();

    if(firstSolutionIdx < 0)
        firstSolutionIdx = library->solutions.begin()->first;

    if(numSolutions < 0)
    {
        auto iter = library->solutions.end();
        iter--;
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
    reporters->addReporter(LibraryUpdateReporter::Default(args));

    if(args.count("log-file"))
    {
        std::string filename = args["log-file"].as<std::string>();
        auto        logFile  = std::make_shared<std::ofstream>(
            filename.c_str(), args["log-file-append"].as<bool>() ? std::ios::app : std::ios::out);

        reporters->addReporter(LogReporter::Default(args, logFile, LogLevel::Normal));
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
                            size_t       eventCount        = gpuTimer ? kernels.size() : 0;
                            TimingEvents warmupStartEvents(warmupInvocations, eventCount);
                            TimingEvents warmupStopEvents(warmupInvocations, eventCount);

                            for(int i = 0; i < warmupInvocations; i++)
                            {
                                listeners.preWarmup();
                                if(gpuTimer)
                                    HIP_CHECK_EXC(adapter.launchKernels(kernels,
                                                                        stream,
                                                                        warmupStartEvents[i],
                                                                        warmupStopEvents[i]));
                                else
                                    HIP_CHECK_EXC(
                                        adapter.launchKernels(kernels, stream, nullptr, nullptr));
                                listeners.postWarmup();
                            }

                            listeners.validateWarmups(inputs, warmupStartEvents, warmupStopEvents);

                            size_t syncs = listeners.numSyncs();
                            size_t enq   = listeners.numEnqueuesPerSync();

                            listeners.preSyncs();

                            for(int i = 0; i < syncs; i++)
                            {
                                TimingEvents startEvents(enq, eventCount);
                                TimingEvents stopEvents(enq, eventCount);

                                listeners.preEnqueues();

                                for(int j = 0; j < enq; j++)
                                {
                                    if(gpuTimer)
                                        HIP_CHECK_EXC(adapter.launchKernels(
                                            kernels, stream, startEvents[j], stopEvents[j]));
                                    else
                                        HIP_CHECK_EXC(adapter.launchKernels(
                                            kernels, stream, nullptr, nullptr));
                                }

                                listeners.postEnqueues(startEvents, stopEvents);
                                listeners.validateEnqueues(inputs, startEvents, stopEvents);
                            }

                            listeners.postSyncs();
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

                if(exitOnError && listeners.error() > 0)
                {
                    // error range in shell is [0-255]
                    return std::min(listeners.error(), 255);
                }
            }

            listeners.postProblem();
        }

        listeners.postBenchmarkRun();
    }

    listeners.finalizeReport();

    // error range in shell is [0-255]
    return std::min(listeners.error(), 255);
}
