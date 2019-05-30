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

#include <Tensile/Contractions.hpp>
#include <Tensile/Tensile.hpp>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

po::options_description all_options()
{
    po::options_description options("Tensile client options");

    options.add_options()("help,h", "Show help message.")

        ("library-file",
         po::value<std::string>(),
         "Load a (YAML) solution library.  If not specified, we will use "
         "the embedded library, if available.")("code-object,c",
                                                po::value<std::vector<std::string>>(),
                                                "Code object file with kernel(s).  If none are "
                                                "specified, we will use the embedded code "
                                                "object(s) if available.")

            ("problem-identifier",
             po::value<std::string>(),
             "Problem identifer (Einstein notation). Either "
             "this or free/batch/bound must be specified.")(
                "free",
                po::value<Tensile::ContractionProblem::FreeIndices>(),
                "Free index. Order: a,b,ca,cb,da,db")(
                "batch",
                po::value<Tensile::ContractionProblem::BatchIndices>(),
                "Batch index. Order: a,b,c,d")(
                "bound",
                po::value<Tensile::ContractionProblem::BoundIndices>(),
                "Bound/summation index. Order: a,b")

                ("type", po::value<Tensile::DataType>(), "Data type")(
                    "a-type", po::value<Tensile::DataType>(), "A data type")(
                    "b-type", po::value<Tensile::DataType>(), "B data type")(
                    "c-type", po::value<Tensile::DataType>(), "C data type")(
                    "d-type", po::value<Tensile::DataType>(), "D data type")(
                    "alpha-type", po::value<Tensile::DataType>(), "alpha data type")(
                    "beta-type", po::value<Tensile::DataType>(), "beta data type")

                    ("init-a", po::value<int>()->default_value(3), "Initialization for A")(
                        "init-b", po::value<int>()->default_value(3), "Initialization for B")(
                        "init-c", po::value<int>()->default_value(3), "Initialization for C")(
                        "init-d", po::value<int>()->default_value(0), "Initialization for D")(
                        "init-alpha",
                        po::value<int>()->default_value(2),
                        "Initialization for alpha")(
                        "init-beta", po::value<int>()->default_value(2), "Initialization for beta")(
                        "c-equal-d", po::value<bool>()->default_value(false), "C equals D")

                        ("print-valids",
                         po::value<bool>()->default_value(false),
                         "Print values that pass validation")("print-max",
                                                              po::value<int>()->default_value(0),
                                                              "Max number of values to print")(
                            "num-elements-to-validate",
                            po::value<int>()->default_value(0),
                            "Number of elements to validate")

                            ("device-idx", po::value<int>()->default_value(0), "Device index")(
                                "platform-idx",
                                po::value<int>()->default_value(0),
                                "OpenCL Platform Index")

                                ("num-benchmarks",
                                 po::value<int>()->default_value(1),
                                 "Number of benchmarks to run")("num-enqueues-per-sync",
                                                                po::value<int>()->default_value(1),
                                                                "Enqueues per sync")(
                                    "num-syncs-per-benchmark",
                                    po::value<int>()->default_value(1),
                                    "Syncs per benchmark")("use-gpu-timer",
                                                           po::value<bool>()->default_value(true),
                                                           "Use GPU timer")(
                                    "sleep-percent",
                                    po::value<int>()->default_value(0),
                                    "Sleep percentage")

                                    ("problem-size,p",
                                     po::value<std::vector<std::string>>(),
                                     "Specify a problem size.  Comma-separated list of "
                                     "sizes, in the order of the Einstein notation.")

                                        ("a-strides",
                                         po::value<std::vector<std::string>>(),
                                         "A strides. Unspecified means no padding, "
                                         "specifying once applies to all problem sizes, "
                                         "otherwise specify once per problem size.")

                                            ("b-strides",
                                             po::value<std::vector<std::string>>(),
                                             "B strides. Unspecified means no padding, "
                                             "specifying once applies to all problem sizes, "
                                             "otherwise specify once per problem size.")

                                                ("c-strides",
                                                 po::value<std::vector<std::string>>(),
                                                 "C strides. Unspecified means no padding, "
                                                 "specifying once applies to all problem sizes, "
                                                 "otherwise specify once per problem size.")

                                                    ("d-strides",
                                                     po::value<std::vector<std::string>>(),
                                                     "D strides. Unspecified means no padding, "
                                                     "specifying once applies to all problem "
                                                     "sizes, "
                                                     "otherwise specify once per problem size.")

                                                        ("solution-start-idx",
                                                         po::value<int>()->default_value(0),
                                                         "First solution to run")(
                                                            "num-solutions",
                                                            po::value<int>()->default_value(-1),
                                                            "Number of solutions to run");

    return options;
}

std::vector<int> split_ints(std::string const& value)
{
    std::vector<std::string> parts;
    boost::split(parts, value, boost::algorithm::is_any_of(","));

    std::vector<int> rv;
    rv.reserve(parts.size());

    for(auto const& part : parts)
        rv.push_back(boost::lexical_cast<int>(part));

    return rv;
}

void parse_arg_ints(po::variables_map& args, std::string const& name)
{
    auto const& inValue = args[name].as<std::vector<std::string>>();

    std::vector<std::vector<int>> outValue;
    outValue.reserve(inValue.size());
    for(auto const& str : inValue)
        outValue.push_back(split_ints(str));

    boost::any v(outValue);

    args.at(name).value() = v;
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

    parse_arg_ints(args, "problem_size");
    parse_arg_ints(args, "a-strides");
    parse_arg_ints(args, "b-strides");
    parse_arg_ints(args, "c-strides");
    parse_arg_ints(args, "d-strides");

    return args;
}

int main(int argc, const char* argv[])
{
    using namespace Tensile;
    using namespace Tensile::Client;

    auto args = parse_args(argc, argv);

    ClientProblemFactory problemFactory(args);

    auto dataInit = DataInitialization::Get(args, problemFactory);

    auto gpuInputs = dataInit->prepareGPUInputs();

    return 0;
}
