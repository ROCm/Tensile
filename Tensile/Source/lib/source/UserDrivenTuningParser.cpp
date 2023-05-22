#include <Tensile/UserDrivenTuningParser.hpp>

#include <Tensile/DataTypes.hpp>

#include <fstream>
#include <sstream>
#include <utility>

namespace Tensile
{
    inline DataType convertToDataType(const std::string& DataTypeStr)
    {
        return DataTypeStr == "f16_r" || DataTypeStr == "h"   ? DataType::Half
               : DataTypeStr == "f32_r" || DataTypeStr == "s" ? DataType::Float
               : DataTypeStr == "f64_r" || DataTypeStr == "d" ? DataType::Double
               : DataTypeStr == "bf16_r"                      ? DataType::BFloat16
               : DataTypeStr == "f16_c"                       ? DataType::None
               : DataTypeStr == "f32_c" || DataTypeStr == "c" ? DataType::ComplexFloat
               : DataTypeStr == "f64_c" || DataTypeStr == "z" ? DataType::ComplexDouble
               : DataTypeStr == "bf16_c"                      ? DataType::None
               : DataTypeStr == "i8_r"                        ? DataType::Int8
               : DataTypeStr == "i32_r"                       ? DataType::Int32
               : DataTypeStr == "i8_c"                        ? DataType::None
               : DataTypeStr == "i32_c"                       ? DataType::None
               : DataTypeStr == "u8_r"                        ? DataType::None
               : DataTypeStr == "u32_r"                       ? DataType::None
               : DataTypeStr == "u8_c"                        ? DataType::None
               : DataTypeStr == "u32_c"                       ? DataType::None
                                                              : DataType::None;
    }

    ContractionProblem ConstructTensileProblem(bool     transA,
                                               bool     transB,
                                               DataType inputType,
                                               DataType outputType,
                                               DataType computeType,
                                               size_t   m,
                                               size_t   n,
                                               size_t   k,
                                               size_t   b,
                                               size_t   ldA,
                                               size_t   strideA,
                                               size_t   ldB,
                                               size_t   strideB,
                                               size_t   ldC,
                                               size_t   strideC,
                                               double   alpha,
                                               double   beta)
    {
        // Tensor descriptors for a, b
        TensorDescriptor tdA;
        TensorDescriptor tdB;

        // Tensor ops for matrices, like complex conjugate
        TensorOps aops, bops, cops, dops;

        // Tensile Indices for contraction problem
        ContractionProblem::FreeIndices  freeIndex(2);
        ContractionProblem::BoundIndices boundIndex(1);
        ContractionProblem::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the inputs.
        // It optimizes all problems with alpha==0 into K=0 and alpha=(don't care)
        k = (k && alpha) ? k : 0;

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(transA)
        {
            tdA = {
                    inputType,
                    {k, m, b},
                    {strideA, strideA, strideA},
                    0
                };
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            tdA = {
                    inputType,
                    {m, k, b},
                    {strideA, strideA, strideA},
                    0
                };
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(transB)
        {
            tdB = {
                    inputType,
                    {n, k, b},
                    {strideB, strideB, strideB},
                    0
                };
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            tdB = {
                    inputType,
                    {k, n, b},
                    {strideB, strideB, strideB},
                    0
                };
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // Descriptor for input matrix C
        TensorDescriptor tdC{outputType, {m, n, b}, {strideC, strideC, strideC}, 0};

        // Descriptor for output matrix D
        TensorDescriptor tdD{outputType, {m, n, b}, {strideC, strideC, strideC}, 0};

        // The ContractionProblem
        ContractionProblem tensileProblem{
            tdA, aops, tdB, bops, tdC, cops, tdD, dops, freeIndex, batchIndex, boundIndex, beta};

        tensileProblem.setAlphaType(computeType);
        tensileProblem.setBetaType(computeType);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(
            ((inputType == DataType::Half) || (inputType == DataType::BFloat16))
            && (computeType == DataType::Float));

        // Environment variable to force use of VALU for double precision gemm
        static bool force_valu_for_dgemm = std::getenv("ROCBLAS_INTERNAL_FORCE_VALU_FOR_DGEMM");
        if((inputType == DataType::Double) && (outputType == DataType::Double)
           && (computeType == DataType::Double) && force_valu_for_dgemm)
        {
            tensileProblem.setArithmeticUnit(Tensile::ArithmeticUnit::VALU);
        }

        // set batch mode
        tensileProblem.setStridedBatched((strideA > 1) || (strideB > 1) || (strideC > 1));

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set tensileAlpha=0
        // Not positive if this is necessary here as well
        // typename AlphaBeta<Ti, To, Tc>::tensile_type tensileAlpha;
        if(!k)
            alpha = 0.0;
        tensileProblem.setAlphaRestriction(Tensile::toScalarValueEnum(alpha));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(true);

        static const char* fp16AltImplEnvStr = std::getenv("ROCBLAS_INTERNAL_FP16_ALT_IMPL");
        static const int   fp16AltImplEnv
            = (fp16AltImplEnvStr == NULL ? -1 : (std::atoi(fp16AltImplEnvStr) == 0 ? 0 : 1));
        if(fp16AltImplEnv != -1)
            tensileProblem.setFp16AltImpl(fp16AltImplEnv);

        static const char* fp16AltImplRoundEnvStr
            = std::getenv("ROCBLAS_INTERNAL_FP16_ALT_IMPL_RNZ");
        static const int fp16AltImplRoundEnv
            = (fp16AltImplRoundEnvStr == NULL ? -1
                                              : (std::atoi(fp16AltImplRoundEnvStr) == 0 ? 0 : 1));
        if(fp16AltImplRoundEnv != -1)
            tensileProblem.setFp16AltImplRound(fp16AltImplRoundEnv);

        return tensileProblem;
    }

    std::pair<ContractionProblem, int> problemFromEntries(std::vector<std::string> entries)
    {
        const size_t entries_n = entries.size();
        if((entries_n != 15) && (entries_n != 18))
        {
            return std::make_pair(ContractionProblem{}, -1);
        }

        // Common
        bool transA = (entries[0] != "N");
        bool transB = (entries[1] != "N");

        size_t m, n, b, k;
        size_t ldA, ldB, ldC;
        size_t strideA, strideB, strideC;
        double alpha, beta;

        DataType inputType   = DataType::None;
        DataType outputType  = DataType::None;
        DataType computeType = DataType::None;

        int solution_idx = -1;

        try
        {
            m = std::stol(entries[2]);
            n = std::stol(entries[3]);
            b = std::stol(entries[4]);
            k = std::stol(entries[5]);

            alpha = std::stod(entries[8]);
            beta  = std::stod(entries[7]);

            ldA = std::stol(entries[8]);
            ldB = std::stol(entries[9]);
            ldC = std::stol(entries[10]);

            strideA = 1;
            strideB = 1;
            strideC = 1;

            if(entries_n == 15)
            {
                // Expected layout: transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,input_type,output_type,compute_type,solution_index
                inputType   = convertToDataType(entries[11]);
                outputType  = convertToDataType(entries[12]);
                computeType = convertToDataType(entries[13]);

                solution_idx = std::stoi(entries[14]);
            }
            else if(entries_n == 18)
            {
                // Expected layout: transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,stride_a,stride_b,stride_c,input_type,output_type,compute_type,solution_index
                strideA = std::stol(entries[11]);
                strideB = std::stol(entries[12]);
                strideC = std::stol(entries[13]);

                inputType   = convertToDataType(entries[14]);
                outputType  = convertToDataType(entries[15]);
                computeType = convertToDataType(entries[16]);

                solution_idx = std::stoi(entries[17]);
            }
        }
        catch(std::invalid_argument const& ex)
        {
            std::make_pair(ContractionProblem{}, -1);
        }
        catch(std::out_of_range const& ex)
        {
            std::make_pair(ContractionProblem{}, -1);
        }

        if(inputType == DataType::None || outputType == DataType::None
           || computeType == DataType::None)
        {
            return std::make_pair(ContractionProblem{}, -1);
        }

        ContractionProblem problem = ConstructTensileProblem(transA,
                                                             transB,
                                                             inputType,
                                                             outputType,
                                                             computeType,
                                                             m,
                                                             n,
                                                             k,
                                                             b,
                                                             ldA,
                                                             strideA,
                                                             ldB,
                                                             strideB,
                                                             ldC,
                                                             strideC,
                                                             alpha,
                                                             beta);

        return std::make_pair(problem, solution_idx);
    }

    std::vector<std::pair<ContractionProblem, int>> getContractionProblemsFromFile(std::string path)
    {
        std::vector<std::pair<ContractionProblem, int>> out;

        std::ifstream file(path);
        std::string   line, entry;

        const auto delim         = ',';
        const auto first_heading = "transA";

        int current_section = -1;

        while(std::getline(file, line))
        {
            // Ignore lines without delimiter
            if(line.find(delim) == std::string::npos)
            {
                continue;
            }

            // Check for section start
            if(line.find(first_heading) != std::string::npos)
            {
                // TODO: Get param index from headings?
                current_section++;
                continue;
            }

            std::vector<std::string> entries{};
            entries.reserve((current_section == 0) ? 15 : 18);

            std::stringstream line_ss(line);
            while(getline(line_ss, entry, delim))
            {
                entries.push_back(entry);
            }

            auto problemSolution = problemFromEntries(entries);
            if(problemSolution.second > 0)
            {
                out.push_back(problemSolution);
            }
        }

        return out;
    }
};