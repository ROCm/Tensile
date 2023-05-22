#include <Tensile/UserDrivenTuningParser.hpp>

#include <Tensile/DataTypes.hpp>

#include <iostream>
#include <functional>
#include <fstream>
#include <optional>
#include <sstream>
#include <utility>


namespace Tensile
{
    inline DataType convertToDataType(const std::string& DataTypeStr)
    {
        return
            DataTypeStr == "f16_r" || DataTypeStr == "h" ? DataType::Half          :
            DataTypeStr == "f32_r" || DataTypeStr == "s" ? DataType::Float         :
            DataTypeStr == "f64_r" || DataTypeStr == "d" ? DataType::Double        :
            DataTypeStr == "bf16_r"                      ? DataType::BFloat16      :
            DataTypeStr == "f16_c"                       ? DataType::None          :
            DataTypeStr == "f32_c" || DataTypeStr == "c" ? DataType::ComplexFloat  :
            DataTypeStr == "f64_c" || DataTypeStr == "z" ? DataType::ComplexDouble :
            DataTypeStr == "bf16_c"                      ? DataType::None          :
            DataTypeStr == "i8_r"                        ? DataType::Int8          :
            DataTypeStr == "i32_r"                       ? DataType::Int32         :
            DataTypeStr == "i8_c"                        ? DataType::None          :
            DataTypeStr == "i32_c"                       ? DataType::None          :
            DataTypeStr == "u8_r"                        ? DataType::None          :
            DataTypeStr == "u32_r"                       ? DataType::None          :
            DataTypeStr == "u8_c"                        ? DataType::None          :
            DataTypeStr == "u32_c"                       ? DataType::None          :
            DataType::None;
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
        double beta;

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

            beta = std::stod(entries[7]);

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

        if(inputType == DataType::None ||
           outputType == DataType::None ||
           computeType == DataType::None)
        {
            return std::make_pair(ContractionProblem{}, -1);
        }

        ContractionProblem problem = ContractionProblem::GEMM_Strides(transA,
                                                                        transB,
                                                                        inputType,
                                                                        inputType,
                                                                        outputType,
                                                                        outputType,
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
                                                                        ldC,
                                                                        strideC,
                                                                        beta);
        
        return std::make_pair(problem, solution_idx);
    }

    std::vector<std::pair<ContractionProblem, int>> getContractionProblemsFromFile(std::string path)
    {
        std::vector<std::pair<ContractionProblem, int>> out;
        
        std::ifstream file(path);
        std::string line, entry;

        const auto delim = ',';
        const auto first_heading = "transA";

        int current_section = -1;
        
        while (std::getline(file, line))
        {
            // Ignore lines without delimiter
            if (line.find(delim) == std::string::npos) {
                continue;
            }

            // Check for section start
            if (line.find(first_heading) != std::string::npos) {
                // TODO: Get param index from headings?
                current_section++;
                continue;
            }
            
            std::vector<std::string> entries{};
            entries.reserve((current_section == 0) ? 15 : 18);

            std::stringstream line_ss(line);
            while(getline(line_ss, entry, delim)) {
                entries.push_back(entry);
            }

            auto problemSolution = problemFromEntries(entries);
            if (problemSolution.second > 0)
            {
                out.push_back(problemSolution);
            }
        }

        return out;
    }
};