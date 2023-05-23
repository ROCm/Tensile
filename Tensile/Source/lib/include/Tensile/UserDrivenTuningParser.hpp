#pragma once

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <string>
#include <vector>

namespace Tensile
{
    template <typename MyProblem>
    class ProblemOverride
    {
    public:
        ProblemOverride();
        ProblemOverride(bool     transA,
                        bool     transB,
                        DataType inputType,
                        DataType outputType,
                        bool     HPA,
                        size_t   m,
                        size_t   n,
                        size_t   k,
                        size_t   batchSize,
                        double   beta,
                        size_t   ldA,
                        size_t   strideA,
                        size_t   ldB,
                        size_t   strideB,
                        size_t   ldC,
                        size_t   strideC);
        ProblemOverride(const MyProblem& problem);

        inline bool transA() const
        {
            return m_transA;
        }
        inline bool transB() const
        {
            return m_transB;
        }
        inline DataType inputType() const
        {
            return m_inputType;
        }
        inline DataType outputType() const
        {
            return m_outputType;
        }
        inline bool HPA() const
        {
            return m_HPA;
        }
        inline size_t m() const
        {
            return m_m;
        }
        inline size_t n() const
        {
            return m_n;
        }
        inline size_t k() const
        {
            return m_k;
        }
        inline size_t batchSize() const
        {
            return m_batchSize;
        }
        inline double beta() const
        {
            return m_beta;
        }
        inline size_t ldA() const
        {
            return m_ldA;
        }
        inline size_t strideA() const
        {
            return m_strideA;
        }
        inline size_t ldB() const
        {
            return m_ldB;
        }
        inline size_t strideB() const
        {
            return m_strideB;
        }
        inline size_t ldC() const
        {
            return m_ldC;
        }
        inline size_t strideC() const
        {
            return m_strideC;
        }

    private:
        bool     m_transA;
        bool     m_transB;
        DataType m_inputType;
        DataType m_outputType;
        bool     m_HPA;
        size_t   m_m;
        size_t   m_n;
        size_t   m_k;
        size_t   m_batchSize;
        double   m_beta;
        size_t   m_ldA;
        size_t   m_strideA;
        size_t   m_ldB;
        size_t   m_strideB;
        size_t   m_ldC;
        size_t   m_strideC;
    };

    template <>
    ProblemOverride<ContractionProblem>::ProblemOverride(const ContractionProblem& problem);

    template <>
    struct Comparison<ProblemOverride<ContractionProblem>>
    {
        enum
        {
            implemented = true
        };

        static int compare(ProblemOverride<ContractionProblem> const& lhs,
                           ProblemOverride<ContractionProblem> const& rhs)
        {
            return LexicographicCompare(lhs.transA(),
                                        rhs.transA(),
                                        lhs.transB(),
                                        rhs.transB(),
                                        lhs.inputType(),
                                        rhs.inputType(),
                                        lhs.outputType(),
                                        rhs.outputType(),
                                        lhs.HPA(),
                                        rhs.HPA(),
                                        lhs.m(),
                                        rhs.m(),
                                        lhs.n(),
                                        rhs.n(),
                                        lhs.k(),
                                        rhs.k(),
                                        lhs.batchSize(),
                                        rhs.batchSize(),
                                        lhs.beta(),
                                        rhs.beta(),
                                        lhs.ldA(),
                                        rhs.ldA(),
                                        lhs.strideA(),
                                        rhs.strideA(),
                                        lhs.ldB(),
                                        rhs.ldB(),
                                        lhs.strideB(),
                                        rhs.strideB(),
                                        lhs.ldC(),
                                        rhs.ldC(),
                                        lhs.strideC(),
                                        rhs.strideC());
        }
    };

    template <typename MyProblem>
    std::pair<ProblemOverride<MyProblem>, int>
        problemFromEntries(const std::vector<std::string>& entries);

    template <typename MyProblem>
    std::vector<std::pair<ProblemOverride<MyProblem>, int>>
        getContractionProblemsFromFile(const std::string& path);
} // namespace Tensile

namespace std
{
    template <>
    struct hash<Tensile::ProblemOverride<Tensile::ContractionProblem>>
    {
        inline size_t
            operator()(Tensile::ProblemOverride<Tensile::ContractionProblem> const& po) const
        {
            return Tensile::hash_combine(po.transA(),
                                         po.transB(),
                                         po.inputType(),
                                         po.outputType(),
                                         po.HPA(),
                                         po.m(),
                                         po.n(),
                                         po.k(),
                                         po.batchSize(),
                                         po.beta(),
                                         po.ldA(),
                                         po.strideA(),
                                         po.ldB(),
                                         po.strideB(),
                                         po.ldC(),
                                         po.strideC());
        }
    };
} // namespace std