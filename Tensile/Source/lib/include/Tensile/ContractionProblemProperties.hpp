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

#pragma once

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/PropertyMatching.hpp>

#include <cstddef>

namespace Tensile
{
    /**
 * \addtogroup PropertyClasses
 * @{
 */
    namespace Contraction
    {
        struct FreeSizeA : public Property_CRTP<FreeSizeA, ContractionProblem>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "FreeSizeA";
            }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.freeSizeA(index);
            }
        };

        struct FreeSizeB : public Property_CRTP<FreeSizeB, ContractionProblem>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "FreeSizeB";
            }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.freeSizeB(index);
            }
        };

        struct BatchSize : public Property_CRTP<BatchSize, ContractionProblem>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "BatchSize";
            }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.batchSize(index);
            }
        };

        struct BoundSize : public Property_CRTP<BoundSize, ContractionProblem>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "BoundSize";
            }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.boundSize(index);
            }
        };

        struct AStride : public Property_CRTP<AStride, ContractionProblem>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "AStride";
            }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.a().strides()[index];
            }
        };

        struct BStride : public Property_CRTP<BStride, ContractionProblem>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "BStride";
            }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.b().strides()[index];
            }
        };

        struct CStride : public Property_CRTP<CStride, ContractionProblem>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "CStride";
            }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.c().strides()[index];
            }
        };

        struct DStride : public Property_CRTP<DStride, ContractionProblem>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "DStride";
            }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.d().strides()[index];
            }
        };

        struct OperationIdentifier
            : public Property_CRTP<OperationIdentifier, ContractionProblem, std::string>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };

            static std::string Type()
            {
                return "OperationIdentifier";
            }

            virtual std::string operator()(ContractionProblem const& problem) const
            {
                return problem.operationIdentifier();
            }
        };

        // Helper functions TODO: Consolidate with ContractionSolution::computeGranularities
        float numTiles0(ContractionProblem const& problem, float macro_tile_0_inv)
        {
            // Get problem size (M)
            float M = problem.freeSizeA(0);
            if(problem.freeIndicesA().size() > 1)
                assert(false); //TODO: Handle this case

            return M * macro_tile_0_inv;
        }

        float numTiles1(ContractionProblem const& problem, float macro_tile_1_inv)
        {
            // Get problem size (N)
            float N = problem.freeSizeB(0);
            if(problem.freeIndicesB().size() > 1)
                assert(false); //TODO: Handle this case

            // Calculate granularity
            return N * macro_tile_1_inv;
        }

        float tileGranularity(float numTiles)
        {
            return numTiles / ceil(numTiles);
        }

        struct Tile0Granularity : public Property_CRTP<Tile0Granularity, ContractionProblem, float>
        {
            enum
            {
                HasIndex = false,
                HasValue = true
            };
            float value;  // 1/mt0

            static std::string Type()
            {
                return "Tile0Granularity";
            }

            virtual float operator()(ContractionProblem const& problem) const
            {
                return tileGranularity(numTiles0(problem, value));
            }
        };

        struct Tile1Granularity : public Property_CRTP<Tile1Granularity, ContractionProblem, float>
        {
            enum
            {
                HasIndex = false,
                HasValue = true
            };
            float value;    // 1/mt1

            static std::string Type()
            {
                return "Tile1Granularity";
            }

            virtual float operator()(ContractionProblem const& problem) const
            {
                return tileGranularity(numTiles1(problem, value));
            }
        };

        struct CUGranularity : public Property_CRTP<CUGranularity, ContractionProblem, float>
        {
            enum
            {
                HasIndex = false,
                HasValue = true,

            };
            ContractionSolution::GranularityScaleFactors value;
            /* General scaling = (1 / (numCUs / globalSplitU / localSplitU))
             * See: `ContractionSolution::computeGranularities`
             */

            static std::string Type()
            {
                return "CUGranularity";
            }

            virtual float operator()(ContractionProblem const& problem) const
            {
                float NumBatches = 1; // TODO: Higher batch sizes
                float tilesPerCu = NumBatches
                                        * ceil(numTiles0(problem, value.mt0_scale)) 
                                        * ceil(numTiles1(problem, value.mt1_scale))
                                        * value.devSolScale;  
                return tileGranularity(tilesPerCu);
            }
        };

        struct WavesPerSIMD : public Property_CRTP<WavesPerSIMD, ContractionProblem, float>
        {
            enum
            {
                HasIndex = false,
                HasValue = true,

            };
            ContractionSolution::GranularityScaleFactors value;  
            /* General scaling = ((globalSplitU / numCUs)
             *                     * ceil((workGroupX * workGroupY) / wavefrontSize)
             *                     / (2 * simdPerCU))
             * See: `ContractionSolution::computeGranularities`
             */

            static std::string Type()
            {
                return "WavesPerSIMD";
            }

            virtual float operator()(ContractionProblem const& problem) const
            {
                float totalTiles = ceil(numTiles0(problem, value.mt0_scale)) * ceil(numTiles1(problem, value.mt1_scale));
                return totalTiles * value.devSolScale;
            }
        };
    } // namespace Contraction

    /**
 * @}
 */
} // namespace Tensile
