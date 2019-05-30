
#pragma once

#include <Tensile.h>

namespace OldTensile
{
    using TensileGEMM = TensileStatus (*)(float*       dataD,
                                          const float* dataC,
                                          const float* dataA,
                                          const float* dataB,
                                          float        alpha,
                                          float        beta,
                                          unsigned int strideC1J,
                                          unsigned int strideC2K,
                                          unsigned int strideA1L,
                                          unsigned int strideA2K,
                                          unsigned int strideB1L,
                                          unsigned int strideB2K,
                                          unsigned int sizeI,
                                          unsigned int sizeJ,
                                          unsigned int sizeK,
                                          unsigned int sizeL,
                                          hipStream_t  stream,
                                          unsigned int numInputEvents,
                                          hipEvent_t*  inputEvents,
                                          hipEvent_t*  outputEvent);

    inline TensileGEMM GetTensileGEMM(bool transA, bool transB)
    {
        if(!transA && !transB)
            return tensile_Cijk_Ailk_Bljk_SB;
        if(!transA && transB)
            return tensile_Cijk_Ailk_Bjlk_SB;
        if(transA && !transB)
            return tensile_Cijk_Alik_Bljk_SB;
        if(transA && transB)
            return tensile_Cijk_Alik_Bjlk_SB;
        return nullptr;
    }

    inline TensileStatus CallOldTensile(bool         transA,
                                        bool         transB,
                                        float*       dataD,
                                        const float* dataC,
                                        const float* dataA,
                                        const float* dataB,
                                        float        alpha,
                                        float        beta,
                                        unsigned int strideC1J,
                                        unsigned int strideC2K,
                                        unsigned int strideA1L,
                                        unsigned int strideA2K,
                                        unsigned int strideB1L,
                                        unsigned int strideB2K,
                                        unsigned int sizeI,
                                        unsigned int sizeJ,
                                        unsigned int sizeK,
                                        unsigned int sizeL,
                                        hipStream_t  stream,
                                        unsigned int numInputEvents,
                                        hipEvent_t*  inputEvents,
                                        hipEvent_t*  outputEvent)
    {
        auto fn = GetTensileGEMM(transA, transB);

        return fn(dataD,
                  dataC,
                  dataA,
                  dataB,
                  alpha,
                  beta,
                  strideC1J,
                  strideC2K,
                  strideA1L,
                  strideA2K,
                  strideB1L,
                  strideB2K,
                  sizeI,
                  sizeJ,
                  sizeK,
                  sizeL,
                  stream,
                  numInputEvents,
                  inputEvents,
                  outputEvent);
    }
}
