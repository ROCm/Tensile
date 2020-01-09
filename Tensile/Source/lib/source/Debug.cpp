/**
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <Tensile/Debug.hpp>

#include <mutex>

#ifndef DEBUG_SM
#define DEBUG_SM 0
#endif

namespace Tensile
{
    std::once_flag debug_init;

    bool Debug::printPropertyEvaluation() const
    {
        return value & (0x2 | 0x4);
    }


    bool Debug::printDeviceSelection() const
    {
        return value & 0x8;
    }

    bool Debug::printPredicateEvaluation() const
    {
        return value & 0x10;
    }

    bool Debug::printCodeObjectInfo() const
    {
        return value & 0x20;
    }

    bool Debug::printKernelArguments() const
    {
        return value & 0x40;
    }

    bool Debug::printTensorInfo() const
    {
        return value & 0x80;
    }

    bool Debug::printConvolutionReference1() const
    {
        return value & 0x100;
    }
    bool Debug::printConvolutionReference2() const
    {
        return value & 0x200;
    }
    bool Debug::printConvolutionReference3() const
    {
        return value & 0x400;
    }

    bool Debug::printTensorModeHex() const
    {
        return value & 0x800;
    }


    Debug::Debug()
        : value(DEBUG_SM)
    {
        const char * db = std::getenv("TENSILE_DB");

        if(db)
            value = strtol(db, nullptr, 0);
    }

}
