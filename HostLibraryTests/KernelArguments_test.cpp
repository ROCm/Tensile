/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>

#include <Tensile/KernelArguments.hpp>

using namespace Tensile;

TEST(KernelArguments, Simple)
{
    KernelArguments args(true);

    std::vector<float> array(5);

    struct
    {
        void * d;
        const void * c;
        const void * a;
        const void * b;

        int x;
        double y;
        float z;
        char t;

        size_t k;

    } ref;

    // Padding bytes would be left uninitialized in the struct but will be zero-filled by
    // the KernelArguments class.  Set them to zero to prevent a test failure. 
    memset(&ref, 0, sizeof(ref));

    ref.d = array.data();
    ref.c = array.data()+1;
    ref.a = array.data()+2;
    ref.b = array.data()+3;
    ref.x = 23;
    ref.y = 90.2;
    ref.z = 16.0f;
    ref.t = 'w';
    ref.k = std::numeric_limits<size_t>::max();

    args.append("d", ref.d);
    args.append("c", ref.c);
    args.append("a", ref.a);
    args.append("b", ref.b);

    args.append("x", ref.x);
    args.append("y", ref.y);
    args.append("z", ref.z);
    args.append("t", ref.t);
    args.append("k", ref.k);

    EXPECT_EQ(args.size(), sizeof(ref));

    std::vector<uint8_t> reference(sizeof(ref), 0);
    memcpy(reference.data(), &ref, sizeof(ref));

    std::vector<uint8_t> result(args.size());
    memcpy(result.data(), args.data(), args.size());

    EXPECT_EQ(result.size(), reference.size());
    for(int i = 0; i < std::min(result.size(), reference.size()); i++)
    {
        EXPECT_EQ(static_cast<uint32_t>(result[i]),
                  static_cast<uint32_t>(reference[i]))
            << "(" << i << ")";
    }

    //std::cout << args << std::endl;

}

TEST(KernelArguments, Binding)
{
    KernelArguments args(true);

    std::vector<float> array(5);

    struct
    {
        void * d;
        const void * c;
        const void * a;
        const void * b;

        int x;
        double y;
        float z;
        char t;

        size_t k;

    } ref;

    // Padding bytes would be left uninitialized in the struct but will be zero-filled by
    // the KernelArguments class.  Set them to zero to prevent a test failure. 
    memset(&ref, 0, sizeof(ref));

    ref.d = array.data();
    ref.c = array.data()+1;
    ref.a = array.data()+2;
    ref.b = array.data()+3;
    ref.x = 23;
    ref.y = 90.2;
    ref.z = 16.0f;
    ref.t = 'w';
    ref.k = std::numeric_limits<size_t>::max();

    args.append("d", ref.d);
    args.append("c", ref.c);
    args.append("a", ref.a);
    args.append("b", ref.b);

    args.appendUnbound<int>("x");
    args.append("y", ref.y);
    args.append("z", ref.z);
    args.append("t", ref.t);
    args.append("k", ref.k);

    EXPECT_EQ(args.size(), sizeof(ref));

    //std::cout << args << std::endl;

    EXPECT_THROW(args.data(), std::runtime_error);

    args.bind("x", ref.x);

    std::vector<uint8_t> reference(sizeof(ref), 0);
    memcpy(reference.data(), &ref, sizeof(ref));

    std::vector<uint8_t> result(args.size());
    memcpy(result.data(), args.data(), args.size());

    EXPECT_EQ(result.size(), reference.size());
    for(int i = 0; i < std::min(result.size(), reference.size()); i++)
    {
        EXPECT_EQ(static_cast<uint32_t>(result[i]),
                  static_cast<uint32_t>(reference[i]))
            << "(" << i << ")";
    }

    //std::cout << args << std::endl;

}

