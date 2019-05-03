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

#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <boost/filesystem.hpp>
#include <boost/version.hpp>

#if BOOST_VERSION >= 106100
#include <boost/dll/runtime_symbol_info.hpp>
#else
#define TEST_DATA_USE_PROC_EXE
#endif

#include <Tensile/Singleton.hpp>

struct TestData: public Tensile::LazySingleton<TestData>
{
    using Base = Tensile::LazySingleton<TestData>;

    static inline boost::filesystem::path File(std::string const& filename)
    {
        return Instance().DataDir() / filename;
    }

    boost::filesystem::path DataDir() const
    {
        return m_dataDir;
    }

private:
    friend Base;

    boost::filesystem::path m_executable;
    boost::filesystem::path m_dataDir;

    static inline boost::filesystem::path ProgramLocation()
    {
#ifdef TEST_DATA_USE_PROC_EXE
        return boost::filesystem::read_symlink("/proc/self/exe");
#else
        return boost::dll::program_location();
#endif

    }

    TestData()
        : m_executable(ProgramLocation()),
          m_dataDir(m_executable.parent_path() / "data")
    {
    }
};


