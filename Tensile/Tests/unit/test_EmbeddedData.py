################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from pathlib import Path
from Tensile.EmbeddedData import generateLibrary

referenceSource = \
"""/*******************************************************************************
* Copyright (C) 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
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
*******************************************************************************/

/**************************************************
* This file was generated by Tensile:             *
* https://github.com/ROCmSoftwarePlatform/Tensile *
**************************************************/




#include <Tensile/EmbeddedData.hpp>

#include <Tensile/Contractions.hpp>
#include <Tensile/Tensile.hpp>

namespace Tensile {
    namespace {
        // myMasterLibrary
        EmbedData<MyBase> TENSILE_EMBED_SYMBOL_NAME("my-library-test", {
            0x31, 0x32, 0x33, 0x34, 0x00});
    } // anonymous namespace
    namespace {
        // mylib1.co
        EmbedData<SolutionAdapter> TENSILE_EMBED_SYMBOL_NAME("my-library-test", {
            0x35, 0x36, 0x37, 0x38});
    } // anonymous namespace
    namespace {
        // mylib2.co
        EmbedData<SolutionAdapter> TENSILE_EMBED_SYMBOL_NAME("my-library-test", {
            0x37, 0x38, 0x39, 0x30});
    } // anonymous namespace
} // namespace Tensile""".split("\n")

def test_generateLibrary():
  name = Path.cwd() / "my-library"
  key = "my-library-test"

  masterLibrary = Path.cwd() / "myMasterLibrary"
  with open(masterLibrary, "w") as f:
    f.write("1234")

  coFiles = [ Path.cwd() / "mylib1.co", Path.cwd() / "mylib2.co" ]
  data = ["5678", "7890"]
  for t in zip(coFiles, data):
    with open(t[0], "w") as f:
      f.write(t[1])

  generateLibrary(name, key, masterLibrary, "MyBase", coFiles)

  with open(name.with_suffix(".cpp")) as f:
    embedSource = f.readlines()
    for e, r in zip(embedSource, referenceSource):
      assert e.rstrip() == r, "Generated file does not match reference."
