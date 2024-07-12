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

from io import TextIOWrapper
from pathlib import Path
from typing import List

def toFile(outputFile: Path | TextIOWrapper, contents: List[str], delimiter: str = "\n") -> None:
    """Generates a user specified delimited file. 

    Writes the elements of a List of strings with a given delimiter. The state of
    the file is inspected to determine if it should be opened. If the file is
    already opened, the file is not closed after writing.
    
    Args: 
        outputFile: Path to file for writing manifest.
        contents: List of items to write manifest.
        delimiter: Symbol used to delimit elements when writing file.
    
    Raises:
        AssertionError: If contents is not a List[str]

    Notes:
        - If outputFile is a TextIOWrapper, the file is assumed to be opened.
        - Providing an open file is useful for writing multiple calls to the same file.
    """
    assert isinstance(contents, list), "contents must be a list."
    assert isinstance(contents[0], str), "contents elements must be a str."

    opened = isinstance(outputFile, TextIOWrapper) 

    f = outputFile if opened else open(outputFile, "w")

    for chunk in contents:
        f.write(f"{chunk}{delimiter}")

    if not opened:
        f.close()

    
