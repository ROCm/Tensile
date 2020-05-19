################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################


import os
import sys
import argparse
import re

import pandas as pd

def RunMain():

    userArgs = sys.argv[1:]

    argParser = argparse.ArgumentParser()
    argParser.add_argument("current_file", help="path where the current results are located")
    argParser.add_argument("new_file", help="path where the new files are located")
    argParser.add_argument("combined_file", help="path where the combined results are located")

    args = argParser.parse_args(userArgs)
    currentFileName = args.current_file
    newFileName = args.new_file
    combinedFileName = args.combined_file

    current_data = pd.read_csv(currentFileName)
    headers = current_data.columns.values.tolist()

    keys = headers[0:len(headers)-4]
    new_data = pd.read_csv(newFileName)

    result1 = pd.merge(current_data, new_data, on=keys, how='inner')
    result = result1.rename(columns={'eff_x':'eff_current','eff_y':'eff_new','rocblas-Gflops_x':'rocblas-Gflops_current', 'rocblas-Gflops_y':'rocblas-Gflops_new','us_x':'us_current','us_y':'us_new','counts_x':'counts_current','score_x':'score_current','counts_y':'counts_new','score_y':'score_new','wa_x':'wa_current','wa_y':'wa_new'})

    result['percent gain'] = 100.0 * (result['rocblas-Gflops_new'] - result['rocblas-Gflops_current']) /result['rocblas-Gflops_current']
    result['weighted gain'] = result['percent gain'] * result['wa_new'] / result['rocblas-Gflops_new']
    result['call count'] = result['weighted gain'] / result['percent gain']
    result['overall gain'] = sum(result['weighted gain']) / sum(result['call count'])

    result.to_csv(combinedFileName, header=True, index=False)

    inputFileBaseName = os.path.basename(combinedFileName)
    outputDir = os.path.dirname(combinedFileName)
    namePart, _ = os.path.splitext(inputFileBaseName)
    excelFileName = os.path.join(outputDir, namePart + ".xlsx")

    result.to_excel(excelFileName)

if __name__ == "__main__":
    RunMain()

