################################################################################
#
# Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

# python3 path-to-Tensile-repo/Tensile/tuning/automation/LibraryLogicConvertor.py <path-to-lib-logics> <output-dir>

import os
import argparse
import shutil

# reference/mother lib logics
validLogics={
            'F8SS_NN':    'aquavanjaram_Cijk_Ailk_Bljk_F8SS_BH.yaml',
            'F8F8S_NN':   'aquavanjaram_Cijk_Ailk_Bljk_F8F8S_BH.yaml',
            'F8HS_NN':    'aquavanjaram_Cijk_Ailk_Bljk_F8HS_BH.yaml',
            'F8F8S_SR_NN':'aquavanjaram_Cijk_Ailk_Bljk_F8F8S_SR_BH.yaml',
            'BBS_NN':     'aquavanjaram_Cijk_Ailk_Bljk_BBS_BH.yaml',
            'BSS_NN':     'aquavanjaram_Cijk_Ailk_Bljk_BSS_BH.yaml',            

            'F8SS_NT':    'aquavanjaram_Cijk_Ailk_Bjlk_F8SS_BH.yaml',
            'F8F8S_NT':   'aquavanjaram_Cijk_Ailk_Bjlk_F8F8S_BH.yaml',
            'F8HS_NT':    'aquavanjaram_Cijk_Ailk_Bjlk_F8HS_BH.yaml',
            'F8F8S_SR_NT':'aquavanjaram_Cijk_Ailk_Bjlk_F8F8S_SR_BH.yaml',
            'BBS_NT':     'aquavanjaram_Cijk_Ailk_Bjlk_BBS_BH.yaml',
            'BSS_NT':     'aquavanjaram_Cijk_Ailk_Bjlk_BSS_BH.yaml',

            'F8SS_TN':    'aquavanjaram_Cijk_Alik_Bljk_F8SS_BH.yaml',
            'F8F8S_TN':   'aquavanjaram_Cijk_Alik_Bljk_F8F8S_BH.yaml',
            'F8HS_TN':    'aquavanjaram_Cijk_Alik_Bljk_F8HS_BH.yaml',
            'F8F8S_SR_TN':'aquavanjaram_Cijk_Alik_Bljk_F8F8S_SR_BH.yaml',
            'BBS_TN':     'aquavanjaram_Cijk_Alik_Bljk_BBS_BH.yaml',
            'BSS_TN':     'aquavanjaram_Cijk_Alik_Bljk_BSS_BH.yaml',

            'F8SS_TT':    'aquavanjaram_Cijk_Alik_Bjlk_F8SS_BH.yaml',
            'F8F8S_TT':   'aquavanjaram_Cijk_Alik_Bjlk_F8F8S_BH.yaml',
            'F8HS_TT':    'aquavanjaram_Cijk_Alik_Bjlk_F8HS_BH.yaml',
            'F8F8S_SR_TT':'aquavanjaram_Cijk_Alik_Bjlk_F8F8S_SR_BH.yaml',
            'BBS_TT':     'aquavanjaram_Cijk_Alik_Bjlk_BBS_BH.yaml',
            'BSS_TT':     'aquavanjaram_Cijk_Alik_Bjlk_BSS_BH.yaml'
            }

# family lib logics
conversions={
        'F8SS'    : ['B8SS', 'F8B8SS', 'B8F8SS', 'I8II'],
        'F8F8S'   : ['F8B8B8S', 'B8B8S', 'B8F8B8S'],
        'F8HS'    : ['B8HS', 'F8B8HS', 'B8F8HS'],
        'F8F8S_SR': ['F8B8B8S_SR', 'B8B8S_SR', 'B8F8B8S_SR'],
        'BBS'     : ['HHS'],
        'BSS'     : ['HSS']
        }

# not needed, for reference/debugging
typeIndexToName = {0: "f32_r", 
                   1: "f64_r", 
                   2: "f32_c", 
                   3: "f64_c", 
                   4: "f16_r", 
                   5: "i8_r", 
                   6: "i32_r", 
                   7: "bf16_r", 
                   8: "i8_r", 
                   10: "f8_r", 
                   11: "bf8_r", 
                   12: "f8b8", 
                   13: "b8f8"}

# datatype of each GEMM functions
datatypes={
            'F8SS':  ['  DataType: 10', '  DestDataType: 0', '  ComputeDataType: 0'],
            'B8SS':  ['  DataType: 11', '  DestDataType: 0', '  ComputeDataType: 0'],
            'F8B8SS':['  DataType: 12', '  DestDataType: 0', '  ComputeDataType: 0'],
            'B8F8SS':['  DataType: 13', '  DestDataType: 0', '  ComputeDataType: 0'],
            'I8II':  ['  DataType: 8',  '  DestDataType: 6', '  ComputeDataType: 6'],

            'F8F8S':  ['  DataType: 10', '  DestDataType: 10', '  ComputeDataType: 0'],
            'B8B8S':  ['  DataType: 11', '  DestDataType: 11', '  ComputeDataType: 0'],
            'F8B8B8S':['  DataType: 12', '  DestDataType: 11', '  ComputeDataType: 0'],
            'B8F8B8S':['  DataType: 13', '  DestDataType: 11', '  ComputeDataType: 0'],

            'F8HS':  ['  DataType: 10', '  DestDataType: 4', '  ComputeDataType: 0'],
            'B8HS':  ['  DataType: 11', '  DestDataType: 4', '  ComputeDataType: 0'],
            'F8B8HS':['  DataType: 12', '  DestDataType: 4', '  ComputeDataType: 0'],
            'B8F8HS':['  DataType: 13', '  DestDataType: 4', '  ComputeDataType: 0'],

            # don't need it, non-SR cases will take care of SR. 
            #'F8F8S_SR':  ['  DataType: 10', '  DestDataType: 10', '  ComputeDataType: 0'],
            #'B8B8S_SR':  ['  DataType: 11', '  DestDataType: 11', '  ComputeDataType: 0'],
            #'F8B8B8S_SR':['  DataType: 12', '  DestDataType: 11', '  ComputeDataType: 0'],
            #'B8F8B8S_SR':['  DataType: 13', '  DestDataType: 11', '  ComputeDataType: 0']
            
            'BBS':  ['  DataType: 7', '  DestDataType: 7', '  ComputeDataType: 0'],
            'HHS':  ['  DataType: 4', '  DestDataType: 4', '  ComputeDataType: 0'],

            'BSS':  ['  DataType: 7', '  DestDataType: 0', '  ComputeDataType: 0'],
            'HSS':  ['  DataType: 4', '  DestDataType: 0', '  ComputeDataType: 0']            
          }


def parseArgs():
    argParser = argparse.ArgumentParser()

    h = {"libLogic" : "path to library logics",
         "outDir"   : "Output directory for rocBLAS-bench yaml files"
    }

    argParser.add_argument("libLogic", metavar="logic-file", type=str, help=h["libLogic"])
    argParser.add_argument("outDir", metavar="output-dir", type=str, help=h["outDir"])

    return argParser.parse_args()


# This def reads all the files in the liblogic and only selects the reference yamls (see the table)
def getLogics(liblogic):
    
    logics = []
    
    for yaml in os.listdir(liblogic):
        for val in validLogics.values():
            if (os.path.isfile(os.path.join(liblogic, yaml)) and yaml == val):
                logics.append(yaml)
                break
    return logics

# replaces the string in the file
def updateLogic(filename, old_string, new_string):
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    with open(filename, 'w') as f:
        print('Replacing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)

# converts reference lib logics to the family
def convert(logics, libLogic, outDir):
    
    for yaml in logics:
        print("\n\n <<< wokring on {}>>>".format(yaml))
        for inType in conversions.keys():
            if inType in yaml:
                print(" reference library: {}\n".format(inType))
                for newType in conversions[inType]:

                    # create the new logic file name
                    newLogic = yaml.replace(inType, newType)
                    print(" New item is: {}. Generating: {}".format(newType, newLogic))
                    
                    # copy the original logic file with new name in the output dir
                    newLogicFile = os.path.join(outDir,newLogic)
                    shutil.copy(os.path.join(libLogic,yaml),newLogicFile)
                    
                    # make changes in the new logic file
                    # change kernel names
                    updateLogic(newLogicFile, inType, newType)
                    
                    #change datatypes
                    for i in range(3):
                        updateLogic(newLogicFile, datatypes[inType][i], datatypes[newType][i])                
                break

def main():
    args = parseArgs()    

    # check if lib logic folder exists
    if not os.path.exists(args.libLogic): 
      raise FileNotFoundError("{0} folder does not exist! Check the lib logic path.".format(args.libLogic))

    if not os.path.exists(args.outDir):
      os.makedirs(args.outDir)

    #print(validLogics)

    logics = getLogics(args.libLogic)
    print(" Here are logics to be converted: {}".format(logics))

    convert(logics, args.libLogic, args.outDir)
    
if __name__ == "__main__":
    main()

