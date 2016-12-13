################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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

from Common import *
import ReadYAML
import BenchmarkProblems
import Analyze
import Library
import Client
import BenchmarkClient

def executeStepsInConfig( config ):
  # ensure working directory exists
  ensurePath(globalParameters["WorkingPath"])


  if "Parameters" in config:
    assignGlobalParameters( config["Parameters"] )
    print ""

  if "BenchmarkProblems" in config:
    BenchmarkProblems.main( config["BenchmarkProblems"] )
    print ""

  if "Analyze" in config:
    Analyze.main( config["Analyze"] )
    print ""

  if "Library" in config:
    Library.main( config["Library"] )
    print ""

  if "Client" in config:
    Client.main( config["Client"] )
    print ""

  if "BenchmarkClient" in config:
    BenchmarkClient.main( config["BenchmarkClient"] )
    print ""




################################################################################
# Tensile - Main
################################################################################
if len(sys.argv) < 2:
  printExit("Usage: python Tensile.py config.yaml")
else:
  configPath = os.path.realpath( sys.argv[1] )
  print("Tensile::Main ConfigFile: %s" % (configPath) )
  config = ReadYAML.readConfig( configPath )
  executeStepsInConfig( config )
  sys.exit(0)

