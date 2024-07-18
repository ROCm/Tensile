################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

import itertools
import os

from joblib import Parallel, delayed

def CPUThreadCount(enable=True):
  from .Common import globalParameters
  if not enable:
    return 1
  else:
    if os.name == "nt":
      cpu_count = os.cpu_count()
    else:
      cpu_count = len(os.sched_getaffinity(0))
    cpuThreads = globalParameters["CpuThreads"]
    if cpuThreads < 1:
        return min(cpu_count, 64) # Max build threads to avoid out-of-memory
    return min(cpu_count, cpuThreads)

def OverwriteGlobalParameters(newGlobalParameters):
  from . import Common
  Common.globalParameters.clear()
  Common.globalParameters.update(newGlobalParameters)

def pcallWithGlobalParamsMultiArg(f, args, newGlobalParameters):
  OverwriteGlobalParameters(newGlobalParameters)
  return f(*args)

def pcallWithGlobalParamsSingleArg(f, arg, newGlobalParameters):
  OverwriteGlobalParameters(newGlobalParameters)
  return f(arg)

def ParallelMap(function, objects, message="", enable=True, multiArg=True, verbose=0):
  """
  Generally equivalent to list(map(function, objects)), possibly executing in parallel.

    message: A message describing the operation to be performed.
    enable: May be set to false to disable parallelism.
    multiArg: True if objects represent multiple arguments
                (differentiates multi args vs single collection arg)
  """
  from .Common import globalParameters
  from . import Utils
  threadCount = CPUThreadCount(enable)
  
  if threadCount <= 1 and globalParameters["ShowProgressBar"]:
    # Provide a progress bar for single-threaded operation.
    return list(map(lambda objs: function(*objs), Utils.tqdm(objects, msg=message)))
  
  message += f": {threadCount} threads"
  try:
    message += f", {len(objects)} tasks"
  except TypeError: pass
  
  pcall = pcallWithGlobalParamsMultiArg if multiArg else pcallWithGlobalParamsSingleArg
  inputs = list(zip(objects, itertools.repeat(globalParameters)))
  pargs = Utils.tqdm(inputs, msg=message)
  rv = Parallel(n_jobs=threadCount, verbose=verbose)(delayed(pcall)(function, a, params) for a, params in pargs)
  
  return rv
