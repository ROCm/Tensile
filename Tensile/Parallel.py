################################################################################
# Copyright (C) 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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
import itertools
import sys


def CPUThreadCount(enable=True):
  from .Common import globalParameters
  if not enable or globalParameters["CpuThreads"] == 0:
    return 0
  else:
    cpu_count = len(os.sched_getaffinity(0))
    cpuThreads = globalParameters["CpuThreads"]
    if cpuThreads < 0:
        return cpu_count*abs(cpuThreads)
    return min(cpu_count, cpuThreads)

def starmap_apply(item):
  func, item = item
  return func(*item)

def apply_print_exception(item, *args):
  #print(item, args)
  try:
    if len(args) > 0:
      func = item
      args = args[0]
      return func(*args)
    else:
      func, item = item
      return func(item)
  except Exception:
    import traceback
    traceback.print_exc()
    raise
  finally:
    sys.stdout.flush()
    sys.stderr.flush()

def ProcessingPool(enable=True):
  import multiprocessing
  import multiprocessing.dummy

  threadCount = CPUThreadCount()

  if (not enable) or threadCount <= 1:
    return multiprocessing.dummy.Pool(1)

  return multiprocessing.Pool(threadCount)

def ParallelMap(function, objects, message="", enable=True, method=None):
  """
  Generally equivalent to list(map(function, objects)), possibly executing in parallel.

    message: A message describing the operation to be performed.
    enable: May be set to false to disable parallelism.
    method: A function which can fetch the mapping function from a processing pool object.
        Leave blank to use .map(), other possiblities:
           - `lambda x: x.starmap` - useful if `function` takes multiple parameters.
           - `lambda x: x.imap` - lazy evaluation
           - `lambda x: x.imap_unordered` - lazy evaluation, does not preserve order of return value.
  """
  from .Common import globalParameters
  threadCount = CPUThreadCount(enable)
  pool = ProcessingPool(enable)

  if threadCount <= 1 and globalParameters["ShowProgressBar"]:
    # Provide a progress bar for single-threaded operation.
    # This works for method=None, and for starmap.
    mapFunc = map
    if method is not None:
      # itertools provides starmap which can fill in for pool.starmap.  It provides imap on Python 2.7.
      # If this works, we will use it, otherwise we will fallback to the "dummy" pool for single threaded
      # operation.
      try:
        mapFunc = method(itertools)
      except NameError:
        mapFunc = None

    if mapFunc is not None:
      from . import Utils
      return list(mapFunc(function, Utils.tqdm(objects, message)))

  mapFunc = pool.map
  if method: mapFunc = method(pool)

  objects = zip(itertools.repeat(function), objects)
  function = apply_print_exception

  countMessage = ""
  try:
    countMessage = " for {} tasks".format(len(objects))
  except TypeError: pass

  if message != "": message += ": "

  print("{0}Launching {1} threads{2}...".format(message, threadCount, countMessage))
  sys.stdout.flush()
  rv = mapFunc(function, objects)
  print("{0}Done.".format(message))
  sys.stdout.flush()
  pool.close()
  return rv