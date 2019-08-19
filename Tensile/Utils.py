################################################################################
# Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import ProgressBar

import functools
import sys

class SpinnyThing:
    def __init__(self):
        self.chars = ['|', '/', '-', '\\']
        self.index = 0;
    
    def increment(self, value=1):
        sys.stdout.write('\b' + self.chars[self.index])
        sys.stdout.flush()
        self.index = (self.index + 1) % len(self.chars)

    def finish(self):
        sys.stdout.write('\b*\n')
        sys.stdout.flush()

def iterate_progress(obj, *args, **kwargs):
    try:
        progress = ProgressBar(len(obj))
    except TypeError:
        progress = SpinnyThing()
    for o in obj:
        yield o
        progress.increment()
    progress.finish()

try:
    from tqdm import tqdm
except ImportError:
    tqdm = iterate_progress

def state(obj):
    if hasattr(obj, 'state'):
        return obj.state()

    if hasattr(obj.__class__, 'StateKeys'):
        rv = {}
        for key in obj.__class__.StateKeys:
            attr = key
            if isinstance(key, tuple):
                (key, attr) = key
            rv[key] = state(getattr(obj, attr))
        return rv

    if isinstance(obj, dict):
        return dict([(k, state(v)) for k,v in list(obj.items())])

    if any([isinstance(obj, cls) for cls in [str, int, float]]):
        return obj

    try:
        obj = [state(i) for i in obj]
        return obj
    except TypeError:
        pass

    return obj

def state_key_ordering(cls):
    def tup(obj):
        return tuple([getattr(obj, k) for k in cls.StateKeys])

    def lt(a, b):
        return tup(a) < tup(b)
    def eq(a, b):
        return tup(a) == tup(b)

    cls.__lt__ = lt
    cls.__eq__ = eq

    return functools.total_ordering(cls)

def hash_combine(*objs, **kwargs):
    shift = 1
    if 'shift' in kwargs:
        shift = kwargs['shift']

    if len(objs) == 1:
        objs = objs[0]

    rv = 0
    try:
        it = iter(objs)
        rv = next(it)
        for value in it:
            rv = (rv << shift) ^ value
    except TypeError:
        return objs
    except StopIteration:
        pass
    return rv

def hash_objs(*objs, **kwargs):
    return hash(tuple(objs))

def ceil_divide(numerator, denominator):
    # import pdb
    # pdb.set_trace()
    try:
        if numerator < 0 or denominator < 0:
            raise ValueError
    except ValueError:
        print("ERROR: Can't have a negative register value")
        return 0
    try:
        div = int((numerator+denominator-1) // denominator)
    except ZeroDivisionError:
        print("ERROR: Divide by 0")
        return 0
    return div
    
def roundUpToNearestMultiple(numerator, denominator):
    return ceil_divide(numerator,denominator)*int(denominator)
