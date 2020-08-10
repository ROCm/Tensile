################################################################################
# Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
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

from . import Properties

class HardwarePredicate(Properties.Predicate):
    @classmethod
    def FromISA(cls, isa):
        gfxArch = 'gfx'+''.join(map(str, isa))
        return cls("AMDGPU", value=cls("Processor", value=gfxArch))

    @classmethod
    def FromHardware(cls, isa, cuCount=None):
        gfxArch = 'gfx'+''.join(map(str, isa))
        if cuCount == None:
            return cls("AMDGPU", value=cls("Processor", value=gfxArch))
        else:
            return cls("AMDGPU", value=cls.And([cls("Processor", value=gfxArch),
                                                cls("CUCount", value=cuCount)]))

    def __lt__(self, other):
        # Use superclass logic for TruePreds
        if other.tag == 'TruePred' or self.tag == 'TruePred':
            return super().__lt__(other)

        # Compute unit counts are embedded as 'And' with
        # 'Processor' and 'ComputeUnitCount' as children
        if self.value.tag == 'And':
            myAndPred = self.value
            myProcPred = next(iter(x for x in myAndPred.value if x.tag == "Processor"), None)
            myCUPred = next(iter(x for x in myAndPred.value if x.tag == "CUCount"), None)
            myCUCount = myCUPred.value if myCUPred != None else 0
        else:
            myProcPred = self.value
            myCUCount = 0

        if other.value.tag == 'And':
            otherAndPred = other.value
            otherProcPred = next(iter(x for x in otherAndPred.value if x.tag == "Processor"), None)
            otherCUPred = next(iter(x for x in otherAndPred.value if x.tag == "CUCount"), None)
            otherCUCount = otherCUPred.value if otherCUPred != None else 0
        else:
            otherProcPred = other.value
            otherCUCount = 0

        # If CU properties are empty, then compare processor predicates
        if myCUCount == otherCUCount == 0:
            # Make sure that we have valid processor preds
            assert myProcPred != None and otherProcPred != None, "Missing processor predicate"
            assert myProcPred.tag == otherProcPred.tag == "Processor", "Invalid processor predicate"

            # Downgrade to base class so that we don't recurse
            myProcPred.__class__ = otherProcPred.__class__ = Properties.Predicate
            return myProcPred < otherProcPred

        # Higher priority given to higher CU count
        return myCUCount > otherCUCount
