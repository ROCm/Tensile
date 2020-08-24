################################################################################
# Copyright 2020 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import Component

class Priority(Component):
    """
    Raise/lower workgroup priority.
    """
    pass

class ConstantPriority(Priority):
    """
    Priority implementation which does nothing.
    """
    kernel = {"AggressivePerfMode": False}

    def __call__(self, writer, prio, message=""):
        return ""

class AggressivePriority(Priority):
    """
    Priority implementation which does set the priority.

    Keeps track of the previous value in the instance and only sets priority
    if the new priority is different.
    """
    kernel = {"AggressivePerfMode": True}

    def __init__(self, currentPrio=None):
        self.currentPrio = currentPrio

    def __call__(self, writer, prio, message=""):
        if prio == self.currentPrio:
            return ""

        self.currentPrio = prio
        vars = {"endLine": writer.endLine, "prio": prio, "message": message}

        return "s_setprio {prio} // {message}{endLine}".format_map(vars)
