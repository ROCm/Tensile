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

import cProfile
import pstats
import os

from pathlib import Path
from datetime import datetime, timezone
from typing import Callable

def profile(func: Callable):
    def wrapper(*args, **kwargs):
        dt = datetime.now(timezone.utc)
        pid = str(os.getpid())
        filename = f"{func.__name__}-{dt.strftime('%Y-%m-%dT%H-%M-%SZ')}.{pid}.prof"
        path = Path().cwd()/f"profiling-results-{dt.strftime('%Y-%m-%d')}"
        path.mkdir(exist_ok=True)

        prof = cProfile.Profile()
        output = prof.runcall(func, *args, **kwargs)
        result = pstats.Stats(prof)
        result.sort_stats(pstats.SortKey.TIME)
        result.dump_stats(path/filename)
        return output
    return wrapper
