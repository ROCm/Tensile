# Gets called by CMake
from Common import *

import sys


################################################################################
# Parse Command Line Arguments
################################################################################
if len(sys.argv) < 9:
  print "Usage: python LibraryWriter.py LogicPath OutputPath Backend MergeFiles ShortFileNames ShortFileNames PrintDebug"
  for arg in sys.argv:
    print arg

  assignGlobalParameters({})


