import Kernel
import Solution

################################################################################
# File Writer
################################################################################
class FileWriter:

  topDirectory = "/Cobalt/"
  kernelSubdirectory = "/Cobalt/Kernels/"
  kernelPreCompiledSubdirectory = "/Cobalt/Kernels/PreCompiled/"
  solutionSubdirectory = "/Cobalt/Solutions/"

  ##############################################################################
  # constructor
  def __init__( self, outputPath, language ):
    self.outputPath = outputPath
    self.language = language

  ##############################################################################
  # writeKernelFiles
  ##############################################################################
  def writeKernelFiles( kernelSet ):

    # main kernel .cpp,.h files
    allKernelsSourceFilePath = outputPath + topDirectory + "CobaltKernels.cpp"
    allkernelsHeaderFilePath = outputPath + topDirectory + "CobaltKernels.h"
    allKernelsSourceFile = open(allKernelSourceFilePath, "w")
    allKernelsHeaderFile = open(allKernelHeaderFilePath, "w")

    for kernel in kernelSet:
      # open .inl,.h files
      kernelName = Kernel.getName(kernel)
      kernelSourceFileName = kernelName + ".inl"
      kernelHeaderFileName = kernelName + ".h"
      kernelSourceFilePath = outputPath + kernelSubdirectory + \
          kernelSourceFileName
      kernelHeaderFilePath = outputPath + kernelSubdirectory + \
          kernelHeaderFileName
      kernelSourceFile = open(kernelSourceFilePath, "w")
      kernelHeaderFile = open(kernelHeaderFilePath, "w")

      # get kernel file string
      kernelSourceFileString = Kernel.getSourceFileString(kernel, language)
      kernelHeaderFileString = Kernel.getHeaderFileString(kernel, language)

      # write kernel file string to file
      kernelSourceFile.write( kernelSourceFileString )
      kernelHeaderFile.write( kernelHeaderFileString )
      kernelSourceFile.close()
      kernelHeaderFile.close()

      # write the main CobaltKernels.h,cpp file which #includes these
      allKernelsSourceFile.write( "#include \"" + kernelSourceFileName + "\"")
      allKernelsHeaderFile.write( "#include \"" + kernelHeaderFileName + "\"")

    allKernelsSourceFile.close()
    allKernelsHeaderFile.close()


  ##############################################################################
  # writeSolutionFiles
  ##############################################################################
  def writeSolutionFiles( solutionSet ):

    # main solution .cpp,.h files
    allSolutionsSourceFilePath = outputPath + topDirectory \
        + "CobaltSolutions.cpp"
    allSolutionsHeaderFilePath = outputPath + topDirectory \
        + "CobaltSolutions.h"
    allSolutionsSourceFile = open(allSolutionsSourceFilePath, "w")
    allSolutionsHeaderFile = open(allSolutionsHeaderFilePath, "w")

    for solution in solutionSet:
      # open file
      solutionName = Solution.getName(solution)
      solutionSourceFileName = solutionName + ".inl"
      solutionHeaderFileName = solutionName + ".h"
      solutionSourceFilePath = outputPath + solutionSubdirectory + \
          solutionSourceFileName
      solutionHeaderFilePath = outputPath + solutionSubdirectory + \
          solutionHeaderFileName
      solutionSourceFile = open(solutionSourceFilePath, "w")
      solutionHeaderFile = open(solutionHeaderFilePath, "w")

      # get solution file string
      solutionSourceFileString = Solution.getSourceString( solution, language )
      solutionHeaderFileString = Solution.getHeaderString( solution, language )

      # write solution file string to file
      solutionSourceFile.write( solutionSourceFileString )
      solutionHeaderFile.write( solutionHeaderFileString )
      solutionSourceFile.close()
      solutionHeaderFile.close()

      # write the main CobaltSolutions.h,cpp file which #includes these
      allSolutionsSourceFile.write( "#include \"" \
          + solutionSourceFileName + "\"")
      allSolutionsHeaderFile.write( "#include \"" \
          + solutionHeaderFileName + "\"")

    allSolutionsSourceFile.close()
    allSolutionsHeaderFile.close()


  ##############################################################################
  # writeBenchmarkFiles
  ##############################################################################
  def writeBenchmarkFiles( problemSolutionCandidates ):
    # write a .h file which creates an array of problem/solution candidates
    pass



