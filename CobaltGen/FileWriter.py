import os
import KernelWriter
import SolutionWriter

################################################################################
# File Writer
################################################################################
class FileWriter:

  topDirectory = "/Cobalt/"
  kernelSubdirectory = "/Cobalt/Kernels/"
  kernelPreCompiledSubdirectory = "/Cobalt/Kernels/PreCompiled/"
  solutionSubdirectory = "/Cobalt/Solutions/"


  ##############################################################################
  # ensurePath
  ##############################################################################
  def ensurePath( self, path ):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
      os.makedirs(dirname)


  ##############################################################################
  # constructor
  ##############################################################################
  def __init__( self, outputPath, backend ):
    self.outputPath = outputPath
    self.backend = backend
    self.kernelWriter = KernelWriter.KernelWriter(backend)

    self.ensurePath( self.outputPath + self.topDirectory )
    self.ensurePath( self.outputPath + self.kernelSubdirectory )
    self.ensurePath( self.outputPath + self.kernelPreCompiledSubdirectory )
    self.ensurePath( self.outputPath + self.solutionSubdirectory )


  ##############################################################################
  # writeKernelFiles
  ##############################################################################
  def writeKernelFiles( self, kernelSet ):
    print "status: writing kernel files"

    # main kernel .cpp,.h files
    allKernelsSourceFilePath = self.outputPath + self.topDirectory \
        + "CobaltKernels.cpp"
    allKernelsHeaderFilePath = self.outputPath + self.topDirectory \
        + "CobaltKernels.h"
    allKernelsSourceFile = open(allKernelsSourceFilePath, "w")
    allKernelsHeaderFile = open(allKernelsHeaderFilePath, "w")


    for kernel in kernelSet:
      # open .inl,.h files
      kernelName = self.kernelWriter.getName(kernel)
      kernelSourceFileName = kernelName + ".inl"
      kernelHeaderFileName = kernelName + ".h"
      kernelSourceFilePath = self.outputPath + self.kernelSubdirectory + \
          kernelSourceFileName
      kernelHeaderFilePath = self.outputPath + self.kernelSubdirectory + \
          kernelHeaderFileName
      kernelSourceFile = open(kernelSourceFilePath, "w")
      kernelHeaderFile = open(kernelHeaderFilePath, "w")

      # get kernel file string
      kernelSourceFileString = self.getKernelSourceFileString(kernel)
      kernelHeaderFileString = self.getKernelHeaderFileString(kernel)

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
  def writeSolutionFiles( self, solutionSet ):
    print "status: writing solution files"

    solutionWriter = SolutionWriter.SolutionWriter( self.backend )

    # main solution .cpp,.h files
    allSolutionsSourceFilePath = self.outputPath + self.topDirectory \
        + "CobaltSolutions.cpp"
    allSolutionsHeaderFilePath = self.outputPath + self.topDirectory \
        + "CobaltSolutions.h"
    allSolutionsSourceFile = open(allSolutionsSourceFilePath, "w")
    allSolutionsHeaderFile = open(allSolutionsHeaderFilePath, "w")

    for solution in solutionSet:
      # open file
      solutionName = solutionWriter.getName(solution)
      solutionSourceFileName = solutionName + ".inl"
      solutionHeaderFileName = solutionName + ".h"
      solutionSourceFilePath = self.outputPath + self.solutionSubdirectory + \
          solutionSourceFileName
      solutionHeaderFilePath = self.outputPath + self.solutionSubdirectory + \
          solutionHeaderFileName
      print "NameLength: " + str(len(solutionSourceFilePath))
      solutionSourceFile = open(solutionSourceFilePath, "w")
      solutionHeaderFile = open(solutionHeaderFilePath, "w")

      # get solution file string
      solutionSourceFileString = solutionWriter.getSourceString( solution )
      solutionHeaderFileString = solutionWriter.getHeaderString( solution )

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
  def writeBenchmarkFiles( self, problemSolutionCandidates ):
    print "status: writing benchmark files"
    # write a .h file which creates an array of problem/solution candidates
    pass


  ##############################################################################
  # get source file string
  ##############################################################################
  def getKernelSourceFileString( self, kernel):
    kernelName = self.kernelWriter.getName(kernel)
    fileString = ""
    #fileString += Common.getFileHeader()
    fileString += "#ifndef KERNEL_" + kernelName.upper() + "_INL\n"
    fileString += "#define KERNEL_" + kernelName.upper() + "_INL\n"
    fileString += "\n"
    fileString += "const unsigned int %s_workGroupDim0 = %u;\n" \
        % (kernelName, kernel.tile.workGroupDim0 )
    fileString += "const unsigned int %s_workGroupDim1 = %u;\n" \
        % (kernelName, kernel.tile.workGroupDim1 )
    fileString += "const unsigned int %s_microTileDim0 = %u;\n" \
        % (kernelName, kernel.tile.microTileDim0 )
    fileString += "const unsigned int %s_microTileDim1 = %u;\n" \
        % (kernelName, kernel.tile.microTileDim1 )
    fileString += "const unsigned int %s_unroll = %u;\n" \
        % (kernelName, kernel.unrolls[len(kernel.unrolls)-1])
    fileString += "\n"
    fileString += "const char * const %s_src =\"" % (kernelName)
    fileString += self.kernelWriter.getBody( kernel )
    fileString += "\";\n"
    fileString += "\n"
    fileString += "#else\n"
    fileString += "#pragma message(\"%s was overriden by user kernel.\")\n" \
        % kernelName
    fileString += "#endif\n"
    return fileString


  ##############################################################################
  # get header file string
  ##############################################################################
  def getKernelHeaderFileString( self, kernel ):
    kernelName = self.kernelWriter.getName(kernel)
    fileString = ""
    #fileString += Common.getFileHeader()
    fileString += "#ifndef KERNEL_" + kernelName.upper() + "_H\n"
    fileString += "#define KERNEL_" + kernelName.upper() + "_H\n"
    fileString += "\n"
    fileString += "extern const unsigned int %s_workGroupDim0;\n" % kernelName
    fileString += "extern const unsigned int %s_workGroupDim1;\n" % kernelName
    fileString += "extern const unsigned int %s_microTileDim0;\n" % kernelName
    fileString += "extern const unsigned int %s_microTileDim1;\n" % kernelName
    fileString += "extern const unsigned int %s_unroll;\n" % kernelName
    fileString += "extern const char * const %s_src;\n" % kernelName
    fileString += "#endif\n"
    return fileString

