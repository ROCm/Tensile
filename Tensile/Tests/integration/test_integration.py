import os, subprocess, shlex, random, pytest
from Tensile import ClientWriter, LibraryIO, Common
from Tensile.SolutionStructs import ProblemType, Problem

# 1. Call TensileCreateLibrary
# 2. Get client instance from ClientExecutable()
# 3. Write some problem types/sizes to ClientParameters with writeClientConfig()
# 4. Run client with small subset of GEMM problems
# $ pytest Tensile/Tests -m integration --capture=tee-sys --builddir /tmp/pytest

def getLogicFileDir(baseDir, schedule):

  prefix = "https://github.com/ROCmSoftwarePlatform/rocBLAS/raw/develop/library/src/blas3/Tensile/Logic/asm_full/"
  testData = {
    "pre_checkin" : [
      "arcturus_Cijk_Ailk_Bjlk_SB.yaml",
      "vega20_Cijk_Ailk_Bjlk_SB.yaml",
      "vega10_Cijk_Ailk_Bjlk_SB.yaml",
      "arcturus_Cijk_Ailk_Bjlk_BBH.yaml",
      "arcturus_Cijk_Ailk_Bjlk_HBH.yaml",
      "vega20_Cijk_Ailk_Bjlk_HBH.yaml",
      "vega10_Cijk_Ailk_Bjlk_HBH.yaml",
      "hip_Cijk_Ailk_Bjlk_CB.yaml"
    ],
    "quick" : [
      "arcturus_Cijk_Ailk_Bjlk_SB.yaml",
      "vega20_Cijk_Ailk_Bjlk_SB.yaml",
      "vega10_Cijk_Ailk_Bjlk_SB.yaml"
    ]
  }

  files = [prefix + e for e in testData[schedule] ]
  destDir = os.path.join(baseDir, "logic_yaml", schedule)
  Common.ensurePath(destDir)

  import glob
  for f in glob.glob(destDir+"/*"):
    print("file: ", f)
    if os.path.basename(f) not in testData[schedule]:
      print("removing file: ", f)
      os.remove(f)

  cmd = list()
  cmd.append("wget")
  cmd.extend(files)
  cmd.extend(["-nc"])
  cmd.extend(["-P", destDir])
  ret = subprocess.run(cmd)
  assert(ret.returncode==0)

  # pytest.set_trace()
  return destDir

# Mock class
class ProblemSizesMock:
  def __init__(self, exactLogic):
    problems = [probSolPair[0] for probSolPair in exactLogic]
    self.problems = [Problem(prob) for prob in problems]

@pytest.mark.parametrize("mergeFiles",        [False, True])
@pytest.mark.parametrize("legacyComponents",  [False, True])
@pytest.mark.parametrize("libraryFormat",     ["yaml", pytest.param("msgpack", marks=pytest.mark.xfail)])
@pytest.mark.parametrize("testYamls",         ["quick", "pre_checkin"])
def test_integration(useGlobalParameters, builddir, testYamls, mergeFiles, libraryFormat, legacyComponents):
  if testYamls == "pre_checkin":
    if not (mergeFiles == True and libraryFormat == "yaml"): pytest.skip("only run pre_checkin test in certain combo")

  logicFileDir = getLogicFileDir(builddir, testYamls)
  outputDir    = os.path.join(builddir, "lib")

  # pytest.set_trace()
  with useGlobalParameters(OutputPath=outputDir,
                           WorkingPath=outputDir,
                           MergeFiles=mergeFiles,
                           LibraryFormat=libraryFormat,
                           LegacyComponents=legacyComponents,
                           CMakeBuildType="Debug"
                           ):
    Common.ensurePath(outputDir)

    # Note: this step is needed for retrieving code object and tensile library yaml paths
    # TODO Do away with this workaround. TensileCreateLibrary.py should take function arguments and return values directly
    #      Refactor TensileCreateLibrary (not .py) to take CLI args instead and pass to underlying python script
    Common.globalParameters["GenerateManifestAndExit"] = True
    createLibraryScript = ClientWriter.getBuildNewClientLibraryScript(outputDir, logicFileDir)
    subprocess.run(shlex.split(createLibraryScript), cwd=outputDir)
    ###

    Common.globalParameters["GenerateManifestAndExit"] = False
    createLibraryScript = ClientWriter.getBuildNewClientLibraryScript(outputDir, logicFileDir)
    subprocess.run(shlex.split(createLibraryScript), cwd=outputDir)

    coList = []
    libList = []
    coExt = "co"
    libExt = "yaml" if libraryFormat == "yaml" else "dat"
    with open(os.path.join(outputDir,"library","TensileManifest.txt"), "r") as f:
      lines = f.read().split("\n")
      coList = [line for line in lines if coExt in line]
      libList = [line for line in lines if libExt in line]

    logicFiles = [os.path.join(logicFileDir, f) for f in os.listdir(logicFileDir) \
      if (os.path.isfile(os.path.join(logicFileDir, f)) and os.path.splitext(f)[1]==".yaml")]

    clientParametersPaths = []
    isaStr = "".join([str(e) for e in Common.globalParameters["CurrentISA"]])
    for logicFileName in logicFiles:
      (scheduleName, _, problemType, _, _, exactLogic, _, newLibrary, archName) = LibraryIO.readLibraryLogicForSchedule(logicFileName)
      problemSizes = ProblemSizesMock(random.sample(exactLogic, min(len(exactLogic), 16))) # sample at most 16 problems
      if isaStr in archName:
        clientParametersPaths.append(ClientWriter.writeClientConfig(
                                      forBenchmark=False,
                                      solutions=None,
                                      problemSizes=problemSizes,
                                      stepName=str(ProblemType(problemType)),
                                      stepBaseDir=outputDir,
                                      newLibrary=newLibrary,
                                      configBase="ClientParameters_%s_%s"%(scheduleName, str(ProblemType(problemType))),
                                      codeObjectFiles=coList,
                                      tileAwareSelection=False,
                                      libraryFile=libList[0]))
    # pytest.set_trace()
    forBenchmark = False
    enableTileSelection = False
    returncode = ClientWriter.runClient(logicFileDir, forBenchmark, enableTileSelection, clientParametersPaths)

    assert(returncode == 0)