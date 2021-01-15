import os, subprocess, shlex, shutil, random, pytest
from Tensile import ClientWriter, LibraryIO, Common
from Tensile.SolutionStructs import ProblemType, ProblemSizesMock

# 1. Call TensileCreateLibrary
# 2. Get client instance from ClientExecutable()
# 3. Write some problem types/sizes to ClientParameters with writeClientConfig()
# 4. Run client with small subset of GEMM problems
# $ pytest Tensile/Tests -m integration --capture=tee-sys --builddir /tmp/pytest

# TODO fix TensileCreateLibrary build configs:
#   --library-format=msgpack is currently known to fail
#   --short-file-names is currently known to fail

@pytest.fixture(scope="module")
def getLogicFileDir(builddir):
  prefix = "library/src/blas3/Tensile/Logic/asm_full"
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

  destDir = os.path.join(builddir, "logic_yaml")
  shutil.rmtree(destDir, ignore_errors=True)

  # basically to query the latest zip release weblink, download it and unzip
  # selected files to destination folder
  cmd = """#!/bin/bash
  set -x
  curl --silent "https://api.github.com/repos/ROCmSoftwarePlatform/rocBLAS/releases/latest" | tee log
  weblink=$(grep -oP '(?<="zipball_url": ")[a-zA-Z:/\.\-0-9]*' log)
  wget -nc $weblink
  archive=$(basename $weblink)
  rootDir=$(zipinfo -1 $archive | head -n 1)
  """
  for schedule in list(testData.keys()):
    for f in testData[schedule]:
      dir = os.path.join(destDir, schedule)
      Common.ensurePath(dir)
      cmd += "unzip -j -d %s -x $archive ${rootDir}%s\n"%(dir, os.path.join(prefix,f))

  scriptFile = os.path.join(builddir,"get_logic.sh")
  with open(scriptFile, "w") as file: file.write(cmd)
  os.chmod(scriptFile, 0o777)
  subprocess.run(scriptFile, cwd=builddir, check=True)

  return destDir

def isSkippedTest(testYamls, mergeFiles, libraryFormat, shortNames, legacyComponents):
  if testYamls == "pre_checkin":
    # for more extensive tests,
    # we run only on single combination of build params
    if (mergeFiles           == True
        and legacyComponents == False
        and shortNames       == False
        and libraryFormat    == "yaml"
        and legacyComponents == False):
      return False
    else:
      return True
  elif testYamls == "quick":
    return False
  else:
    assert(False) # should've caught all params already

def str2bool(mergeFiles, shortNames, legacyComponents):
  return (True if mergeFiles=="mergeFiles" else False,
          True if shortNames=="shortNames" else False,
          True if legacyComponents=="legacyComponents" else False)

@pytest.mark.parametrize("testYamls",         ["quick", "pre_checkin"])
@pytest.mark.parametrize("mergeFiles",        ["mergeFiles", "noMergeFiles"])
@pytest.mark.parametrize("libraryFormat",     ["yaml", pytest.param("msgpack", marks=pytest.mark.xfail)])
@pytest.mark.parametrize("shortNames",        [pytest.param("shortNames", marks=pytest.mark.xfail), "noShortName"])
@pytest.mark.parametrize("legacyComponents",  ["legacyComponents", "noLegacyComponents"])
def test_integration(useGlobalParameters, builddir, getLogicFileDir,
                     testYamls, mergeFiles, libraryFormat, shortNames, legacyComponents):
  mergeFiles, shortNames, legacyComponents = str2bool(mergeFiles, shortNames, legacyComponents)

  if isSkippedTest(testYamls, mergeFiles, libraryFormat, shortNames, legacyComponents):
    pytest.skip("only run pre_checkin test in certain combo")

  logicFileDir = os.path.join(getLogicFileDir, testYamls)
  outputDir    = os.path.join(builddir, "lib")

  with useGlobalParameters(OutputPath=outputDir,
                           WorkingPath=outputDir,
                           MergeFiles=mergeFiles,
                           LibraryFormat=libraryFormat,
                           LegacyComponents=legacyComponents,
                           ShortNames=shortNames,
                           GenerateManifestAndExit=False
                           ):
    Common.ensurePath(outputDir)

    createLibraryScript = ClientWriter.getBuildNewClientLibraryScript(outputDir, logicFileDir)
    subprocess.run(shlex.split(createLibraryScript), cwd=outputDir, check=True)

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

    forBenchmark = False
    enableTileSelection = False
    returncode = ClientWriter.runClient(logicFileDir, forBenchmark, enableTileSelection, clientParametersPaths)

    assert(returncode == 0)