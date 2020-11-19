
if __name__ == "__main__":
    print("This file can no longer be run as a script.  Run 'Tensile/bin/TensileBenchmarkCluster' instead.")
    exit(1)

import yaml
import shlex, subprocess
import sys
import os
import re
import itertools
import datetime
import time
import copy
import math
import argparse
import mgzip
from .Common import HR
from Tensile.Utilities.merge import mergePartialLogics

BenchmarkParameters = {}

def readConfigFile(configFile):
    with open(configFile) as f:
        data = yaml.safe_load(f)
    return data

# data: a loaded .yaml file
# returns: a list of yaml files that
# are differentiated by the benchmark problem
def splitByProblem(data):
    rv = []
    problemKey = "BenchmarkProblems"
    for i in range(len(data[problemKey])):
        result = {}
        for k in data.keys():
            if k == problemKey:
                result[k] = [copy.deepcopy(data[k][i])]
            else:
                result[k] = copy.deepcopy(data[k])
        rv.append(result)
    return rv

# data: a loaded .yaml file, containing one benchmark problem section
# returns: a list of yaml files that are differentiated by the
# benchmark groupings
def splitByBenchmarkGroup(data):
    rv = []
    problemKey = "BenchmarkProblems"

    assert len(data[problemKey]) == 1, "Config file must have one BenchmarkProblems group"

    benchmarkProblems = data[problemKey][0]

    # Find the index of the problem type group
    problemIdx = -1
    for i in range(len(benchmarkProblems)):
        if("OperationType" in benchmarkProblems[i].keys()):
            problemIdx = i
            break

    assert problemIdx != -1, "Could not find problem type group"

    # Split files on the benchmark group
    for i in range(len(benchmarkProblems)):
        if i != problemIdx:
            result = {}
            for k in data.keys():
                if k == problemKey:
                    # Take only the problem group and one benchmarkgroup
                    result[k] = [[copy.deepcopy(benchmarkProblems[problemIdx]), \
                                  copy.deepcopy(benchmarkProblems[i])]]
                else:
                    # copy other sections verbatim
                    result[k] = copy.deepcopy(data[k])
            rv.append(result)
    return rv

# data: a loaded .yaml file, containing one benchmark problem section
# and one benchmark group section
# returns: a list of yaml files that are differentiated by the
# benchmark sizes
def splitByBenchmarkSizes(data, numSizes=1):
    rv = []
    problemKey = "BenchmarkProblems"

    assert len(data[problemKey]) == 1, "Config file must have one BenchmarkProblems group"

    benchmarkProblems = data[problemKey][0]

    # Find the index of the problem type group
    # and the benchmark group
    problemIdx = -1
    benchmarkIdx = -1
    for i in range(len(benchmarkProblems)):
        groupKeys = benchmarkProblems[i].keys()
        if("OperationType" in groupKeys):
            problemIdx = i
        elif("BenchmarkFinalParameters" in groupKeys):
            benchmarkIdx = i

    assert len(benchmarkProblems) == 2 \
           and problemIdx is not -1 \
           and benchmarkIdx is not -1, \
           "Config file must have one ProblemType group and one Benchmark group"

    # Grab the problem sizes from the Benchmark group
    benchmarkGroup = benchmarkProblems[benchmarkIdx]
    assert "ProblemSizes" in benchmarkGroup["BenchmarkFinalParameters"][0] \
            and len(benchmarkGroup["BenchmarkFinalParameters"][0]["ProblemSizes"]), \
            "Benchmark group must have non-empty ProblemSizes"

    problemSizesGroup = benchmarkGroup["BenchmarkFinalParameters"][0]["ProblemSizes"]
    problemSizesCount = len(problemSizesGroup)

    numFiles = math.ceil(problemSizesCount / numSizes)

    # Split files on the benchmark sizes
    for i in range(numFiles):
        result = {}
        for k in data.keys():
            if k == problemKey:

                # Create a new benchmark group that has all the old keys but split the sizes.
                newBenchmarkGroup = {}
                for bk in benchmarkGroup.keys():
                    if bk == "BenchmarkFinalParameters":
                        newBenchmarkGroup[bk] = [ {"ProblemSizes": [] } ]
                        for j in range(i*numSizes, min((i+1)*numSizes, problemSizesCount)):
                            newBenchmarkGroup[bk][0]["ProblemSizes"].append(copy.deepcopy(problemSizesGroup[j]))
                    else:
                        newBenchmarkGroup[bk] = copy.deepcopy(benchmarkGroup[bk])

                result[k] = [[copy.deepcopy(benchmarkProblems[problemIdx]), copy.deepcopy(newBenchmarkGroup)]]
            else:
                result[k] = copy.deepcopy(data[k])
        rv.append(result)
    return rv

# filePath: Name of the file (can be a path)
# suffix: Add a suffix of _## to fileName by default
# formatting: How the stuffix string is to be treated.
# If more indices are needed, you can increase them on the formatting.
def appendFileNameSuffix(filePath, suffix, formatting="{:02}"):
    root, ext = os.path.splitext(filePath)
    suffixString = ("_" + formatting).format(suffix)
    return root + suffixString + ext

def ensurePath( path ):
    try:
        os.makedirs(path)
    except OSError:
        pass
    return path

def splitBenchmark(configFile, numSizes=1):
    data = readConfigFile(configFile)
    benchmarksByProblem = splitByProblem(data)
    benchmarksByGroup = []
    benchmarksBySize = []
    for problem in benchmarksByProblem:
        benchmarksByGroup += splitByBenchmarkGroup(problem)
    for group in benchmarksByGroup:
        benchmarksBySize += splitByBenchmarkSizes(group, numSizes)

    return benchmarksBySize

def makeExecutable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)

def writeLine(text):
    return str("%s\n") % text

def writeComment(text):
    return writeLine(str("#") + str(text))

# filePath: full file path of the new script
# Creates the script that is to run on the worker node.
# E.g. slurm:
def createBenchmarkTaskScript(filePath):

    # Header
    result = writeComment("!/bin/bash") \
        + writeComment("Benchmark task script") \
        + writeComment("To be run by SLURM worker node")

    # Funcs
    result  \
        += writeLine("usage() { echo \"Usage: $0 [-i <path_to_docker_image>] [-r <result_dir>] [-t <task_dir>]\" 1>&2; exit 1; }\n") \
        + writeLine("failAndExit() { echo \"FAILED: $1\" 1>&2; exit 1; }\n")

    # Args
    result += \
          writeComment("Parse arguments") \
        + writeLine("while getopts i:r:t: flag") \
        + writeLine("do") \
        + writeLine("    case \"${flag}\" in") \
        + writeLine("        i) dockerImagePath=${OPTARG};; ") \
        + writeLine("        r) resultDir=${OPTARG};;") \
        + writeLine("        t) taskDir=${OPTARG};;") \
        + writeLine("        *) usage;;") \
        + writeLine("    esac") \
        + writeLine("done") \
        + writeLine("[ -z \"${dockerImagePath}\" ] || [ -z \"${resultDir}\" ]  || [ -z \"${taskDir}\" ] && usage;") \
        + writeLine("echo \"Docker image path: $dockerImagePath\";") \
        + writeLine("echo \"Path to task: $taskDir\"; ") \
        + writeLine("echo \"Path to result: $resultDir\"; ") \
        + writeLine("")

    # Import docker image
    result += \
          writeComment("Import docker image") \
        + writeComment("Successful output is \"Loaded image: imageName:TAG\"") \
        + writeLine("echo \"Loading docker image...\"") \
        + writeLine("dockerLoadOutput=`docker load < $dockerImagePath || failAndExit \"Importing docker image\"`") \
        + writeLine("echo \"Docker Load Result: $dockerLoadOutput\"") \
        + writeLine("[ -z \"${dockerLoadOutput}\" ] && failAndExit \"Loading docker image $dockerImagePath\"") \
        + writeLine("") \
        + writeComment("Split the output on ':' or spaces") \
        + writeComment("Capture the docker image name and tag") \
        + writeLine("outputSplit=(${dockerLoadOutput//:/ })") \
        + writeLine("dockerName=${outputSplit[2]}") \
        + writeLine("dockerTag=${outputSplit[3]} ") \
        + writeLine("") \
        + writeComment("Find docker image record with correct name and tag") \
        + writeLine("dockerImageQuery=`docker images | grep -i \"$dockerName\" | grep -i \"$dockerTag\"`") \
        + writeLine("[ -z \"${dockerImageQuery}\" ] && failAndExit \"Finding docker image record for $dockerName:$dockerTag\"") \
        + writeLine("") \
        + writeComment("Get docker image ID") \
        + writeLine("outputSplit=(${dockerImageQuery//   / })") \
        + writeLine("dockerImageId=${outputSplit[2]}") \
        + writeLine("[ -z \"${dockerImageId}\" ] && failAndExit \"Finding docker image id for $dockerName:$dockerTag\"") \
        + writeLine("echo \"Loaded Image ID: ${dockerImageId}\"") \
        + writeLine("") \

    # Invoke task
    result += \
          writeComment("Check the task dir exists") \
        + writeLine("[ -d \"${taskDir}\" ] || failAndExit \"Finding task directory $taskDir\"") \
        + writeLine("") \
        + writeComment("Run container with mounted taskDir") \
        + writeLine("echo \"Running container... (this might take some time):\"") \
        + writeLine("runCmd=\"docker run --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v \"$taskDir\":\"/TaskDir\" -v \"$resultDir\":\"/ResultDir\" $dockerImageId\"") \
        + writeLine("echo \"$runCmd\"") \
        + writeLine("$runCmd") \
        + writeLine("echo \"Done!\"")

    with open(filePath, "w") as f:
        f.write(result)

    makeExecutable(filePath)

# baseImage: remote URL to rocm build
# dockerFilePath: path to docker file for slurm tensile build
# tag: name:TAG of new image
# outDir: where to export the new image
# tensile<Fork/Branch/Commit>: tensile code to use on github
def createTensileBenchmarkContainer(baseImage, dockerFilePath, tag, outDir, tensileFork, tensileBranch, tensileCommit):

    # Save stdout and stderr to file
    with open(os.path.join(outDir, "containerBuildLog.log"), 'w') as logFile:

        # Docker build command
        dockerFileName = os.path.basename(dockerFilePath)
        dockerFileDir = os.path.dirname(dockerFilePath)
        buildCmd = str("\
            docker build \
            -t %s \
            --pull -f %s \
            --build-arg user_uid=$UID \
            --build-arg base_image=%s \
            --build-arg tensile_fork=%s \
            --build-arg tensile_branch=%s \
            --build-arg tensile_commit=%s \
            . ") % (tag, dockerFilePath, baseImage, tensileFork, tensileBranch, tensileCommit)

        # Build container and save output streams
        print("Building docker image: %s ..." % tag)
        print(buildCmd)
        subprocess.check_call(shlex.split(buildCmd), stdout=logFile, stderr=logFile)
        print("Done building docker image!")

        # Docker save command
        imageBaseName = tag.split(':')[0]
        archivePath = os.path.join(outDir, imageBaseName + str(".tar.gz"))
        saveCmd = str("docker save %s") % (tag)

        # Docker will save .tar binary to stdout as long as it's not attached to console.
        # Pipe stdout into mgzip to get smaller .tar.gz archive
        print("Saving docker image: %s  to %s ..." % (tag, archivePath))
        with mgzip.open(archivePath, 'wb') as zipFile:
            with subprocess.Popen(shlex.split(saveCmd), stdout=subprocess.PIPE, stderr=logFile) as proc:
                zipFile.write(proc.stdout.read())
        print("Done saving docker image!")

# One config per task
def createClusterBenchmarkTasks(configsDir):

    configFiles = [f for f in os.listdir(configsDir) if os.path.isfile(os.path.join(configsDir, f))]

    # Move each individual config into it's own subfolder and create the benchmark script
    for f in configFiles:
        (baseFileName, _) = os.path.splitext(f)
        configSubdir = os.path.join(configsDir, baseFileName)
        ensurePath(configSubdir)
        os.rename(os.path.join(configsDir, f), os.path.join(configSubdir, f))
        createBenchmarkTaskScript(os.path.join(configSubdir, baseFileName + ".sh"))

def createClusterMasterScript(filePath):
    # Header
    result = writeComment("!/bin/bash") \
        + writeComment("Master job script") \
        + writeComment("This is the master script that will be run with sbatch")

    # Funcs
    result  \
        += writeLine("usage() { echo \"Usage: $0 [-i <mount_image_dir>] [-r <mount_results_dir>] [-t <mount_tasks_dir>] \" 1>&2; exit 1; }\n") \
        + writeLine("failAndExit() { echo \"FAILED: $1\" 1>&2; exit 1; }\n")

    # Args
    result += \
          writeComment("Parse arguments") \
        + writeLine("while getopts i:r:t: flag") \
        + writeLine("do") \
        + writeLine("    case \"${flag}\" in") \
        + writeLine("        i) imageDir=${OPTARG};; ") \
        + writeLine("        r) resultsDir=${OPTARG};;") \
        + writeLine("        t) tasksDir=${OPTARG};;") \
        + writeLine("        *) usage;;") \
        + writeLine("    esac") \
        + writeLine("done") \
        + writeLine("[ -z \"${imageDir}\" ] || [ -z \"${tasksDir}\" ] || [ -z \"${resultsDir}\" ] && usage;") \
        + writeLine("[ -d \"${imageDir}\" ] && [ -d \"${tasksDir}\" ] && [ -d \"${resultsDir}\" ] || failAndExit \"Directory does not exist\";") \
        + writeLine("echo \"Image dir: $imageDir\";") \
        + writeLine("echo \"Tasks dir: $tasksDir\"; ") \
        + writeLine("echo \"Results dir: $resultsDir\"; ") \
        + writeLine("")

    # Assemble array task instructions
    result += \
          writeComment("Setup task parameters") \
        + writeLine("pushd $tasksDir") \
        + writeLine("subDirs=(*)") \
        + writeLine("taskName=${subDirs[$SLURM_ARRAY_TASK_ID]}") \
        + writeLine("taskDir=$tasksDir/$taskName") \
        + writeLine("taskResultDir=$resultsDir/$taskName") \
        + writeLine("task=($taskDir/*.sh)") \
        + writeLine("image=($imageDir/*.tar.gz)") \
        + writeLine("echo \"Task ID: $SLURM_ARRAY_TASK_ID / $SLURM_TASK_MAX\"") \
        + writeLine("echo \"Task Name: $taskName\"") \
        + writeLine("echo \"Host Name: `hostname`\"")

    # Invoke
    result += \
          writeComment("Invoke task") \
        + writeLine("srun -N 1 \"$task\" -i \"$image\" -r \"$taskResultDir\" -t \"$taskDir\"")

    with open(filePath, "w") as f:
        f.write(result)

    makeExecutable(filePath)

    print("Wrote %s" % filePath)

def createClusterRunScript(filePath):
    # Header
    result = writeComment("!/bin/bash") \
        + writeComment("Run Script") \
        + writeComment("Run this launch script on any node to kick off benchmarking")

    # Funcs
    result  \
        += writeLine("usage() { echo \"Usage: $0 Must be run in a folder containing Image, Results and Tasks subfolders. \" 1>&2; exit 1; }\n") \
        + writeLine("failAndExit() { echo \"FAILED: $1\" 1>&2; exit 1; }\n")

    # Assemble array task instructions
    result += \
          writeComment("Setup local environment") \
        + writeLine("pushd " + os.path.dirname(filePath)) \
        + writeLine("tasksDir=$PWD/Tasks") \
        + writeLine("resultsDir=$PWD/Results") \
        + writeLine("imageDir=$PWD/Image") \
        + writeLine("batchScript=$PWD/master.sh") \
        + writeLine("[ -d \"${imageDir}\" ] && [ -d \"${tasksDir}\" ] && [ -d \"${resultsDir}\" ] || failAndExit \"Directory does not exist\";") \
        + writeLine("[ -f \"${batchScript}\" ] || failAndExit \"No master.sh script\";") \
        + writeLine("subDirs=($tasksDir/*)") \
        + writeLine("let arraySize=${#subDirs[@]}") \
        + writeLine("[ -z \"${arraySize}\" ] && failAndExit \"No task count\";") \
        + writeLine("let arrayStart=0") \
        + writeLine("let arrayEnd=$arraySize-1") \
        + writeLine("let minNodes=1") \
        + writeLine("let maxNodes=$arraySize") \
        + writeLine("") \
        + writeComment("Invoke") \
        + writeLine("runCmd=\"sbatch --nodes=$minNodes-$maxNodes --array=$arrayStart-$arrayEnd --wait $batchScript -i $imageDir -r $resultsDir -t $tasksDir\"") \
        + writeLine("echo \"$runCmd\"") \
        + writeLine("$runCmd") \
        + writeLine("popd") \
        + writeLine("exit 0")

    with open(filePath, "w") as f:
        f.write(result)

    makeExecutable(filePath)

    print("Wrote %s" % filePath)

# One config per task
def prepareClusterBenchmark(configFile, nfsMountDir):
    benchmarkTasksDir = os.path.join(nfsMountDir,"Tasks")
    benchmarkImageDir = os.path.join(nfsMountDir,"Image")
    benchmarkResultsDir = os.path.join(nfsMountDir,"Results")
    masterScriptFile = os.path.join(nfsMountDir,"master.sh")
    runScriptFile = os.path.join(nfsMountDir,"RUN.sh")

    ensurePath(benchmarkTasksDir)
    ensurePath(benchmarkImageDir)
    ensurePath(benchmarkResultsDir)

    # Split master config into smaller task configs
    benchmarkFilesData = splitBenchmark(configFile, BenchmarkParameters["BenchmarkTaskSize"])
    baseTaskFileName = os.path.join(benchmarkTasksDir, os.path.basename(configFile))
    for i in range(len(benchmarkFilesData)):
        outFileName = appendFileNameSuffix(baseTaskFileName, i)
        with open(outFileName, "w") as f:
            yaml.safe_dump(benchmarkFilesData[i], f)

    # For each task, make its own folder and drop a slave run script
    createClusterBenchmarkTasks(benchmarkTasksDir)

    # Create the base image for task executable
    createTensileBenchmarkContainer( \
        BenchmarkParameters["DockerBaseImage"], \
        BenchmarkParameters["DockerBuildFile"], \
        "%s:%s" % (BenchmarkParameters["DockerImageName"], BenchmarkParameters["DockerImageTag"]), \
        benchmarkImageDir, \
        BenchmarkParameters["TensileFork"], \
        BenchmarkParameters["TensileBranch"], \
        BenchmarkParameters["TensileCommit"])

    createClusterMasterScript(masterScriptFile)
    createClusterRunScript(runScriptFile)

def runClusterBenchmark(runScriptFile):
    with open(os.path.join(os.path.dirname(runScriptFile), "runlog.log"), "wt") as logFile:
        subprocess.check_call(runScriptFile, stdout=logFile, stderr=logFile)

def combineClusterBenchmarkResults(resultsDir, outputDir):
    # Find the partial logic .yaml files
    # These are in each result directory under 3_LibraryLogic
    resultsDirs = [os.path.join(resultsDir, d, "3_LibraryLogic") for d in os.listdir(resultsDir) if os.path.isdir(os.path.join(resultsDir, d, "3_LibraryLogic"))]

    resultsFiles = []
    for d in resultsDirs:
        resultsFiles += [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) ]

    if length(resultsDirs) != length(resultsFiles):
        print("Warning: inconsistent number of expected results. Check that results are complete.")


    mergePartialLogics(
        resultsFiles, \
        outputDir, \
        BenchmarkParameters["FinalLogicForceMerge"], \
        BenchmarkParameters["FinalLogicTrim"])


def assignDefaultBenchmarkParameters():
    """
    Assign default benchmark parameters
    Each parameter has a default parameter,
    and those assignments happen here
    """
    RepoRootDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

    global BenchmarkParameters
    BenchmarkParameters["DeployStep"] = True
    BenchmarkParameters["RunStep"] = True
    BenchmarkParameters["ResultsStep"] = True
    BenchmarkParameters["BenchmarkTaskSize"] = 10 # Sizes per benchmark file
    BenchmarkParameters["DockerBaseImage"] = "compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-dkms-amd-feature-targetid:106-STG1"
    BenchmarkParameters["DockerBuildFile"] =  os.path.join(RepoRootDir, "docker", "dockerfile-tensile-tuning-slurm")
    BenchmarkParameters["DockerImageName"] = "tensile-tuning-cluster-executable"
    BenchmarkParameters["DockerImageTag"] = "TEST"
    BenchmarkParameters["TensileFork"] = "ROCmSoftwarePlatform"
    BenchmarkParameters["TensileBranch"] = "develop"
    BenchmarkParameters["TensileCommit"] = "HEAD"
    BenchmarkParameters["FinalLogicForceMerge"] = False
    BenchmarkParameters["FinalLogicTrim"] = True

def assignBenchmarkParameters( arguments ):
    """
    Assign Benchmark Parameters
    Each parameter has a default parameter, and the user
    can override them, those overridings happen here
    """
    global BenchmarkParameters

    for key in arguments:
        value = arguments[key]
        if key not in BenchmarkParameters:
            print("Warning: Benchmark parameter %s = %s unrecognised." % ( key, value ))
        else:
            BenchmarkParameters[key] = value


def TensileBenchmarkCluster(userArgs):

    def splitExtraParameters(par):
        """
        Allows the --benchmark-parameters option to specify any parameters from the command line.
        """

        (key, value) = par.split("=")
        value = eval(value)
        return (key, value)

    # Declare the benchmark config globally
    global BenchmarkParameters

    assignDefaultBenchmarkParameters()

    argParser = argparse.ArgumentParser()
    argParser.add_argument("BenchmarkLogicPath",    help="Path to benchmark config .yaml files.")
    argParser.add_argument("DeploymentPath",         help="Where to deploy benchmarking files. Should target a directory on shared nfs mount of cluster.")
    argParser.add_argument("--deploy-only",          dest="DeployOnly", action="store_true", default=False, help="Deploy benchmarking files only without running or reducing results")
    argParser.add_argument("--run-only",             dest="RunOnly", action="store_true", default=False, help="Run benchmark without deploying or reducing results")
    argParser.add_argument("--results-only",         dest="ResultsOnly", action="store_true", default=False, help="Reduce results without deploying or running")
    argParser.add_argument("--run-and-results-only", dest="RunAndResultsOnly", action="store_true", default=False, help="Run benchmark and reduce results without deploying")
    argParser.add_argument("--benchmark-parameters", nargs="+", type=splitExtraParameters, default=[])

    args = argParser.parse_args()

    arguments = {}

    for key, value in args.benchmark_parameters:
        arguments[key] = value

    arguments["DeployStep"] = True and (not args.RunOnly and not args.ResultsOnly and not args.RunAndResultsOnly)
    arguments["RunStep"] = True and (not args.DeployOnly and not args.ResultsOnly)
    arguments["ResultsStep"] = True and (not args.DeployOnly and not args.RunOnly)

    assignBenchmarkParameters(arguments)

    # Deploy benchmark package only if needed
    if BenchmarkParameters["DeployStep"] is True:
        print("Preparing benchmarking files...")
        prepareClusterBenchmark(args.BenchmarkLogicPath, args.DeploymentPath)
        print("Finished preparing benchmarking files")

    # Run the benchmark only if needed
    if BenchmarkParameters["RunStep"] is True:
        print("Running benchmark tasks (this might take a while)...")
        runClusterBenchmark(os.path.join(args.DeploymentPath, "RUN.sh"))
        print("Finished benchmark tasks")

    # Combining results only if needed
    if BenchmarkParameters["ResultsStep"] is True:
        print("Combining benchmark results...")
        resultsDir = os.path.join(args.DeploymentPath, "Results")
        finalLogicDir = os.path.join(resultsDir, "Final")
        combineClusterBenchmarkResults(resultsDir, finalLogicDir)
        print("Final logic file saved to: %s" % finalLogicDir)

    print("Finished")

def main():
    TensileBenchmarkCluster(sys.argv[1:])
