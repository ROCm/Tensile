
import argparse
import os
import sys
import subprocess
import TensileGenBenchmark
import Structs
import glob
import multiprocessing

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
TENSILE_PATH = os.path.dirname(SCRIPT_PATH)
BENCHMARK_PATH = os.path.join(TENSILE_PATH, 'TensileBenchmark')

def which(p, paths=None):
    exes = [p+x for x in ['', '.exe', '.bat']]
    for dirname in list(paths or [])+os.environ['PATH'].split(os.pathsep):
        for exe in exes:
            candidate = os.path.join(os.path.expanduser(dirname), exe)
            if os.path.exists(candidate):
                return candidate
    raise Exception('Cannot find ' + p)

def cmd(args, **kwargs):
    subprocess.check_call(args, env=os.environ, **kwargs)

def cmake(args, **kwargs):
    cmd([which('cmake')] + args, **kwargs)


def mkdir(*ps):
    p = os.path.join(*ps)
    if not os.path.exists(p): os.makedirs(p)
    return p

def benchmark(cargs):
    # arguments
    ap = argparse.ArgumentParser(description="TensileGenBenchmark")
    ap.add_argument('-D', '--define', nargs='+', default=[])
    ap.add_argument('-G', '--generator', default=None)
    ap.add_argument("--problems-path", "-p", dest="problemsPath", default=os.path.join(os.getcwd(),"ProblemXMLs") )
    ap.add_argument("--solutions-path", "-s", dest="solutionsPath", default=os.path.join(os.getcwd(),"SolutionXMLs") )
    ap.add_argument("--build-path", "-B", dest="buildPath", default=os.path.join(os.getcwd(),"TensileBenchmark") )
    ap.add_argument("--backend", "-b", dest="backend", required=True)
    ap.add_argument("--optimize-alpha", dest="optimizeAlphaStr")
    ap.add_argument("--optimize-beta", dest="optimizeBetaStr")
    ap.add_argument("--validate", "-v", dest="validate", action="store_true")
    ap.set_defaults(optimizeAlphaStr="Off")
    ap.set_defaults(optimizeBetaStr="Off")
    ap.set_defaults(validate=False)

    # parse arguments
    args = ap.parse_args(args=cargs)
    inputFiles = glob.glob(args.problemsPath + "/*.xml")
    backend = Structs.Backend()
    generatedPath = os.path.join(args.buildPath, "Generated")
    mkdir(args.buildPath)
    mkdir(args.solutionsPath)
    mkdir(generatedPath)
    if args.backend.lower() in ["opencl_1.2", "opencl", "ocl", "cl"]: backend.value = 0
    elif args.backend.lower() == "hip": backend.value = 1

    # print settings
    print "\nGenBenchmarkFromFiles:"
    print "  backend=" + str(backend)
    print "  problemsPath=" + args.problemsPath
    print "  solutionsPath=" + args.solutionsPath
    print "  buildPath=" + args.buildPath
    print "  inputFiles=" + str(inputFiles)

    # generate benchmark
    TensileGenBenchmark.GenBenchmarkFromFiles( \
        inputFiles, \
        args.solutionsPath, \
        generatedPath, \
        backend,
        args.optimizeAlphaStr=="On" or args.optimizeAlphaStr=="ON",
        args.optimizeBetaStr=="On" or args.optimizeBetaStr=="ON" )

    # Build exe
    cmake_args = [BENCHMARK_PATH]
    if args.generator: cmake_args.append('-G ' + args.generator)
    cmake_args.append('-DTensile_BACKEND='+str(backend).replace(' ', '_'))
    cmake_args.append('-DTensileBenchmark_DIR_SOLUTIONS=' + args.solutionsPath)
    cmake_args.append('-DTensileBenchmark_DIR_GENERATED=' + generatedPath)

    for d in args.define:
        cmake_args.append('-D{0}'.format(d))
    cmake(cmake_args, cwd=args.buildPath)

    build_args = ['--build', args.buildPath, '--config', 'Release']
    if os.path.exists(os.path.join(args.buildPath, 'Makefile')):
        build_args.extend(['--', '-j', str(multiprocessing.cpu_count())])
        build_args.extend(['VERBOSE=1'])
    cmake(build_args)

    validateArgs = []
    if args.validate:
      validateArgs = ["--validate"]


    cmd([os.path.join(args.buildPath, 'bin', 'TensileBenchmark')]+validateArgs)




################################################################################
# TensileDriver - Main
################################################################################
if __name__ == "__main__":

    command = sys.argv[1]

    if command == 'benchmark':
        benchmark(sys.argv[2:])
    else:
        print "Usage: tensile benchmark [args]"
