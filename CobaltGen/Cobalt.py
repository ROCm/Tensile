
import argparse
import os
import sys
import subprocess
import CobaltGenBenchmark
import Structs
import glob
import multiprocessing

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
COBALT_PATH = os.path.dirname(SCRIPT_PATH)
BENCHMARK_PATH = os.path.join(COBALT_PATH, 'CobaltBenchmark')

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
    ap = argparse.ArgumentParser(description="CobaltGenBenchmark")
    ap.add_argument('-D', '--define', nargs='+', default=[])
    ap.add_argument('-G', '--generator', default=None)
    ap.add_argument("--input-path", "-i", dest="inputPath", default=os.getcwd())
    ap.add_argument("--output-path", "-o", dest="outputPath", default=os.getcwd())
    ap.add_argument("--backend", "-b", dest="backend", required=True)
    ap.add_argument("--optimize-alpha", dest="optimizeAlphaStr")
    ap.add_argument("--optimize-beta", dest="optimizeBetaStr")
    ap.set_defaults(optimizeAlphaStr="Off")
    ap.set_defaults(optimizeBetaStr="Off")

    # parse arguments
    args = ap.parse_args(args=cargs)
    inputFiles = glob.glob(args.inputPath + "/*.xml")
    backend = Structs.Backend()
    if args.backend.lower() in ["opencl_1.2", "opencl", "ocl", "cl"]: backend.value = 0
    elif args.backend.lower() == "hip": backend.value = 1

    # print settings
    print "\nGenBenchmarkFromFiles:"
    print "  backend=" + str(backend)
    print "  outputPath=" + args.outputPath
    print "  inputPath=" + args.inputPath
    print "  inputFiles=" + str(inputFiles)

    # generate benchmark
    CobaltGenBenchmark.GenBenchmarkFromFiles( \
        inputFiles, \
        args.outputPath, \
        backend,
        args.optimizeAlphaStr=="On" or args.optimizeAlphaStr=="ON",
        args.optimizeBetaStr=="On" or args.optimizeBetaStr=="ON" )

    # Build exe
    build_path = os.path.join(args.outputPath, '_cobalt_build')
    mkdir(build_path)
    cmake_args = [BENCHMARK_PATH]
    if args.generator: cmake_args.append('-G ' + args.generator)
    cmake_args.append('-DCobalt_BACKEND='+str(backend).replace(' ', '_'))
    cmake_args.append('-DCobaltBenchmark_DIR_GENERATED=' + args.outputPath)

    for d in args.define:
        cmake_args.append('-D{0}'.format(d))
    cmake(cmake_args, cwd=build_path)

    build_args = ['--build', build_path, '--config', 'Release']
    if os.path.exists(os.path.join(build_path, 'Makefile')):
        build_args.extend(['--', '-j', str(multiprocessing.cpu_count())])
        # build_args.extend(['VERBOSE=1'])
    cmake(build_args)

    cmd([os.path.join(build_path, 'bin', 'CobaltBenchmark')])




################################################################################
# CobaltDriver - Main
################################################################################
if __name__ == "__main__":

    command = sys.argv[1]

    if command == 'benchmark':
        benchmark(sys.argv[2:])
    else:
        print "Usage: cobalt benchmark [args]"
