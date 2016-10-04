
# python ./CobaltGen/Cobalt.py benchmark --backend OpenCL_1.2 --input-path `pwd`/work --output-path `pwd`/work

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

def run_script(script, args):
    subprocess.check_call([sys.executable, os.path.join(SCRIPT_PATH, script)] + args, env=os.environ)

def which(p, paths=None):
    exes = [p+x for x in ['', '.exe', '.bat']]
    for dirname in list(paths or [])+os.environ['PATH'].split(os.pathsep):
        for exe in exes:
            candidate = os.path.join(os.path.expanduser(dirname), exe)
            if os.path.exists(candidate):
                return candidate
    return None

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
    ap.add_argument("--input-path", dest="inputPath", required=True)
    ap.add_argument("--output-path", dest="outputPath", required=True)
    ap.add_argument("--backend", dest="backend", required=True, choices=["OpenCL_1.2", "HIP"])
    ap.add_argument("--optimize-alpha", dest="optimizeAlphaStr")
    ap.add_argument("--optimize-beta", dest="optimizeBetaStr")
    ap.add_argument("--prefix", dest="prefixPath")
    ap.set_defaults(optimizeAlphaStr="Off")
    ap.set_defaults(optimizeBetaStr="Off")
    ap.set_defaults(prefixPath="/usr")

    # parse arguments
    args = ap.parse_args(args=cargs)
    inputFiles = glob.glob(args.inputPath + "/*.xml")
    backend = Structs.Backend();
    if args.backend == "OpenCL_1.2": backend.value = 0
    elif args.backend == "HIP": backend.value = 1

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
    build_path = os.path.join(args.outputPath, 'build')
    mkdir(build_path)
    cmake([BENCHMARK_PATH, '-DCMAKE_PREFIX_PATH=' + args.prefixPath, '-DCobalt_BACKEND='+args.backend, '-DCobaltBenchmark_DIR_GENERATED=' + args.outputPath], cwd=build_path)
    build_flags = ['--build', build_path, '--config', 'Release']
    if os.path.exists(os.path.join(build_path, 'Makefile')):
        build_flags.extend(['--', '-j', str(multiprocessing.cpu_count())])
    cmake(build_flags)

    cmd([os.path.join(build_path, 'CobaltBenchmark')])

    


################################################################################
# CobaltDriver - Main
################################################################################
if __name__ == "__main__":

    command = sys.argv[1]

    if command == 'benchmark':
        benchmark(sys.argv[2:])
    else:
        print "Unknown command"



