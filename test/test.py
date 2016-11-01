
import argparse
import os
import sys
import subprocess
import glob
import multiprocessing
import tempfile
import contextlib
import shutil

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
COBALT_PATH = os.path.dirname(SCRIPT_PATH)
TEST_PATH = os.path.join(COBALT_PATH, 'test')
PROBLEMS_PATH = os.path.join(TEST_PATH, 'problems')
SIMPLE_PATH = os.path.join(TEST_PATH, 'simple')

def which(p, paths=None):
    exes = [p+x for x in ['', '.exe', '.bat']]
    for dirname in list(paths or [])+os.environ['PATH'].split(os.pathsep):
        for exe in exes:
            candidate = os.path.join(os.path.expanduser(dirname), exe)
            if os.path.exists(candidate):
                return candidate
    raise Exception('Cannot find ' + p)

def cmd(args, **kwargs):
    print args
    subprocess.check_call(args, env=os.environ, **kwargs)

def cmake(args, **kwargs):
    cmd([which('cmake')] + args, **kwargs)

def build(p, defines=None, prefix=None, target=None, generator=None):
    with tempdir() as build_path:
        cmake_args = [p]
        if generator: cmake_args.append('-G ' + args.generator)
        if prefix: 
            cmake_args.append('-DCMAKE_INSTALL_PREFIX='+prefix)
            cmake_args.append('-DCMAKE_PREFIX_PATH='+prefix)

        for d in defines or []:
            cmake_args.append('-D{0}'.format(d))
        cmake(cmake_args, cwd=build_path)

        build_args = ['--build', build_path, '--config', 'Release']
        if target: build_args.extend(['--target', target])
        if os.path.exists(os.path.join(build_path, 'Makefile')):
            build_args.extend(['--', '-j', str(multiprocessing.cpu_count())])
        cmake(build_args)


def mkdir(*ps):
    p = os.path.join(*ps)
    if not os.path.exists(p): os.makedirs(p)
    return p

@contextlib.contextmanager
def tempdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

if __name__ == "__main__":
    # arguments
    ap = argparse.ArgumentParser(description="CobaltTest")
    ap.add_argument('-D', '--define', nargs='+', default=[])
    ap.add_argument('-G', '--generator', default=None)
    ap.add_argument("--backend", "-b", dest="backend", required=True)

    args = ap.parse_args(args=sys.argv[1:])

    with tempdir() as d:
        build_path = os.path.join(d, '_cobalt_build')
        solutions_path = os.path.join(d, 'solutions')
        install_path = os.path.join(d, 'usr')

        #install cobalt
        build(COBALT_PATH, generator=args.generator, prefix=install_path, defines=args.define, target='install')
        # Run benchmark
        cmd([os.path.join(install_path, 'bin', 'cobalt'), 'benchmark', '-b', args.backend, '-i', PROBLEMS_PATH, '-o', solutions_path])
        # Build library
        build(SIMPLE_PATH, generator=args.generator, prefix=install_path, defines=args.define+['Cobalt_SOLUTIONS='+solutions_path])

