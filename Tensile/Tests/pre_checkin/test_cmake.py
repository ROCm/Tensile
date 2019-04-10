import Tensile.Tensile as Tensile
import os, subprocess

def get_root(p):
    r = os.path.split(p)
    while r[1] != 'lib' or not r[0]:
        r = os.path.split(r[0])
    return r[0]

def which(p, paths=None):
    exes = [p+x for x in ['', '.exe', '.bat']]
    for dirname in list(paths or [])+os.environ['PATH'].split(os.pathsep):
        for exe in exes:
            candidate = os.path.join(os.path.expanduser(dirname), exe)
            if os.path.isfile(candidate):
                return candidate
    return None

__test_dir__ = os.path.dirname(os.path.realpath(__file__))

__tensile_dir__ = get_root(Tensile.__file__)

__cmake_dir__ = os.path.join(__test_dir__, 'cmake')

__compiler__ = which('hcc', paths=['/opt/rocm/bin'])

def test_cmake(tmpdir):
    env = os.environ
    env['CXX'] = __compiler__
    subprocess.check_call(['cmake', '-DCMAKE_PREFIX_PATH='+__tensile_dir__, __cmake_dir__], cwd=tmpdir.strpath, env=env)
