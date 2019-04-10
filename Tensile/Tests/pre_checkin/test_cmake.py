import Tensile.Tensile as Tensile
import os, subprocess

def get_root(p):
    r = os.path.split(p)
    while r[1] != 'lib' or not r[0]:
        r = os.path.split(r[0])
    return r[0]

__test_dir__ = os.path.dirname(os.path.realpath(__file__))

__tensile_dir__ = get_root(Tensile.__file__)

__cmake_dir__ = os.path.join(__test_dir__, 'cmake')

def test_cmake(tmpdir):
    subprocess.check_call(['cmake', '-DCMAKE_PREFIX_PATH='+__tensile_dir__, __cmake_dir__], cwd=tmpdir.strpath)
