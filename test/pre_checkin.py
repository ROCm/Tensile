#
# These pre-checkin tests are fast but lack coverage. Slow tests with more coverage are in nightly.py.
#
# To execute this test file, apt-get install python-pytest, then
#   PYTHONPATH=. py.test -v test/pre_checkin.py
#
# To run test directly, with complete output:
#   mkdir build && cd build
#   python ../Tensile/Tensile.py ../Tensile/Configs/test_hgemm_defaults.yaml ./
#

import Tensile.Tensile as Tensile

# defaults
def test_hgemm_asm_nn(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_asm_nn.yaml"), tmpdir.strpath])
def test_hgemm_asm_tn(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_asm_tn.yaml"), tmpdir.strpath])
def test_hgemm_asm_nt(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_asm_nt.yaml"), tmpdir.strpath])
def test_hgemm_asm_tt(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_asm_tt.yaml"), tmpdir.strpath])

def test_sgemm_asm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm_asm.yaml"), tmpdir.strpath])
def test_dgemm_asm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_dgemm_asm.yaml"), tmpdir.strpath])

def test_hgemm_defaults(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_defaults.yaml"), tmpdir.strpath])
def test_sgemm_defaults(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm_defaults.yaml"), tmpdir.strpath])
def test_dgemm_defaults(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_dgemm_defaults.yaml"), tmpdir.strpath])


