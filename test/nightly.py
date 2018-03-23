#
# These nightly tests are slow but have good coverage. Fast tests with less coverage are in pre_checkin.py.
#
# To execute this test file, install Tensile, then:
#   py.test -v test/nightly.py
#
# To run test directly, with complete output:
#   mkdir build && cd build
#   python ../Tensile/Tensile.py ../Tensile/Configs/test_hgemm_defaults.yaml ./
#

import Tensile.Tensile as Tensile

# defaults
def test_hgemm_defaults(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_defaults.yaml"), tmpdir.strpath])
def test_sgemm_defaults(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm_defaults.yaml"), tmpdir.strpath])
def test_dgemm_defaults(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_dgemm_defaults.yaml"), tmpdir.strpath])

# thorough tests
def test_hgemm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm.yaml"), tmpdir.strpath])
def test_sgemm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm.yaml"), tmpdir.strpath])

# vectors
def test_hgemm_vectors(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_vectors.yaml"), tmpdir.strpath])
def test_sgemm_vectors(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm_vectors.yaml"), tmpdir.strpath])

# tensor convolution
def test_tensor_convolution(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_convolution.yaml"), tmpdir.strpath])

# tensor contractions
def test_tensor_contraction(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_tensor_contraction.yaml"), tmpdir.strpath])

# assembly
def test_hgemm_asm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_asm.yaml"), tmpdir.strpath])
def test_sgemm_asm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm_asm.yaml"), tmpdir.strpath])
def test_dgemm_asm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_dgemm_asm.yaml"), tmpdir.strpath])

