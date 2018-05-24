#
# These nightly tests are slow but have good coverage. Fast tests with less coverage are in pre_checkin.py.
#
# To execute this test file, apt-get install python-pytest, then
#   PYTHONPATH=. py.test -v Tensile/Tests/nightly
#
# To run test directly, with complete output:
#   mkdir build && cd build
#   python ../Tensile/Tensile.py ../Tensile/Configs/test_hgemm_defaults.yaml ./
#

import Tensile.Tensile as Tensile

# new features
def test_persistent(tmpdir):
    Tensile.Tensile([Tensile.TensileTestPath("nightly/classic/test_persistent.yaml"), tmpdir.strpath])

# tensor convolution
def test_convolution(tmpdir):
    Tensile.Tensile([Tensile.TensileTestPath("nightly/classic/test_convolution.yaml"), tmpdir.strpath])

# tensor contractions
def test_tensor_contraction(tmpdir):
    Tensile.Tensile([Tensile.TensileTestPath("nightly/classic/test_tensor_contraction.yaml"), tmpdir.strpath])
