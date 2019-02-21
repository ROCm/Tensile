import Tensile.Tensile as Tensile

def test_sgemm_nn_source(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/vector_width/sgemm_nn_source.yaml"), tmpdir.strpath])

def test_sgemm_nn_asm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/vector_width/sgemm_nn_asm.yaml"), tmpdir.strpath])

def test_hgemm_nn_asm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/vector_width/hgemm_nn_asm.yaml"), tmpdir.strpath])

#disabled for now due to hanging with ROCm 1.9
#
#def test_hgemm_nn_source(tmpdir):
# Tensile.Tensile([Tensile.TensileTestPath("nightly/vector_width/hgemm_nn_source.yaml"), tmpdir.strpath])

