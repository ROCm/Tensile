import Tensile.Tensile as Tensile

def test_create_library(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/test_create_library.yaml"), tmpdir.strpath])

def test_assertion_selection(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/test_assertion_selection.yaml"), tmpdir.strpath])

def test_hgemm_nn_source(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/hgemm_nn_source.yaml"), tmpdir.strpath])
