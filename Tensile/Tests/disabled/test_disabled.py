import Tensile.Tensile as Tensile

def test_create_library(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/test_create_library.yaml"), tmpdir.strpath])

def test_assertion_selection(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/test_assertion_selection.yaml"), tmpdir.strpath])

