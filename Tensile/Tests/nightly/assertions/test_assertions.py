import Tensile.Tensile as Tensile

def test_hgemm_asem2_asm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/assertions/test_hgemm_asem2_asm.yaml"), tmpdir.strpath])

def test_create_library(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/assertions/test_create_library.yaml"), tmpdir.strpath])

def test_assertion_selection(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/assertions/test_assertion_selection.yaml"), tmpdir.strpath])

