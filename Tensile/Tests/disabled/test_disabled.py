import Tensile.Tensile as Tensile

def test_starter_packed_case(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/starter_packed_case.yaml"), tmpdir.strpath])

def test_create_library(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/test_create_library.yaml"), tmpdir.strpath])

def test_hgemm_nn_source(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/hgemm_nn_source.yaml"), tmpdir.strpath])

def test_assertion_selection(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/test_assertion_selection.yaml"), tmpdir.strpath])

def test_stridea0_pack_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/stridea0_pack_nt.yaml"), tmpdir.strpath])

def test_strideb0_pack_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/strideb0_pack_nn.yaml"), tmpdir.strpath])

