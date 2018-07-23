import Tensile.Tensile as Tensile

def test_glvw4_edge_no_asem(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("bugs/test_glvw4_edge_no_asem.yaml"), tmpdir.strpath])

