import Tensile.Tensile as Tensile

def test_glvw4_edge_no_asem(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("bugs/test_glvw4_edge_no_asem.yaml"), tmpdir.strpath])

def test_sgemm_fractional_edge(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("bugs/test_sgemm_fractional_edge.yaml"), tmpdir.strpath])

def test_bug_largeK_depthu8(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("bugs/bug_largeK_depthu8.yaml"), tmpdir.strpath])

def test_fractional_plus_pbc(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("bugs/fractional_plus_pbc.yaml"), tmpdir.strpath])

def test_hpa_beta(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("bugs/hpa_beta.yaml"), tmpdir.strpath])

