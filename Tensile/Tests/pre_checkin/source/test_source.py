import Tensile.Tensile as Tensile

def test_dgemm_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/source/test_dgemm_defaults.yaml"), tmpdir.strpath])

def test_hgemm_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/source/test_hgemm_defaults.yaml"), tmpdir.strpath])

def test_sgemm_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/source/test_sgemm_defaults.yaml"), tmpdir.strpath])

