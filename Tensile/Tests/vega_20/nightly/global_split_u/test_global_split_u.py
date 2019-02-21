import Tensile.Tensile as Tensile

def test_igemm_gsu_beta2(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/nightly/global_split_u/igemm_gsu_beta2.yaml"), tmpdir.strpath])

def test_igemm_gsu_beta1(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/nightly/global_split_u/igemm_gsu_beta1.yaml"), tmpdir.strpath])

def test_igemm_gsu_beta0(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/nightly/global_split_u/igemm_gsu_beta0.yaml"), tmpdir.strpath])

