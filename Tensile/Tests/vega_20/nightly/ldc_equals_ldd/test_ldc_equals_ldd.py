import Tensile.Tensile as Tensile

def test_gemm_ldc_equals_ldd(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/nightly/ldc_equals_ldd/gemm_ldc_equals_ldd.yaml"), tmpdir.strpath])


