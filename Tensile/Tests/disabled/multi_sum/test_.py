import Tensile.Tensile as Tensile

def test_2sum_gsu_src(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum_gsu_src.yaml"), tmpdir.strpath])

def test_2sum(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum.yaml"), tmpdir.strpath])

def test_2sum_gsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum_gsu.yaml"), tmpdir.strpath])

def test_3sum_gsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/3sum_gsu.yaml"), tmpdir.strpath])

def test_2sum_gsu_simple(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum_gsu_simple.yaml"), tmpdir.strpath])

def test_2sum_src(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum_src.yaml"), tmpdir.strpath])

def test_simple_sum2_scrambled(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/simple_sum2_scrambled.yaml"), tmpdir.strpath])

