import pytest
import Tensile.Tensile as Tensile

def test_2sum_gsu_src(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/multi_sum_psd/2sum_gsu_src.yaml"), tmpdir.strpath])

def test_2sum(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/multi_sum_psd/2sum.yaml"), tmpdir.strpath])

@pytest.mark.skip("Some bug with PSD ")
def test_2sum_gsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/multi_sum_psd/2sum_gsu.yaml"), tmpdir.strpath])

@pytest.mark.skip("PSD needs to handle >2 summation dims")
def test_3sum_gsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/multi_sum_psd/3sum_gsu.yaml"), tmpdir.strpath])

def test_2sum_gsu_simple(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/multi_sum_psd/2sum_gsu_simple.yaml"), tmpdir.strpath])

def test_2sum_src(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/multi_sum_psd/2sum_src.yaml"), tmpdir.strpath])

def test_simple_sum2_scrambled(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/multi_sum_psd/simple_sum2_scrambled.yaml"), tmpdir.strpath])

