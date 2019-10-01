import Tensile.Tensile as Tensile

def test_strideb0_pack_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/pack_tensor_dims/strideb0_pack_tn.yaml"), tmpdir.strpath])

def test_simple_stridea0_pack(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/pack_tensor_dims/simple_stridea0_pack.yaml"), tmpdir.strpath])

def test_simple_strideb0_pack(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/pack_tensor_dims/simple_strideb0_pack.yaml"), tmpdir.strpath])

def test_multi_free_batch(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/pack_tensor_dims/multi_free_batch.yaml"), tmpdir.strpath])

def test_packed_perf_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/pack_tensor_dims/packed_perf_nn.yaml"), tmpdir.strpath])

def test_strideb0_pack_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/pack_tensor_dims/strideb0_pack_nt.yaml"), tmpdir.strpath])

