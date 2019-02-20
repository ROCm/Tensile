import Tensile.Tensile as Tensile


def test_biga(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("nightly/big_tensor/biga.yaml"), tmpdir.strpath
    ])


def test_bigskinny_nt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("nightly/big_tensor/bigskinny_nt.yaml"),
        tmpdir.strpath
    ])


def test_largec(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("nightly/big_tensor/largec.yaml"),
        tmpdir.strpath
    ])
