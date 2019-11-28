
import pytest
import os
import sys

testdir = os.path.dirname(__file__)
moddir = os.path.dirname(testdir)
rootdir = os.path.dirname(moddir)
sys.path.append(rootdir)

from Tensile import ClientExecutable

def pytest_addoption(parser):
    parser.addoption("--tensile-options")
    parser.addoption("--no-common-build", action="store_true")
    parser.addoption("--builddir", "--client_dir")

@pytest.fixture(scope="session")
def builddir(pytestconfig, tmpdir_factory):
    userDir = pytestconfig.getoption("--builddir")
    if userDir is not None:
        return userDir
    return str(tmpdir_factory.mktemp("0_Build"))

@pytest.fixture
def tensile_args(pytestconfig, builddir):
    rv = []
    extraOptions = pytestconfig.getoption("--tensile-options")
    if extraOptions is not None:
        rv += extraOptions.split(",")
    if not pytestconfig.getoption("--no-common-build"):
        rv += ["--client-build-path", builddir]
    return rv

def pytest_collection_modifyitems(items):
    """
    Mainly for tests that aren't simple YAML files (including unit tests).
    Adds a mark for the root directory name to each test.
    """
    for item in items:
        relpath = item.fspath.relto(testdir)
        components = relpath.split(os.path.sep)
        item.add_marker(getattr(pytest.mark, components[0]))
