import pytest
import os
import sys

testdir = os.path.dirname(__file__)
moddir = os.path.dirname(testdir)
rootdir = os.path.dirname(moddir)
sys.path.append(rootdir)

def pytest_addoption(parser):
    parser.addoption("--tensile-options")
    parser.addoption("--no-common-build", action="store_true")
    parser.addoption("--builddir", "--client-dir")
    parser.addoption("--timing-file")

@pytest.fixture
def timing_path(pytestconfig, tmpdir_factory):
    userDir = pytestconfig.getoption("--timing-file")
    if userDir is not None:
        return userDir
    return str(tmpdir_factory.mktemp("results") / "timing.csv")

@pytest.fixture(autouse=True)
def incremental_timer(request, worker_lock_instance, timing_path):
    #import pdb
    #pdb.set_trace()
    import datetime
    start = datetime.datetime.now()
    yield
    stop = datetime.datetime.now()

    testtime = stop - start
    with worker_lock_instance:
        with open(timing_path, "a") as f:
            f.write("{},{}\n".format(request.node.name, testtime.total_seconds()))

@pytest.fixture(scope="session")
def builddir(pytestconfig, tmpdir_factory):
    userDir = pytestconfig.getoption("--builddir")
    if userDir is not None:
        return userDir
    return str(tmpdir_factory.mktemp("0_Build"))

@pytest.fixture(scope="session")
def worker_lock_path(worker_id, tmp_path_factory):
    if not worker_id:
        return None

    return tmp_path_factory.getbasetemp().parent / "client_execution.lock"

@pytest.fixture
def worker_lock_instance(worker_lock_path):
    if not worker_lock_path:
        return open(os.devnull)

    import filelock
    return filelock.FileLock(str(worker_lock_path))

@pytest.fixture
def tensile_args(pytestconfig, builddir, worker_lock_path):
    rv = []
    if worker_lock_path:
        rv += ["--client-lock", str(worker_lock_path)]

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
