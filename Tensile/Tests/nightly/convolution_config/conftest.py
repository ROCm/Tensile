import os,sys
import pytest
import Tensile.Tensile as Tensile
from YamlBuilder.YamlBuilder import YamlBuilder

run_client = False

@pytest.fixture(scope="session", autouse=True)
def tensile_client(tmpdir_factory):

    print ("setup=", YamlBuilder.setupHeader)
    tensile_dir = ""
    if run_client:
        tensile_dir=tmpdir_factory.mktemp("sharedTensileClient")
        setupYamlFile=tensile_dir.join("setup_tensile_client.yaml")
        with open(str(setupYamlFile), 'w') as outfile:
            outfile.write(YamlBuilder.setupHeader)
        print("info: running tensile to build new client with yaml=%s"%str(setupYamlFile))
        Tensile.Tensile([Tensile.TensileTestPath(str(setupYamlFile)), str(tensile_dir)])

    return tensile_dir

def pytest_addoption(parser):
    parser.addoption(
        "--run_client", action="store_true", default=False,
        help="generate YAML and run tensile client")

def pytest_configure(config):
    #import settings
    #settings.run_client = config.getoption('--run_client')
    if config.getoption('--run_client'):
      run_client = True
    print ("configure run_test=", config.getoption('--run_client'))

