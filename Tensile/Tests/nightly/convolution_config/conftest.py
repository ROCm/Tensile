import os,sys
import pytest
import Tensile.Tensile as Tensile
from YamlBuilder.YamlBuilder import YamlBuilder

args={}

@pytest.fixture(scope="session", autouse=True)
def tensile_client(tmpdir_factory):
    """
    Build Tensile client and return LocalPath to that client.
    Can be shared among multiple tests.
    """

    tensile_dir = ""
    if args['run_client']:
        if args['client_dir']:
            tensile_dir=os.path.abspath(args['client_dir'])
        else:
            tensile_dir=tmpdir_factory.mktemp("sharedTensileClient")
            setupYamlFile=tensile_dir.join("setup_tensile_client.yaml")
            with open(str(setupYamlFile), 'w') as outfile:
                outfile.write(YamlBuilder.setupHeader)
            print("info: running tensile to build new client with yaml=%s"%str(setupYamlFile))
            Tensile.Tensile([Tensile.TensileTestPath(str(setupYamlFile)), str(tensile_dir)])

    return tensile_dir

def pytest_addoption(parser):
    parser.addoption(
        "--run_client", action="store", type=int, default=0,
        help='''
        0=test Tensile.Convolution,
        1=generate YAML,
        2=and run tensile client, compare generated contraction vs CPU
        3=and run tensile_client with convolution-vs-contraction
        '''
        )
    parser.addoption(
        "--client_dir", action="store", type=str, default=None,
        help="directory that contains existing client build")

def pytest_configure(config):
    args["run_client"] = config.getoption('--run_client')
    args["client_dir"] = config.getoption('--client_dir')

    print ("client=dir=", str(config.getoption("--client_dir")))

