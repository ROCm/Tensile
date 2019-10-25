import os,sys
import pytest
import Tensile.Tensile as Tensile
from YamlBuilder.YamlBuilder import YamlBuilder

args={}

@pytest.fixture(scope="session", autouse=True)
def tensile_client_dir(tmpdir_factory):
    """
    Return directory with path to the tensile client.  This directory will contain ClientBuildPath (typically 0_Build)
    If --client_dir is specified then the specified client dir is used ;
    else it is built from source.
    In either case the client directory is shared among all tests.
    """

    tensile_client_dir = ""
    if args['run_client'] >= 2:
        if args['client_dir']:
            tensile_client_dir=os.path.abspath(args['client_dir'])
        else:
            tensile_client_dir=tmpdir_factory.mktemp("sharedTensileClient")
            setupYamlFile=tensile_client_dir.join("setup_tensile_client.yaml")
            with open(str(setupYamlFile), 'w') as outfile:
                outfile.write(YamlBuilder.setupHeader)
            print("info: running tensile to build new client with yaml=%s in dir=%s"%(str(setupYamlFile),str(tensile_client_dir)))
            Tensile.Tensile([Tensile.TensileTestPath(str(setupYamlFile)), str(tensile_client_dir)])

    return str(tensile_client_dir)

def pytest_addoption(parser):
    parser.addoption(
        "--run_client", action="store", type=int, default=2,
        help='''
        0= test Tensile.Convolution class;
        1= 0 plus generate YAML;
        2= 1 plus run tensile client and compare generated contraction vs CPU;
        3= 2 plus run tensile_client with convolution-vs-contraction (only forward conv tests which define Spatial will PASS )
        '''
        )
    parser.addoption(
        "--client_dir", action="store", type=str, default=None,
        help="directory that contains existing client build")

def pytest_configure(config):
    args["run_client"] = config.getoption('--run_client')
    args["client_dir"] = config.getoption('--client_dir')

    print ("client=dir=", str(config.getoption("--client_dir")))

