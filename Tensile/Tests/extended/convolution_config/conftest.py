import pytest
import Tensile.Tensile as Tensile
from YamlBuilder.YamlBuilder import YamlBuilder

args={}

@pytest.fixture(scope="function")
def file_with_test_name(request, tmp_path):
    def get(suffix):
        filename = request.node.name + suffix
        return tmp_path.joinpath(filename)
    return get

@pytest.fixture
def run_nothing():
    def run(conv, problemType, dataType='s'):
        pass
    return run

@pytest.fixture
def run_generate_yaml(file_with_test_name):
    def run(conv, problemType, dataType='s'):
        config = YamlBuilder.ConvolutionContraction(conv, problemType, dataType)
        configFile = file_with_test_name(".contraction.yaml")
        config.write(configFile)
        return configFile
    return run

@pytest.fixture
def run_contraction(tensile_args, tmp_path, run_generate_yaml, request):
    def run(conv, problemType, dataType='s'):
        configFile = run_generate_yaml(conv, problemType, dataType)
        Tensile.Tensile([str(configFile), str(tmp_path), *tensile_args])
    return run

@pytest.fixture
def run_convolution_vs_contraction(tensile_args, tmp_path, file_with_test_name):
    def run(conv, problemType={}, dataType='s'):
        config = YamlBuilder.ConvolutionVsContraction(conv, dataType)
        configFile = file_with_test_name(".conv.yaml")
        config.write(configFile)

        Tensile.Tensile([str(configFile), str(tmp_path), *tensile_args])
    return run

level_params = [pytest.param(0, id="Convolution Class"),
                pytest.param(1, id="Generate YAML"),
                pytest.param(2, id="Run Contraction"),
                pytest.param(3, id="Run Convolution vs Contraction")]

@pytest.fixture(params=level_params)
def run_convolution_level(request,
                          run_nothing,
                          run_generate_yaml,
                          run_contraction,
                          run_convolution_vs_contraction):
    level_fixtures = [run_nothing,
                      run_generate_yaml,
                      run_contraction,
                      run_convolution_vs_contraction]
    curLevel = request.param
    argLevel = request.config.getoption("--level")
    if curLevel > argLevel:
        pytest.skip()
    return level_fixtures[request.param]

def pytest_addoption(parser):
    parser.addoption(
        "--level", action="store", type=int, default=2,
        help='''
        0= test Tensile.Convolution class;
        1= 0 plus generate YAML;
        2= 1 plus run tensile client and compare generated contraction vs CPU;
        3= 2 plus run tensile_client with convolution-vs-contraction (only forward conv tests which define Spatial will PASS )
        '''
        )

def pytest_configure(config):
    args["level"] = config.getoption('--level')

