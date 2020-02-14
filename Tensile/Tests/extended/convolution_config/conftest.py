import pytest
import Tensile.Tensile as Tensile
from collections import namedtuple
from YamlBuilder.YamlBuilder import YamlBuilder
from YamlBuilder.YamlBuilder import Solutions

TestConfig=namedtuple("TestConfig", ["solution", "problem_func"])
Runner=namedtuple("Runner", ["func", "solution"])
TensileState=namedtuple("TensileState", ["args"])

# this control the default solutions used for each test.
solutions = ["src1", "src5_gsu", "asm3_pbd", "asm3_splitu"]
#solutions = ["asm3_pbd"] # run a smaller set of tests


@pytest.fixture(scope="function")
def file_with_test_name(request, tmp_path):
    def get(suffix):
        filename = request.node.name + suffix
        return tmp_path.joinpath(filename)
    return get

@pytest.fixture
def run_nothing():
    def run(conv, problemType, solution=Solutions.defaultSolution(), problemFunc=None, problemLevel=-1, dataType='s'):
        pass
    return run

@pytest.fixture
def run_generate_yaml(request, file_with_test_name):
    def run(conv, problemType, solution=Solutions.defaultSolution(), problemFunc=None, problemLevel=-1, dataType='s'):
        if problemFunc == None:
            problemFunc = YamlBuilder.ProblemSizes
        if problemLevel==-1:
            problemLevel = request.config.getoption("--problem-level")
        config = YamlBuilder.ConvolutionContraction(conv, problemType, solution, dataType, problemFunc, True, problemLevel)
        configFile = file_with_test_name(".contraction.yaml")
        print("Generate_YAML output:", configFile)
        config.write(configFile)
        return configFile
    return run

@pytest.fixture
def run_contraction(tensile_args, tmp_path, run_generate_yaml, request):
    def run(conv, problemType, solution, problemFunc=None, problemLevel=-1, dataType='s'):
        configFile = run_generate_yaml(conv, problemType, solution, problemFunc, problemLevel, dataType)
        Tensile.Tensile([str(configFile), str(tmp_path), *tensile_args])
    return run

@pytest.fixture
def run_convolution_vs_contraction(tensile_args, tmp_path, file_with_test_name):
    def run(conv, problemType={}, solution=Solutions.defaultSolution(), dataType='s'):
        config = YamlBuilder.ConvolutionVsContraction(conv, solution, dataType)
        configFile = file_with_test_name(".conv.yaml")
        config.write(configFile)

        Tensile.Tensile([str(configFile), str(tmp_path), *tensile_args])
    return run

level_params = [pytest.param((0, Solutions.src1), id="Convolution_Class"),
                pytest.param((1, Solutions.defaultSolution()), id="Generate_YAML-" + Solutions.defaultSolution().__name__)] + \
               [pytest.param((2, getattr(Solutions,s)), id="Run_Contraction-"+s) for s in solutions]
               #[pytest.param((3, Solutions.defaultSolution()), id="Run_Convolution_vs_Contraction:" + Solutions.defaultSolution().__name__)]

@pytest.fixture(params=level_params)
def run_convolution_level(request,
                          pytestconfig,
                          run_nothing,
                          run_generate_yaml,
                          run_contraction,
                          run_convolution_vs_contraction):
    level_fixtures = [run_nothing,
                      run_generate_yaml,
                      run_contraction,
                      run_convolution_vs_contraction]
    curLevel = request.param[0]
    argLevel = request.config.getoption("--test-level")
    if curLevel > argLevel:
        pytest.skip()
    return Runner(level_fixtures[curLevel], request.param[1])

def pytest_addoption(parser):
    parser.addoption(
        "--no-conv-assertions", action="store_true", default=1,
        help='''
        Disable assertiong on Convolution class.
        ''')

    parser.addoption(
        "--test-level", action="store", type=int, default=2,
        help='''
        0= test Tensile.Convolution class;
        1= 0 plus generate YAML;
        2= 1 plus run tensile client and compare generated contraction vs CPU;
        3= 2 plus run tensile_client with convolution-vs-contraction (only forward conv tests which define Spatial will PASS )
        '''
        )
    parser.addoption(
        "--problem-level", action="store", type=int, default=2,
        help='''
        How many exact configurations to generate for contraction testing for non-src solutions (typically asm).
        1= single problem
        2= tens of problems
        3= hundreds of problems
        '''
        )

@pytest.fixture()
def tensile_state(request):
    """
    Shared tensile state, including args.
    """
    args={}
    try:
        args["no_conv_assertions"]= request.config.getoption('--no-conv-assertions')
    except ValueError:
        args["no_conv_assertions"]=1
    return TensileState(args=args)
