
import os
import pytest
import yaml

from Tensile import Tensile
from Tensile import DataType

def walkDict(root, path=""):
    """
    Recursively walks a structure which may consist of dictionaries, lists,
    and other objects. Yields (object, path) for each object in the
    structure.
    """
    yield root, path
    if isinstance(root, dict):
        for key, value in root.items():
            keypath = key
            if path != "":
                keypath = path + "." + keypath
            yield from walkDict(value, keypath)
    elif isinstance(root, list):
        for i,obj in enumerate(root):
            keypath = str(i)
            if path != "":
                keypath = path + "." + keypath
            yield from walkDict(obj, keypath)

def markNamed(name):
    """
    Gets a mark by a name contained in a variable.
    """
    return getattr(pytest.mark, name)

def configMarks(filepath, rootDir):
    """
    Returns a list of marks to add to a particular YAML config path.  Currently gets a mark for:

     - Root directory name.  This separates tests into pre_checkin, nightly, etc.
     - Expected failures. Include 'xfail' in the name of the YAML file.
     - Anything in yaml["TestParameters"]["marks"]
     - validate / validateAll - whether the test validates (all?) results.
     - Data type(s) used in the YAML
     - Problem type(s) used in the YAML
     - Kernel language(s) used in the YAML
    """
    relpath = os.path.relpath(filepath, rootDir)
    components = relpath.split(os.path.sep)

    # First part of directory - nightly, pre-checkin, etc.
    marks = [markNamed(components[0])]

    if 'xfail' in relpath:
        marks.append(pytest.mark.xfail)

    with open(filepath) as f:
        doc = yaml.load(f, yaml.SafeLoader)

    if "TestParameters" in doc:
        if "marks" in doc["TestParameters"]:
            marks += [markNamed(m) for m in doc["TestParameters"]["marks"]]
    
    validate = True
    validateAll = False
    try:
        if doc["GlobalParameters"]['NumElementsToValidate'] == 0:
            validate = False
        if doc["GlobalParameters"]['NumElementsToValidate'] == -1:
            validateAll = True
    except KeyError:
        pass

    if validate:
        marks.append(pytest.mark.validate)
    if validateAll:
        marks.append(pytest.mark.validateAll)

    dataTypes = set([problem[0]["DataType"] for problem in doc["BenchmarkProblems"]])
    operationTypes = set([problem[0]["OperationType"] for problem in doc["BenchmarkProblems"]])
    
    languages = set()
    for obj, path in walkDict(doc):
        if "KernelLanguage" in path and isinstance(obj, str):
            languages.add(obj)
    
    for l in languages:
        marks.append(markNamed(l))

    for dt in dataTypes:
        dataType = DataType.DataType(dt)
        marks.append(markNamed(dataType.toName()))
    
    for operationType in operationTypes:
        marks.append(markNamed(operationType))

    return marks


def findConfigs(rootDir=None):
    """
    Walks rootDir (defaults to trying to find Tensile/Tests) and returns a
    list of test parameters, one for each YAML file.

    Ignores directories called "disabled".
    """
    if rootDir ==  None:
        rootDir = os.path.dirname(os.path.dirname(__file__))
    
    params = []
    for (dirpath, dirnames, filenames) in os.walk(rootDir):
        if 'disabled' in dirnames:
            dirnames.remove('disabled')

        for filename in filenames:
            if filename.endswith('.yaml'):
                filepath = os.path.join(rootDir, dirpath, filename)
                marks = configMarks(filepath, rootDir)
                testname = os.path.splitext(filename)[0]
                params.append(pytest.param(filepath, marks=marks, id=testname))
    return params

@pytest.mark.parametrize("config", findConfigs())
def test_config(tensile_args, config, tmpdir):
    Tensile.Tensile([config, tmpdir.strpath, *tensile_args])
