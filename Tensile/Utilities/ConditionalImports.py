import warnings

TENSILE_TERM_COLORS: bool = False
try:
    from rich import print
    TENSILE_TERM_COLORS = True
except ImportError:
    pass

try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    from yaml import SafeLoader as yamlLoader

try:
    from yaml import CSafeDumper as yamlDumper 
except ImportError:
    from yaml import SafeDumper as yamlDumper
