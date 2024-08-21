TENSILE_TERM_COLORS: bool = False

try:
    from rich import print as print
    TENSILE_TERM_COLORS = True
except ImportError:
    print = print


try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    from yaml import SafeLoader as yamlLoader

try:
    from yaml import CSafeDumper as yamlDumper 
except ImportError:
    from yaml import SafeDumper as yamlDumper

try:
    import joblib
except:
    joblib = None
