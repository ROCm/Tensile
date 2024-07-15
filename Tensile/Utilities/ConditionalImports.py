import warnings

try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    warnings.warn("Couldn't import CSafeLoader backend, defaulting to slower SafeDumper")
    from yaml import SafeLoader as yamlLoader

try:
    from yaml import CSafeDumper as yamlDumper 
except ImportError:
    warnings.warn("Couldn't import CSafeDumper backend, defaulting to slower SafeDumper")
    from yaml import SafeDumper as yamlDumper
