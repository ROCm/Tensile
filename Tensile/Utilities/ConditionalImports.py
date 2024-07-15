import warnings

TENSILE_TERM_COLORS: bool = False
try:
    from rich import print as _print
    TENSILE_TERM_COLORS = True
except ImportError:
    _print = print


def showwarning(message, category, filename, lineno, file=None, line=None):
    msg = f"{category.__name__}: {message}"
    if TENSILE_TERM_COLORS:
        msg = f"[yellow]{msg}[/yellow]"
    _print(msg)


try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    from yaml import SafeLoader as yamlLoader

try:
    from yaml import CSafeDumper as yamlDumper 
except ImportError:
    from yaml import SafeDumper as yamlDumper
