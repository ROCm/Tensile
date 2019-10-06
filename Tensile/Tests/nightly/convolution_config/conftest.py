# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run_client", action="store_true", default=False, 
        help="generate YAML and run tensile client")

