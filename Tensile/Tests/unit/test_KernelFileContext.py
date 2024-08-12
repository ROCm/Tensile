from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from Tensile.TensileCreateLib.KernelFileContext import (
    KernelFileContextManager,
    _closeKernelFiles,
    _openKernelFiles,
)


@pytest.fixture
def mock_openFile():
    mockFile = MagicMock()
    mockOpen = MagicMock(return_value=mockFile)
    with patch("builtins.open", mockOpen):
        yield mockOpen, mockFile


@pytest.fixture
def mock_pathResolve():
    with patch("pathlib.Path.resolve", return_value=Path("/fake/path")) as mockResolve:
        yield mockResolve


# @pytest.fixture
# def mock_pathOpenWithFixedNames():
#     with patch("Tensile.TensileCreateLib.KernelFileContext._openFilesWithFixedNames", return_value=ope) as mockResolve:
#         yield mockResolve


def test_openKernelFiles_withOnlyMergeFilesEnabled(mock_openFile, mock_pathResolve):
    x, y = _openKernelFiles(0, True, False, Path("/some/path"), [])
    mock_openFile[0].assert_not_called()
    assert x == None and y == None, "Should return None when only mergeFiles is enabled"


def test_openKernelFiles_withOnlyLazyLoadingEnabled(mock_openFile, mock_pathResolve):
    x, y = _openKernelFiles(0, False, True, Path("/some/path"), [])
    mock_openFile[0].assert_not_called()
    assert x == None and y == None, "Should return None when only lazyLoading is enabled"


def test_openKernelFiles_withMergeAndLazyEnabled(mock_openFile, mock_pathResolve):
    x, y = _openKernelFiles(2, True, True, Path("/fake/path"), [])
    mock_openFile[0].assert_called()
    assert mock_openFile[0].call_args_list == [
        call(Path("/fake/path/Kernels.cpp"), "a", encoding="utf-8"),
        call(Path("/fake/path/Kernels.h"), "a", encoding="utf-8"),
    ]


def test_openKernelFiles_withNumMergedFiles(mock_openFile, mock_pathResolve):
    x, y = _openKernelFiles(2, False, False, Path("/some/path"), ["kernel1.cpp", "kernel2.cpp"])
    mock_openFile[0].assert_called()


def test_openKernelFiles_noAction(mock_openFile, mock_pathResolve):
    _openKernelFiles(2, False, False, Path("/some/path"), [])
    mock_openFile[0].assert_not_called()


def test_closeKerneFiles(mock_openFile, mock_pathResolve):
    _, mock_file = mock_openFile
    _closeKernelFiles(mock_file, mock_file)
    assert mock_file.close.call_count == 2


def test_KernelFileContextManager(mock_openFile, mock_pathResolve):
    with pytest.raises(
        ValueError, match="To merge files, lazy loading must be set to True, and vice versa."
    ):
        with KernelFileContextManager(True, False, 0, Path("/some/path")):
            pass
