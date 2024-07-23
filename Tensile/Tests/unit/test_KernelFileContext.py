from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_openKernelFiles_withMergeFilesEnabled(mock_openFile, mock_pathResolve):
    _openKernelFiles(0, True, False, Path("/some/path"))
    mock_openFile[0].assert_called()


def test_openKernelFiles_withNumMergedFiles(mock_openFile, mock_pathResolve):
    _openKernelFiles(2, False, False, Path("/some/path"), ["kernel1.cpp", "kernel2.cpp"])
    mock_openFile[0].assert_called()


def test_openKernelFiles_noAction(mock_openFile, mock_pathResolve):
    _openKernelFiles(1, False, False, Path("/some/path"))
    mock_openFile[0].assert_not_called()


def test_closeKerneFiles(mock_openFile, mock_pathResolve):
    _, mock_file = mock_openFile
    _closeKernelFiles(mock_file, mock_file)
    assert mock_file.close.call_count == 2


def test_KernelFileContextManager(mock_openFile, mock_pathResolve):
    with KernelFileContextManager(
        {"NumMergedFiles": 0, "MergeFiles": True, "LazyLibraryLoading": False}, Path("/some/path")
    ):
        mock_openFile[0].assert_called()
