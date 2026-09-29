"""Test helpers.

The modules under test are deliberately dependency-free, but importing them
through their package (``asr_mcp.speaker`` / ``asr_mcp.diarization``) pulls in
torch + onnxruntime.  ``load_module`` imports a single file by path and
pre-seeds ``sys.modules`` so the light-weight tests run without the ML stack.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relpath, package=None):
    """Import a single module file by path, bypassing its package __init__."""
    if name in sys.modules:
        return sys.modules[name]
    if package:
        pkg = sys.modules.get(package)
        if pkg is None:
            pkg = types.ModuleType(package)
            pkg.__path__ = [str(REPO_ROOT / package.replace(".", "/"))]
            sys.modules[package] = pkg
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if package:
        setattr(sys.modules[package], name.rsplit(".", 1)[-1], module)
    spec.loader.exec_module(module)
    return module


def load_uncertainty():
    """Import asr_mcp/speaker/uncertainty.py under its canonical dotted name."""
    return load_module(
        "asr_mcp.speaker.uncertainty",
        "asr_mcp/speaker/uncertainty.py",
        package="asr_mcp.speaker",
    )


@pytest.fixture
def uncertainty():
    return load_uncertainty()


@pytest.fixture
def segment_ops():
    load_uncertainty()
    return load_module("_t_segment_ops", "asr_mcp/diarization/segment_ops.py")


@pytest.fixture
def turn_detector():
    return load_module("_t_turn_detector", "asr_mcp/streaming/turn_detector.py")
