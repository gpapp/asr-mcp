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


CLIENT_DIR = REPO_ROOT / "asr-client"
CLIENT_MISSING_REASON = "asr-client/live_client.py not present (excluded from the server image)"

# The server image excludes asr-client/ (.dockerignore) — the client ships as a
# standalone zip. Tests that compare the client against the server can only run
# where both are present.
client_missing = pytest.mark.skipif(
    not (CLIENT_DIR / "live_client.py").is_file(),
    reason=CLIENT_MISSING_REASON,
)


@pytest.fixture(scope="module")
def live():
    """live_client.py imported by path (it lives outside the asr_mcp package).

    Skips rather than errors when the file is absent, so a test module can mix
    server-side and client-side cases without a module-wide skip mark that would
    also silence the server tests in the image.
    """
    import importlib.util

    path = CLIENT_DIR / "live_client.py"
    if not path.is_file():
        pytest.skip(CLIENT_MISSING_REASON)
    spec = importlib.util.spec_from_file_location("live_client_undertest", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
