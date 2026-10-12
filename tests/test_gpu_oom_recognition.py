"""GPU-memory-failure recognition (lesson 5) and the load-time CPU escalation.

The predicate ``is_gpu_oom`` is the single gate every backend consults before
recovering on CPU, so a wording it does not know is not a cosmetic gap: it is a
recoverable condition that kills the request instead. These tests pin the
wording that actually reached us in production -- cuBLAS failing at
``cublasCreate()`` -- plus the load-time path where the same failure used to
abort the whole model reload.

``model_state`` needs onnxruntime and ``model_loader`` needs it plus
huggingface_hub and pydantic, none of which are needed by the logic under test
here, so both are stubbed and the two modules are imported by path.
"""

import importlib.util
import sys
import types

import pytest

from conftest import REPO_ROOT, load_module


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------

def _ort_stub(providers):
    """A stand-in for onnxruntime that records the providers it was asked for.

    ``providers`` is a callable the tests set to the behaviour they want:
    it receives the provider list and returns the session (or raises).
    """
    ort = types.ModuleType("onnxruntime")

    class RunOptions:
        def add_run_config_entry(self, *a, **kw):
            pass

    class SessionOptions:
        graph_optimization_level = None
        intra_op_num_threads = 1
        inter_op_num_threads = 1

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = "all"

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            self.providers = providers
            outcome = ort.behaviour(path, providers)
            if isinstance(outcome, BaseException):
                raise outcome
            self.outcome = outcome

    ort.RunOptions = RunOptions
    ort.SessionOptions = SessionOptions
    ort.GraphOptimizationLevel = GraphOptimizationLevel
    ort.InferenceSession = InferenceSession
    # Not routed through the recorder: this is a capability query, not a
    # session creation, and the call counts below are about session builds.
    ort.get_available_providers = lambda: list(providers.available)
    ort.behaviour = lambda path, providers: object()
    return ort


@pytest.fixture
def ort(monkeypatch):
    """Install a fresh onnxruntime stub; `providers` is set by the test."""

    class Providers:
        #: The stub reports CUDA by default, so the tests exercise the GPU
        #: path; a CPU-only server is a separate test that says so.
        available = ("CUDAExecutionProvider", "CPUExecutionProvider")

        def __init__(self):
            self.calls = []
            self.fn = lambda path, providers: object()

        def __call__(self, path, providers):
            self.calls.append(providers)
            return self.fn(path, providers)

        @property
        def cuda_calls(self):
            return [c for c in self.calls
                    if isinstance(c, list) and "CUDAExecutionProvider" in str(c)]

    p = Providers()
    stub = _ort_stub(p)
    stub.behaviour = p
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)
    return p


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Keep the stubbed imports out of every other test module."""
    for name in list(sys.modules):
        if name in ("_t_model_state", "_t_model_loader"):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _model_state():
    return load_module("_t_model_state", "asr_mcp/core/model_state.py")


def _model_loader():
    """model_loader, with huggingface_hub + settings stubbed out.

    It dlopens CUDA libs at import time and reaches for the hub; neither is
    what these tests are about, so both are replaced.
    """
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *a, **kw: ""
    hub.snapshot_download = lambda *a, **kw: ""
    sys.modules.setdefault("huggingface_hub", hub)

    settings = types.ModuleType("asr_mcp.config.settings")

    class Settings:  # noqa: D401 - annotation target only
        cpu_threads = 1
        gpu_memory_limit_gb = 4.0

    settings.Settings = Settings
    settings.get_settings = lambda: Settings()
    sys.modules.setdefault("asr_mcp.config.settings", settings)
    return load_module("_t_model_loader", "asr_mcp/core/model_loader.py")


# --------------------------------------------------------------------------
# is_gpu_oom: the wording that reached production
# --------------------------------------------------------------------------

#: Verbatim from logs/app.log, 2026-10-01 12:55:19. cuBLAS cannot create its
#: own handle because the card is full -- before any weight of ours is placed,
#: which is why the message mentions no allocation of ours at all.
CUBLAS_ALLOC_FAILED = (
    "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Exception during "
    "initialization: cuda_call.cc:129 CudaCall(...) CUBLAS failure 3: "
    "CUBLAS_STATUS_ALLOC_FAILED ; GPU=0 ; expr=cublasCreate(&cublas_handle);"
)

#: The same condition as faster-whisper/CTranslate2 words it.
CUBLAS_STATUS_ALLOC_FAILED = "cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED"


def test_a_cublas_alloc_failure_is_recognised_as_an_oom(ort):
    """The whole point: unreported wording turned a recoverable fault fatal."""
    assert _model_state().is_gpu_oom(RuntimeError(CUBLAS_ALLOC_FAILED)) is True


def test_cublas_is_recognised_however_the_runtime_words_it(ort):
    is_oom = _model_state().is_gpu_oom
    assert is_oom(RuntimeError(CUBLAS_STATUS_ALLOC_FAILED)) is True
    # CT2 and the CUDA driver have both used these for the same condition.
    assert is_oom(RuntimeError("CUDA error: out of memory")) is True
    assert is_oom(RuntimeError("cudaErrorMemoryAllocation")) is True


def test_the_allocator_wordings_still_work(ort):
    """The original markers are the reason other backends recover; keep them."""
    is_oom = _model_state().is_gpu_oom
    for msg in ("Failed to allocate memory", "CUDA out of memory",
                "NOT_ENOUGH_MEMORY: available memory of 200MiB",
                "smaller than requested bytes"):
        assert is_oom(RuntimeError(msg)) is True, msg


def test_a_broken_model_is_not_an_oom(ort):
    """Matching loosely must not turn every failure into a CPU retry."""
    is_oom = _model_state().is_gpu_oom
    for msg in ("Invalid argument: non-empty input tensor",
                "Unexpected input data type", "model file is corrupt"):
        assert is_oom(RuntimeError(msg)) is False, msg


def test_the_gate_works_on_any_exception_type(ort):
    """ORT raises its own types and has no single base class (lesson 5)."""
    class WeirdError(Exception):
        pass

    assert _model_state().is_gpu_oom(WeirdError(CUBLAS_ALLOC_FAILED)) is True


def test_a_string_is_not_an_exception(ort):
    """Callers pass `str(err)` around; the gate must not blow up on one."""
    assert _model_state().is_gpu_oom("CUBLAS_STATUS_ALLOC_FAILED") is True


# --------------------------------------------------------------------------
# load_embedding_session: the escalation that had no caller
# --------------------------------------------------------------------------

def _load_and_call(ort, tmp_path, providers_fn):
    loader = _model_loader()
    ort.fn = providers_fn
    loader.ensure_embedding_model = lambda settings=None: tmp_path / "emb.onnx"
    return loader.load_embedding_session()


def test_a_gpu_load_failure_falls_back_to_a_cpu_session(ort, tmp_path):
    """cuBLAS failing at session construction must not abort the reload.

    Before the escalation this propagated out of ``reload_models``, leaving the
    server with no embedding session at all: every diarization and attribution
    request failed until whatever else shared the card went away.
    """
    def gpu_then_cpu(path, providers):
        if "CUDAExecutionProvider" in str(providers):
            return RuntimeError(CUBLAS_ALLOC_FAILED)
        return object()

    session = _load_and_call(ort, tmp_path, gpu_then_cpu)

    assert session is not None
    assert session.providers == ["CPUExecutionProvider"]
    assert len(ort.calls) == 2
    # The first attempt really did ask for the GPU. ORT always appends the CPU
    # provider as a fallback, so "is the CPU EP listed" cannot be the test for
    # "were we on the GPU" -- that mistake disables the escalation silently.
    assert ort.cuda_calls, ort.calls


def test_a_healthy_gpu_session_is_not_rebuilt_on_cpu(ort, tmp_path):
    """The retry costs a model load; it must not happen on a good path."""
    session = _load_and_call(ort, tmp_path, lambda path, providers: object())

    assert len(ort.calls) == 1
    assert session.outcome is not None


def test_a_broken_model_still_raises_on_cpu(ort, tmp_path):
    """A non-OOM failure is a real error; retrying it would only add noise."""
    def always_broken(path, providers):
        return RuntimeError("Unexpected input data type")

    with pytest.raises(RuntimeError, match="Unexpected input data type"):
        _load_and_call(ort, tmp_path, always_broken)

    assert len(ort.calls) == 1


def test_a_cpu_only_server_does_not_retry_itself(ort, tmp_path):
    """With no CUDA provider available there is nothing to escalate away from."""
    ort.fn = lambda path, providers: RuntimeError(CUBLAS_ALLOC_FAILED)
    loader = _model_loader()  # noqa: F841 -- re-read per test, see _isolate
    loader.ensure_embedding_model = lambda settings=None: tmp_path / "emb.onnx"
    loader._get_providers = lambda settings=None, kind="encoder": [
        "CPUExecutionProvider"]

    with pytest.raises(RuntimeError, match="CUBLAS_STATUS_ALLOC_FAILED"):
        loader.load_embedding_session()

    assert len(ort.calls) == 1
