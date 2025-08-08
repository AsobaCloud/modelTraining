# Pytest plugin to enforce anti-mock policy
import ast, inspect, pathlib, pytest

PROJECT = "llm-training"  # Our project name

def _src(obj):
    try: return inspect.getsource(obj)
    except Exception: return ""

def pytest_collection_modifyitems(session, config, items):
    """Block internal mocks unless whitelisted"""
    errs = []
    for it in items:
        src = _src(it.function)
        if not src: continue
        uses_mock = any(k in src for k in ["unittest.mock", "mocker", "monkeypatch", ".patch("])
        touches_internal = (f"{PROJECT}." in src)
        if uses_mock and touches_internal and not it.get_closest_marker("allow_mock"):
            path = pathlib.Path(inspect.getsourcefile(it.function) or "unknown")
            errs.append(f"{path}:{it.name}: internal mocking detected")
    if errs:
        raise pytest.UsageError("Mocking policy violations:\n" + "\n".join(" - "+e for e in errs))

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "allow_mock: allow a specific test to mock internals (rare)")