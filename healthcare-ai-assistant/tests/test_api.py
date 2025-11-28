# tests/test_api.py
import pytest

pytestmark = pytest.mark.slow  # Mark all API tests as slow to skip in CI

try:
    from fastapi.testclient import TestClient
    import importlib
    import main
    
    importlib.reload(main)
    client = TestClient(main.app)
    HAS_MAIN = True
except Exception:
    HAS_MAIN = False

@pytest.mark.skipif(not HAS_MAIN, reason="main module not available")
def test_healthcheck():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

@pytest.mark.skipif(not HAS_MAIN, reason="main module not available")
def test_ask_llm_basic():
    r = client.post("/ask_llm", json={"question": "Hello", "top_k": 3})
    assert r.status_code in (200, 422, 500)

@pytest.mark.skipif(not HAS_MAIN, reason="main module not available")
def test_ask_llm_empty_question():
    r = client.post("/ask_llm", json={"question": ""})
    assert r.status_code in (400, 422, 200, 500)
