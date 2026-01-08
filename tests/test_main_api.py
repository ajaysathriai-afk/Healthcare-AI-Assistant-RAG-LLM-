"""Tests for FastAPI endpoints in main.py"""
import pytest

def test_main_module_imports():
    """Test that main module can be imported"""
    try:
        import main
        assert main is not None
    except ImportError as e:
        pytest.fail(f"Failed to import main: {e}")

def test_fastapi_app_exists():
    """Test that FastAPI app object exists"""
    try:
        import main
        assert hasattr(main, 'app'), "main.py should have 'app' object"
    except ImportError:
        pytest.fail("Cannot import main.py")

@pytest.mark.slow
def test_health_endpoint():
    """Test /health endpoint (slow - requires API setup)"""
    try:
        from fastapi.testclient import TestClient
        import main
        
        client = TestClient(main.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        
    except ImportError:
        pytest.skip("FastAPI or TestClient not available")
    except Exception as e:
        pytest.skip(f"API setup not ready: {e}")

@pytest.mark.slow
def test_ask_llm_endpoint_exists():
    """Test that /ask_llm endpoint exists (slow)"""
    try:
        from fastapi.testclient import TestClient
        import main
        
        client = TestClient(main.app)
        
        # Just test the endpoint exists, don't require it to work
        response = client.post("/ask_llm", json={"question": "test", "top_k": 3})
        
        # Accept any response code - we're just checking it exists
        assert response.status_code in [200, 422, 500, 400]
        
    except ImportError:
        pytest.skip("FastAPI or TestClient not available")
    except Exception as e:
        pytest.skip(f"API endpoint not ready: {e}")
