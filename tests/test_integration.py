"""Integration tests - marked as slow"""
import pytest

pytestmark = pytest.mark.slow  # All tests in this file are slow

def test_full_pipeline():
    """Test complete pipeline (requires all components)"""
    pytest.skip("Full pipeline test requires data and API keys - implement later")

def test_api_with_retrieval():
    """Test API with actual retrieval (requires FAISS index)"""
    pytest.skip("Retrieval test requires FAISS index - implement later")
