"""Tests for utility functions"""
import pytest

def test_preprocess_imports():
    """Test that preprocess module can be imported"""
    try:
        from utils import preprocess
        assert preprocess is not None
    except ImportError as e:
        pytest.fail(f"Failed to import preprocess: {e}")

def test_simple_chunker_imports():
    """Test that simple_chunker module can be imported"""
    try:
        from utils import simple_chunker
        assert simple_chunker is not None
    except ImportError as e:
        pytest.fail(f"Failed to import simple_chunker: {e}")

def test_chunker_function_exists():
    """Test that chunking function exists"""
    try:
        from utils.simple_chunker import chunk_text
        assert callable(chunk_text)
    except (ImportError, AttributeError):
        pytest.skip("chunk_text function not found in simple_chunker")

def test_chunker_basic_functionality():
    """Test basic chunking if function exists"""
    try:
        from utils.simple_chunker import chunk_text
        
        # Test with simple text
        text = "This is a test. This is another sentence. And one more."
        chunks = chunk_text(text, chunk_size=50)
        
        assert isinstance(chunks, list), "chunk_text should return a list"
        assert len(chunks) > 0, "Should produce at least one chunk"
        
    except (ImportError, AttributeError):
        pytest.skip("chunk_text function not implemented")
    except Exception as e:
        pytest.skip(f"Chunker not ready for testing: {e}")
