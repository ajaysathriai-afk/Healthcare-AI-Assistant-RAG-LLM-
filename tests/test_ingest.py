"""Tests for ingestion pipeline"""
import pytest

def test_ingest_module_imports():
    """Test that ingest module can be imported"""
    try:
        import ingest
        assert ingest is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ingest: {e}")

def test_ingest_has_main_function():
    """Test that ingest has a main ingestion function"""
    try:
        import ingest
        
        # Check for common function names
        possible_functions = ['load_and_index', 'main', 'ingest_data', 'build_index']
        
        has_function = any(hasattr(ingest, func) for func in possible_functions)
        assert has_function, f"ingest.py should have one of: {possible_functions}"
        
    except ImportError:
        pytest.fail("Cannot import ingest.py")

@pytest.mark.slow
def test_ingest_can_run():
    """Test that ingestion can be run (slow - requires data)"""
    pytest.skip("Ingestion requires large data files - skipping for now")
