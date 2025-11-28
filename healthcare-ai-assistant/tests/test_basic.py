"""Basic sanity tests that don't require external dependencies"""

def test_python_works():
    """Verify Python is working"""
    assert True

def test_imports():
    """Test that basic modules can be imported"""
    try:
        import sys
        import os
        assert sys.version_info.major >= 3
        assert sys.version_info.minor >= 10
    except Exception as e:
        assert False, f"Basic imports failed: {e}"

def test_project_structure():
    """Verify project structure exists"""
    import os
    assert os.path.exists("main.py"), "main.py should exist"
    assert os.path.exists("ingest.py"), "ingest.py should exist"
    assert os.path.exists("requirements.txt"), "requirements.txt should exist"
