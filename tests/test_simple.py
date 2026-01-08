def test_always_passes():
    assert True
    
def test_python_version():
    import sys
    assert sys.version_info >= (3, 10)
