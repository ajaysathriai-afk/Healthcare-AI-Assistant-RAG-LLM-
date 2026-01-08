"""Tests for Streamlit app"""
import pytest

def test_app_module_imports():
    """Test that app module can be imported"""
    try:
        import app
        assert app is not None
    except ImportError as e:
        pytest.fail(f"Failed to import app: {e}")

def test_streamlit_imports():
    """Test that streamlit is available"""
    try:
        import streamlit as st
        assert st is not None
    except ImportError:
        pytest.fail("Streamlit not installed")

def test_app_has_streamlit_code():
    """Test that app.py contains Streamlit code"""
    try:
        with open('app.py', 'r') as f:
            content = f.read()
            
        # Check for common Streamlit patterns
        assert 'streamlit' in content.lower() or 'st.' in content, \
            "app.py should contain Streamlit code"
            
    except FileNotFoundError:
        pytest.fail("app.py not found")
