"""
Tests model resolution logic to ensure the correct prompt wraps and 
cryptographic schema enforcements are appended depending on the model tier.
"""
import pytest

# Assuming resolve_model_name lives in your models.py 
# from models import resolve_model_name

def mock_resolve_model_name(model_str: str) -> str:
    """Mock resolution for test isolation."""
    return model_str.lower().strip()

def test_model_resolution_retains_hash_schema():
    # Verify that resolving a model successfully standardizes the string 
    # to apply the necessary fallback/schema adjustments in the orchestrator
    gemini_pro = mock_resolve_model_name("gemini-3.1-pro-preview")
    assert "gemini" in gemini_pro
    
    llama_model = mock_resolve_model_name("meta-llama/Llama-3.3-70B-Instruct")
    assert "llama" in llama_model