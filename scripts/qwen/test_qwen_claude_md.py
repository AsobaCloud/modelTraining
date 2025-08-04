#!/usr/bin/env python3
"""
Test suite for Qwen CLAUDE.md integration
"""

import json
import time
import requests
import pytest
from typing import Dict, Any

# Test configuration
SERVER_URL = "http://localhost:8001"
TEST_TIMEOUT = 60  # seconds

class TestQwenClaudeMD:
    """Test suite for Qwen CLAUDE.md integration"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{SERVER_URL}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("model_loaded", False):
                        break
            except:
                pass
            time.sleep(2)
        else:
            pytest.fail("Server not ready within timeout")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{SERVER_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model"] == "Qwen3-14B"
        assert data["model_loaded"] is True
        assert data["claude_md_enabled"] is True
        assert data["system_prompt_loaded"] is True
        assert "gpu_memory_total" in data
        assert "gpu_memory_used" in data
    
    def test_simple_request_direct_code(self):
        """Test simple request triggers direct code generation"""
        request_data = {
            "prompt": "Create a function to add two numbers",
            "complexity": "simple",
            "max_length": 512
        }
        
        response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["complexity_detected"] in ["simple"]
        assert data["methodology_applied"] == "Direct Code Generation"
        assert "def" in data["code"] or "function" in data["code"]
        assert data["generation_time"] < 15  # Should be fast for simple requests
    
    def test_medium_request_plan_and_code(self):
        """Test medium complexity request triggers PLAN → CODE"""
        request_data = {
            "prompt": "Design a REST API for user management with authentication and database integration",
            "complexity": "medium",
            "max_length": 1024
        }
        
        response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["complexity_detected"] in ["medium"]
        assert data["methodology_applied"] == "PLAN → CODE"
        assert len(data["code"]) > 200  # Should be substantial
        assert data["generation_time"] < 30
    
    def test_complex_request_full_methodology(self):
        """Test complex request triggers full CLAUDE.md methodology"""
        request_data = {
            "prompt": "Design and implement a production-ready microservices architecture with monitoring, security, and CI/CD pipeline",
            "complexity": "complex",
            "max_length": 2048
        }
        
        response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["complexity_detected"] in ["complex"]
        assert data["methodology_applied"] == "EXPLORE → PLAN → CODE → COMMIT"
        assert len(data["code"]) > 500  # Should be comprehensive
        assert data["generation_time"] < 45
    
    def test_auto_complexity_detection(self):
        """Test automatic complexity detection"""
        test_cases = [
            {
                "prompt": "Print hello world",
                "expected_complexity": "simple"
            },
            {
                "prompt": "Create a web application with user authentication and database",
                "expected_complexity": "medium"
            },
            {
                "prompt": "Design enterprise-grade microservices architecture with monitoring, security, scalability, and production deployment",
                "expected_complexity": "complex"
            }
        ]
        
        for case in test_cases:
            request_data = {
                "prompt": case["prompt"],
                "complexity": "auto",
                "max_length": 1024
            }
            
            response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=TEST_TIMEOUT)
            assert response.status_code == 200
            
            data = response.json()
            assert data["complexity_detected"] == case["expected_complexity"]
    
    def test_test_code_inclusion(self):
        """Test that test code is included when requested"""
        request_data = {
            "prompt": "Create a calculator class with basic operations",
            "include_tests": True,
            "max_length": 1024
        }
        
        response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        code_lower = data["code"].lower()
        
        # Should contain test-related keywords
        test_indicators = ["test", "assert", "unittest", "pytest", "def test_"]
        assert any(indicator in code_lower for indicator in test_indicators)
    
    def test_security_focus(self):
        """Test that security considerations are included"""
        request_data = {
            "prompt": "Create a user authentication system",
            "complexity": "medium",
            "max_length": 1024
        }
        
        response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        code_lower = data["code"].lower()
        
        # Should contain security-related keywords
        security_indicators = ["password", "hash", "encrypt", "secure", "auth", "token", "validate"]
        assert any(indicator in code_lower for indicator in security_indicators)
    
    def test_error_handling_generation(self):
        """Test error handling in generated code"""
        request_data = {
            "prompt": "Create a function that reads and processes a JSON file",
            "complexity": "medium",
            "max_length": 1024
        }
        
        response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        code_lower = data["code"].lower()
        
        # Should contain error handling
        error_indicators = ["try", "except", "catch", "error", "exception"]
        assert any(indicator in code_lower for indicator in error_indicators)
    
    def test_metadata_completeness(self):
        """Test that response metadata is complete"""
        request_data = {
            "prompt": "Create a simple web server",
            "max_length": 512
        }
        
        response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        metadata = data["metadata"]
        
        assert "model" in metadata
        assert "quantization" in metadata
        assert "prompt_length" in metadata
        assert "response_length" in metadata
        assert "gpu_memory_used" in metadata
        assert metadata["model"] == "Qwen3-14B"
        assert metadata["quantization"] == "8-bit"
    
    def test_performance_benchmarks(self):
        """Test performance characteristics"""
        test_cases = [
            {"complexity": "simple", "max_time": 15},
            {"complexity": "medium", "max_time": 30},
            {"complexity": "complex", "max_time": 45}
        ]
        
        for case in test_cases:
            request_data = {
                "prompt": f"Create a {case['complexity']} example with appropriate methodology",
                "complexity": case["complexity"],
                "max_length": 1024
            }
            
            start_time = time.time()
            response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=TEST_TIMEOUT)
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < case["max_time"]
    
    def test_claude_md_methodology_phases(self):
        """Test that CLAUDE.md methodology phases are correctly applied"""
        complex_request = {
            "prompt": "Design a complete data pipeline with monitoring, security, and scalability",
            "complexity": "complex",
            "max_length": 2048
        }
        
        response = requests.post(f"{SERVER_URL}/generate", json=complex_request, timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        code = data["code"].lower()
        
        # Should contain methodology phase indicators
        methodology_indicators = [
            "explore", "plan", "code", "commit",
            "requirements", "architecture", "implementation", "deployment"
        ]
        
        # At least half of the methodology indicators should be present
        found_indicators = sum(1 for indicator in methodology_indicators if indicator in code)
        assert found_indicators >= len(methodology_indicators) // 2

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Qwen CLAUDE.md Integration Tests...")
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        f"--timeout={TEST_TIMEOUT}"
    ])
    
    if exit_code == 0:
        print("✅ All tests passed!")
        print("🎉 Qwen CLAUDE.md integration is working correctly")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)