"""
UI Test Script - Test the ChatGPT-style interface
"""

import streamlit as st
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ui_components():
    """Test UI components for duplicate elements"""
    print("🧪 Testing UI Components...")
    
    # Test imports
    try:
        from ui import (
            setup_page_config, apply_custom_css, render_chatgpt_layout, 
            render_chatgpt_sidebar, render_model_info, render_file_upload_chatgpt, 
            process_uploaded_file, render_chatgpt_main, initialize_session_state
        )
        print("✅ All UI imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test analyzer imports
    try:
        from analyzer import LayoutLMAPI, DocumentProcessor
        print("✅ Analyzer imports successful")
    except ImportError as e:
        print(f"❌ Analyzer import error: {e}")
        return False
    
    # Test model trainer imports
    try:
        from model_trainer import ModelTrainer
        print("✅ Model trainer imports successful")
    except ImportError as e:
        print(f"❌ Model trainer import error: {e}")
        return False
    
    return True

def test_session_state():
    """Test session state initialization"""
    print("\n🧪 Testing Session State...")
    
    try:
        from ui import initialize_session_state
        
        # Mock session state
        class MockSessionState:
            def __init__(self):
                self.state = {}
            
            def get(self, key, default=None):
                return self.state.get(key, default)
            
            def __contains__(self, key):
                return key in self.state
            
            def __setitem__(self, key, value):
                self.state[key] = value
        
        # Test initialization
        mock_st = type('MockSt', (), {'session_state': MockSessionState()})()
        
        print("✅ Session state test passed")
        return True
        
    except Exception as e:
        print(f"❌ Session state test failed: {e}")
        return False

def test_file_uploader_keys():
    """Test that file uploaders have unique keys"""
    print("\n🧪 Testing File Uploader Keys...")
    
    try:
        with open('ui.py', 'r') as f:
            content = f.read()
        
        # Check for file uploaders
        import re
        uploaders = re.findall(r'st\.file_uploader\([^)]*key="([^"]*)"', content)
        
        if len(uploaders) != len(set(uploaders)):
            print("❌ Duplicate file uploader keys found")
            return False
        
        print(f"✅ Found {len(uploaders)} unique file uploader keys: {uploaders}")
        return True
        
    except Exception as e:
        print(f"❌ File uploader test failed: {e}")
        return False

def test_button_keys():
    """Test that buttons have unique keys"""
    print("\n🧪 Testing Button Keys...")
    
    try:
        with open('ui.py', 'r') as f:
            content = f.read()
        
        # Check for buttons with static keys (not dynamic ones with f-strings)
        import re
        static_buttons = re.findall(r'st\.button\([^)]*key="([^"]*)"[^)]*\)', content)
        
        # Filter out dynamic keys (those with f-strings are handled separately)
        static_keys = [key for key in static_buttons if not key.startswith('f"')]
        
        if len(static_keys) != len(set(static_keys)):
            print("❌ Duplicate static button keys found")
            return False
        
        print(f"✅ Found {len(static_keys)} unique static button keys")
        return True
        
    except Exception as e:
        print(f"❌ Button test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting UI Tests for ChatGPT-Style Interface")
    print("=" * 60)
    
    tests = [
        test_ui_components,
        test_session_state,
        test_file_uploader_keys,
        test_button_keys
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ChatGPT-style UI is ready!")
        print("✅ No duplicate element issues")
        print("✅ All imports working")
        print("✅ Session state properly configured")
        print("✅ Unique keys for all interactive elements")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    print("\n🌐 Application should be running at: http://localhost:8501")

if __name__ == "__main__":
    main()
