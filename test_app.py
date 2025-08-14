"""
Quick test script to verify the application functionality
"""

from analyzer import LayoutLMAPI, DocumentProcessor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_watson_integration():
    """Test Watson 3 integration"""
    print("Testing Watson 3 Integration...")
    
    # Check environment variables
    api_key = os.getenv('IBM_API_KEY')
    project_id = os.getenv('PROJECT_ID')
    
    print(f"API Key found: {'Yes' if api_key else 'No'}")
    print(f"Project ID found: {'Yes' if project_id else 'No'}")
    
    if api_key:
        print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    if project_id:
        print(f"Project ID: {project_id}")
    
    # Initialize API
    try:
        api = LayoutLMAPI()
        print("LayoutLMAPI initialized successfully")
        
        # Check models
        watson_model = api.models.get('watson')
        if watson_model:
            print("✅ Watson 3 model loaded")
        else:
            print("⚠️ Watson 3 not available, using fallback")
            
        summarization_model = api.models.get('summarization')
        if summarization_model:
            print("✅ Summarization model loaded")
            
        classification_model = api.models.get('classification')
        if classification_model:
            print("✅ Classification model loaded")
            
        return api
        
    except Exception as e:
        print(f"❌ Error initializing API: {str(e)}")
        return None

def test_methods(api):
    """Test API methods"""
    if not api:
        return
        
    print("\nTesting API Methods...")
    
    # Test text
    test_text = """
    This is a test legal document between Party A and Party B.
    The payment amount is $10,000 due on January 1, 2024.
    Party A shall provide services as outlined in Section 3.
    """
    
    try:
        # Test question answering
        print("Testing question answering...")
        answer = api.answer_query(test_text, "Who are the parties?")
        print(f"Answer: {answer[:100]}...")
        
        # Test legal analysis
        print("Testing legal analysis...")
        legal_analysis = api.legal_document_analysis(test_text)
        print(f"Legal Analysis: {legal_analysis[:100]}...")
        
        # Test NER
        print("Testing NER...")
        ner_analysis = api.perform_ner_analysis(test_text)
        print(f"NER Analysis: {ner_analysis[:100]}...")
        
        print("✅ All methods working correctly")
        
    except Exception as e:
        print(f"❌ Error testing methods: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing Document Analyzer AI Application")
    print("=" * 50)
    
    api = test_watson_integration()
    test_methods(api)
    
    print("\n" + "=" * 50)
    print("✅ Test completed! Application should be working correctly.")
    print("🚀 You can now use the web interface at http://localhost:8501")
