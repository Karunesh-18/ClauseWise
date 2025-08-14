"""
Document Analyzer AI - ChatGPT Style Interface
Main application file
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components
from ui import setup_page_config, apply_custom_css, render_chatgpt_interface, initialize_session_state
from analyzer import LayoutLMAPI, DocumentProcessor

def main():
    """Main application function"""
    try:
        # Setup page configuration
        setup_page_config()
        
        # Apply custom CSS
        apply_custom_css()
        
        # Initialize session state
        initialize_session_state()
        
        # Initialize API and processor
        try:
            if 'layoutlm_api' not in st.session_state:
                with st.spinner("🤖 Initializing AI models..."):
                    st.session_state.layoutlm_api = LayoutLMAPI()
            
            if 'doc_processor' not in st.session_state:
                st.session_state.doc_processor = DocumentProcessor()
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            st.info("Some features may not be available. Please refresh the page.")
            return
        
        # Render the ChatGPT-style interface
        render_chatgpt_interface()
            
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
