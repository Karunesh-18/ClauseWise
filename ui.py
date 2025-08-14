import streamlit as st
import os
from analyzer import LayoutLMAPI, DocumentProcessor
import io

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Document Analyzer AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def apply_custom_css():
    """Apply exact ChatGPT website styling"""
    st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Söhne:wght@400;500;600&display=swap');

    /* Reset and global styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [data-testid="stAppViewContainer"] {
        height: 100vh;
        background-color: #212121;
        font-family: "Söhne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #ececf1;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stSidebar"] {display: none !important;}

    /* Main container */
    .main .block-container {
        padding: 0;
        max-width: 100%;
        height: 100vh;
        background-color: #212121;
    }

    /* Sidebar - exact ChatGPT styling */
    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2f2f2f;
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
    }

    [data-testid="stSidebar"] > div {
        background-color: #171717;
        padding: 0;
    }

    /* Sidebar content */
    .sidebar-content {
        padding: 12px;
        color: #ececf1;
    }

    /* New chat button */
    .new-chat-button {
        width: 100%;
        background: transparent;
        border: 1px solid #565869;
        border-radius: 8px;
        color: #ececf1;
        padding: 12px;
        margin-bottom: 12px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .new-chat-button:hover {
        background-color: #2f2f2f;
    }

    /* Main content area - dark theme like ChatGPT */
    .main-content {
        background-color: #212121;
        height: 100vh;
        display: flex;
        flex-direction: column;
        color: #ececf1;
    }

    /* Welcome screen - ChatGPT style */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        max-width: 768px;
        margin: 0 auto;
        padding: 24px;
        text-align: center;
    }

    .welcome-title {
        font-size: 32px;
        font-weight: 600;
        color: #ececf1;
        margin-bottom: 32px;
    }

    .welcome-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
        width: 100%;
        max-width: 900px;
        margin-bottom: 32px;
    }

    .welcome-card {
        background-color: #2f2f2f;
        border: 1px solid #565869;
        border-radius: 12px;
        padding: 20px;
        text-align: left;
        transition: all 0.2s ease;
    }

    .welcome-card:hover {
        background-color: #353740;
    }

    .welcome-card h3 {
        color: #ececf1;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .welcome-card p {
        color: #c5c5d2;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Chat messages */
    .chat-message {
        width: 100%;
        padding: 24px 0;
        border-bottom: 1px solid #2f2f2f;
    }

    .chat-message.user {
        background-color: #2f2f2f;
    }

    .chat-message.assistant {
        background-color: #212121;
    }

    .message-content {
        max-width: 768px;
        margin: 0 auto;
        padding: 0 24px;
        display: flex;
        gap: 16px;
        align-items: flex-start;
    }

    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
        flex-shrink: 0;
    }

    .user-avatar {
        background-color: #10a37f;
        color: white;
    }

    .assistant-avatar {
        background-color: #ab68ff;
        color: white;
    }

    .message-text {
        flex: 1;
        color: #ececf1;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Input area - ChatGPT style */
    .chat-input-area {
        position: fixed;
        bottom: 0;
        left: 260px;
        right: 0;
        background-color: #212121;
        padding: 24px;
        border-top: 1px solid #2f2f2f;
    }

    .chat-input-container {
        max-width: 768px;
        margin: 0 auto;
        position: relative;
    }

    .chat-input {
        width: 100%;
        background-color: #2f2f2f;
        border: 1px solid #565869;
        border-radius: 12px;
        color: #ececf1;
        font-size: 16px;
        padding: 12px 48px 12px 16px;
        resize: none;
        outline: none;
        min-height: 52px;
        max-height: 200px;
        font-family: inherit;
    }

    .chat-input:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2);
    }

    .chat-input::placeholder {
        color: #8e8ea0;
    }

    .send-button {
        position: absolute;
        right: 8px;
        bottom: 8px;
        background-color: #10a37f;
        border: none;
        border-radius: 6px;
        color: white;
        width: 32px;
        height: 32px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
    }

    .send-button:hover {
        background-color: #0d8f6f;
    }

    .send-button:disabled {
        background-color: #565869;
        cursor: not-allowed;
    }

    /* Analysis buttons - ChatGPT style */
    .analysis-suggestions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 24px;
        justify-content: center;
    }

    .analysis-button {
        background-color: #2f2f2f;
        border: 1px solid #565869;
        border-radius: 20px;
        color: #ececf1;
        padding: 8px 16px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: nowrap;
    }

    .analysis-button:hover {
        background-color: #353740;
        border-color: #10a37f;
    }

    /* File upload area - ChatGPT style */
    .upload-container {
        max-width: 768px;
        margin: 0 auto;
        padding: 24px;
    }

    .upload-area {
        border: 2px dashed #565869;
        border-radius: 12px;
        padding: 48px 24px;
        text-align: center;
        background-color: #2f2f2f;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    .upload-area:hover {
        border-color: #10a37f;
        background-color: #353740;
    }

    .upload-icon {
        font-size: 48px;
        margin-bottom: 16px;
        color: #8e8ea0;
    }

    .upload-text {
        color: #ececf1;
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 8px;
    }

    .upload-subtext {
        color: #8e8ea0;
        font-size: 14px;
    }

    /* Feedback buttons */
    .feedback-buttons {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        opacity: 0;
        transition: opacity 0.2s ease;
    }

    .chat-message:hover .feedback-buttons {
        opacity: 1;
    }

    .feedback-button {
        background: transparent;
        border: 1px solid #565869;
        border-radius: 6px;
        color: #8e8ea0;
        padding: 4px 8px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .feedback-button:hover {
        background-color: #2f2f2f;
        color: #ececf1;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #171717;
    }

    ::-webkit-scrollbar-thumb {
        background: #565869;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #8e8ea0;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .chat-input-area {
            left: 0;
        }

        [data-testid="stSidebar"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }

        .message-content {
            padding: 0 16px;
        }

        .welcome-container {
            padding: 16px;
        }

        .welcome-cards {
            grid-template-columns: 1fr;
        }
    }

    /* Streamlit specific overrides */
    .stButton > button {
        background: transparent !important;
        border: 1px solid #565869 !important;
        color: #ececf1 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: #2f2f2f !important;
        border-color: #10a37f !important;
    }

    .stTextArea > div > div > textarea {
        background-color: #2f2f2f !important;
        border: 1px solid #565869 !important;
        color: #ececf1 !important;
        border-radius: 12px !important;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2) !important;
    }

    /* Hide file uploader label */
    .stFileUploader > label {
        display: none !important;
    }

    .stFileUploader {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_chatgpt_layout():
    """Render ChatGPT-like layout"""
    st.markdown("""
    <div class="chat-layout">
        <div class="sidebar">
            <div class="sidebar-header">
                <button class="new-chat-btn" onclick="window.location.reload()">
                    ➕ New Analysis
                </button>
            </div>
            <div class="sidebar-content" id="sidebar-content">
                <!-- Sidebar content will be populated by Streamlit -->
            </div>
        </div>
        <div class="main-content">
            <div class="chat-header">
                <h1 class="chat-title">Document Analyzer AI</h1>
                <div style="font-size: 14px; color: #6b7280;">Powered by IBM Watson 3</div>
            </div>
            <div class="chat-messages" id="chat-messages">
                <!-- Messages will be populated here -->
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_chatgpt_sidebar():
    """Render ChatGPT-style sidebar"""
    with st.sidebar:
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, key="new_chat"):
            # Clear session state for new analysis
            for key in ['document_text', 'document_images', 'document_name', 'chat_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        st.markdown("---")

        # Model Status
        st.markdown("**🤖 AI Status**")
        if os.getenv('IBM_API_KEY'):
            st.success("Watson 3 Active")
        else:
            st.warning("Watson 3 Offline")

        # Document Stats
        if st.session_state.get('document_text'):
            st.markdown("---")
            st.markdown("**📄 Document**")
            doc_name = st.session_state.get('document_name', 'Unknown')
            st.text(f"📁 {doc_name[:20]}...")

            word_count = len(st.session_state.document_text.split())
            st.text(f"📊 {word_count:,} words")

        return None





def process_uploaded_file(uploaded_file, doc_processor):
    """Process the uploaded file"""
    if uploaded_file is not None:
        st.success(f"▣ File uploaded: {uploaded_file.name}")
        
        # Process the document
        with st.spinner("▶ Processing document..."):
            file_buffer = io.BytesIO(uploaded_file.getvalue())
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Extract text and images based on file type with intelligent processing
            if file_extension == 'pdf':
                # For PDF files, use intelligent image extraction (only when needed)
                text, images = doc_processor.extract_text_from_pdf(file_buffer, extract_images=False)
            elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif']:
                # For image files, always extract images and use OCR
                text, images = doc_processor.extract_text_from_image(file_buffer)
            elif file_extension == 'docx':
                text, images = doc_processor.extract_text_from_docx(file_buffer)
            elif file_extension == 'txt':
                text, images = doc_processor.extract_text_from_txt(file_buffer)
            elif file_extension == 'csv':
                text, images = doc_processor.extract_text_from_csv(file_buffer)
            elif file_extension == 'json':
                text, images = doc_processor.extract_text_from_json(file_buffer)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                st.info("Supported formats: PDF, DOCX, TXT, CSV, JSON, JPG, PNG, BMP, TIFF, GIF")
                return
            
            st.session_state.document_text = text
            st.session_state.document_images = images
            st.session_state.document_name = uploaded_file.name
            
        st.success("✓ Document processed successfully!")
        return True
    return False

def render_analysis_buttons(layoutlm_api):
    """Render analysis buttons"""
    if not st.session_state.get('document_text'):
        st.info("▣ Please upload a document to begin analysis")
        return
    
    st.markdown("### 🔍 Analysis Options")
    
    # Create button layout
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    with col1:
        if st.button("▣ Generate Summary", use_container_width=True):
            with st.spinner("Generating document summary..."):
                first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                st.session_state.document_summary = layoutlm_api.summarize_document(st.session_state.document_text, first_image)
                st.rerun()
    
    with col2:
        if st.button("⚖ Legal Analysis", use_container_width=True):
            with st.spinner("Performing legal analysis..."):
                st.session_state.legal_analysis = layoutlm_api.legal_document_analysis(st.session_state.document_text)
                st.rerun()
    
    with col3:
        if st.button("▲ Document Type", use_container_width=True):
            with st.spinner("Classifying document type..."):
                st.session_state.document_classification = layoutlm_api.perform_document_classification(st.session_state.document_text)
                st.rerun()
    
    with col4:
        if st.button("▪ Named Entities", use_container_width=True):
            with st.spinner("Extracting named entities..."):
                st.session_state.ner_analysis = layoutlm_api.perform_ner_analysis(st.session_state.document_text)
                st.rerun()
    
    with col5:
        if st.button("▦ Clause Analysis", use_container_width=True):
            with st.spinner("Analyzing clauses..."):
                st.session_state.clause_analysis = layoutlm_api.perform_clause_analysis(st.session_state.document_text)
                st.rerun()
    
    with col6:
        if st.button("▶ Complete Analysis", use_container_width=True):
            with st.spinner("Performing comprehensive legal analysis..."):
                # Perform all analyses
                first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                st.session_state.document_summary = layoutlm_api.summarize_document(st.session_state.document_text, first_image)
                
                comprehensive_results = layoutlm_api.comprehensive_legal_analysis(st.session_state.document_text)
                st.session_state.legal_analysis = comprehensive_results['legal_analysis']
                st.session_state.ner_analysis = comprehensive_results['ner_analysis']
                st.session_state.clause_analysis = comprehensive_results['clause_analysis']
                st.session_state.document_classification = comprehensive_results['document_classification']
                st.rerun()

def render_visual_analysis_option():
    """Render visual analysis option for PDFs"""
    if st.session_state.get('document_name', '').lower().endswith('.pdf'):
        st.markdown("#### Visual Analysis Options")
        col_visual = st.columns(1)[0]
        
        with col_visual:
            if st.button("🔍 Force Visual Analysis (Extract Images)", use_container_width=True):
                with st.spinner("Extracting images for visual analysis..."):
                    st.info("▣ **Visual Analysis Mode**\n"
                           "To use visual analysis, please re-upload the PDF file.\n"
                           "The system will automatically extract images for documents with poor text quality.")

def render_chatgpt_main():
    """Render the main ChatGPT-style interface"""
    # File upload area (only show if no document)
    if not st.session_state.get('document_text'):
        render_file_upload_chatgpt()

    # Analysis suggestions (show if document is loaded)
    render_analysis_suggestions()

    # Chat messages
    render_chatgpt_messages()

    # Chat input
    render_chatgpt_input()

def render_file_upload_chatgpt():
    """Render ChatGPT-style file upload"""
    st.markdown("""
    <div class="upload-container">
        <div class="upload-area">
            <div class="upload-icon">📄</div>
            <div class="upload-text">Upload Your Document</div>
            <div class="upload-subtext">Supports PDF, DOCX, TXT, CSV, JSON, Images</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt', 'csv', 'json', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'],
        help="Supported formats: PDF, DOCX, TXT, CSV, JSON, JPG, PNG, BMP, TIFF, GIF",
        label_visibility="collapsed",
        key="chatgpt_file_uploader"
    )

    return uploaded_file

def render_analysis_suggestions():
    """Render ChatGPT-style analysis suggestions"""
    if st.session_state.get('document_text'):
        st.markdown("""
        <div class="analysis-suggestions">
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📄 Summarize", key="summary_btn"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Please provide a summary of this document."
                })
                _process_analysis_request("summary")

        with col2:
            if st.button("⚖️ Legal Analysis", key="legal_btn"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Please analyze the legal aspects of this document."
                })
                _process_analysis_request("legal")

        with col3:
            if st.button("👥 Find Entities", key="entities_btn"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Please extract the named entities from this document."
                })
                _process_analysis_request("entities")

        with col4:
            if st.button("🔍 Complete Analysis", key="complete_btn"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Please provide a complete analysis of this document."
                })
                _process_analysis_request("complete")

        st.markdown("""
        </div>
        """, unsafe_allow_html=True)

def _process_analysis_request(analysis_type):
    """Process analysis request and add response to chat"""
    layoutlm_api = st.session_state.get('layoutlm_api')
    if not layoutlm_api:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "❌ AI models not initialized. Please refresh the page."
        })
        st.rerun()
        return

    with st.spinner("🤖 AI is analyzing..."):
        try:
            if analysis_type == "summary":
                first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                result = layoutlm_api.summarize_document(st.session_state.document_text, first_image)
            elif analysis_type == "legal":
                result = layoutlm_api.legal_document_analysis(st.session_state.document_text)
            elif analysis_type == "entities":
                result = layoutlm_api.perform_ner_analysis(st.session_state.document_text)
            elif analysis_type == "complete":
                first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                summary = layoutlm_api.summarize_document(st.session_state.document_text, first_image)
                comprehensive_results = layoutlm_api.comprehensive_legal_analysis(st.session_state.document_text)

                result = f"""**Complete Document Analysis:**

**📄 Summary:**
{summary}

**⚖️ Legal Analysis:**
{comprehensive_results['legal_analysis']}

**👥 Named Entities:**
{comprehensive_results['ner_analysis']}

**📋 Clauses:**
{comprehensive_results['clause_analysis']}

**🏷️ Classification:**
{comprehensive_results['document_classification']}
"""

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result
            })

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ Error during analysis: {str(e)}"
            })

    st.rerun()

def render_chatgpt_messages():
    """Render ChatGPT-style messages"""
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Welcome screen - exact ChatGPT style
    if not st.session_state.get('document_text') and not st.session_state.chat_history:
        st.markdown("""
        <div class="welcome-container">
            <h1 class="welcome-title">Ready when you are.</h1>

            <div class="welcome-cards">
                <div class="welcome-card">
                    <h3>📄 Document Analysis</h3>
                    <p>Upload and analyze legal documents with advanced AI. Supports PDF, DOCX, TXT, CSV, JSON, and image formats.</p>
                </div>

                <div class="welcome-card">
                    <h3>🤖 AI-Powered Q&A</h3>
                    <p>Ask natural language questions about your documents using IBM Watson 3 and specialized language models.</p>
                </div>

                <div class="welcome-card">
                    <h3>⚖️ Legal Insights</h3>
                    <p>Extract parties, dates, obligations, financial terms, and key clauses from legal documents automatically.</p>
                </div>

                <div class="welcome-card">
                    <h3>📊 Smart Classification</h3>
                    <p>Identify document types, generate summaries, and perform comprehensive legal document analysis.</p>
                </div>

                <div class="welcome-card">
                    <h3>🎯 Continuous Learning</h3>
                    <p>Provide feedback to improve AI responses and train custom models for better document analysis.</p>
                </div>

                <div class="welcome-card">
                    <h3>🔍 Advanced Features</h3>
                    <p>Named entity recognition, clause analysis, document classification, and intelligent text extraction.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Display chat messages - ChatGPT style
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message-content">
                    <div class="message-avatar user-avatar">You</div>
                    <div class="message-text">{message['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="message-content">
                    <div class="message-avatar assistant-avatar">AI</div>
                    <div class="message-text">{message['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_chatgpt_input():
    """Render ChatGPT-style input area"""
    # Initialize question counter for unique keys
    if 'question_counter' not in st.session_state:
        st.session_state.question_counter = 0

    # Show input area even without document for ChatGPT-like experience
    st.markdown("""
    <div class="chat-input-area">
        <div class="chat-input-container">
    """, unsafe_allow_html=True)

    # Chat input form
    with st.form(key=f"chat_form_{st.session_state.question_counter}", clear_on_submit=True):
        if not st.session_state.get('document_text'):
            placeholder_text = "Upload a document to start analyzing..."
        else:
            placeholder_text = "Message Document Analyzer AI..."

        user_question = st.text_area(
            "Message",
            key=f"question_input_{st.session_state.question_counter}",
            height=60,
            placeholder=placeholder_text,
            label_visibility="collapsed"
        )

        # Send button
        submitted = st.form_submit_button("Send", use_container_width=False)

        if submitted and user_question.strip():
            if not st.session_state.get('document_text'):
                st.warning("Please upload a document first to start asking questions.")
                return

            # Increment counter for next form
            st.session_state.question_counter += 1

            # Add user question to history
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            # Get answer from Watson/LayoutLM
            with st.spinner("🤖 AI is thinking..."):
                layoutlm_api = st.session_state.get('layoutlm_api')
                if layoutlm_api:
                    first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                    answer = layoutlm_api.answer_query(st.session_state.document_text, user_question, first_image)

                    # Add answer to history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    error_msg = "❌ AI models not initialized. Please refresh the page."
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            st.rerun()

    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)



def render_chatgpt_interface():
    """Render the complete ChatGPT-style interface without sidebar"""
    # Simple header
    st.markdown("<h1 style='text-align: center;'>📄 Document Analyzer AI</h1>", unsafe_allow_html=True)

    # Render main content without sidebar
    render_main_content()



def render_main_content():
    """Render the main ChatGPT-style content area"""
    # Top navigation bar with New Chat button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
            # Clear session state for new chat
            for key in ['document_text', 'document_images', 'document_name', 'chat_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Show welcome screen or chat messages
    if not st.session_state.get('document_text') and not st.session_state.chat_history:
        render_welcome_screen()
    else:
        render_chat_area()

    # Always show input area at bottom
    render_input_area()

def render_welcome_screen():
    """Render the welcome screen"""
    st.markdown("## Ready when you are.")
    st.markdown("Upload a document to start analyzing with AI.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt', 'csv', 'json', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'],
        help="Supported formats: PDF, DOCX, TXT, CSV, JSON, JPG, PNG, BMP, TIFF, GIF",
        label_visibility="collapsed",
        key="main_file_uploader"
    )

    # Process uploaded file
    if uploaded_file:
        process_file_upload(uploaded_file)

def render_chat_area():
    """Render the chat messages area"""
    # Analysis suggestions if document is loaded
    if st.session_state.get('document_text'):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📄 Summarize", key="sum_btn"):
                add_analysis_to_chat("summary")
        with col2:
            if st.button("⚖️ Legal Analysis", key="leg_btn"):
                add_analysis_to_chat("legal")
        with col3:
            if st.button("👥 Find Entities", key="ent_btn"):
                add_analysis_to_chat("entities")
        with col4:
            if st.button("🔍 Complete", key="comp_btn"):
                add_analysis_to_chat("complete")

    # Display chat messages
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div style="padding: 24px 0; background-color: #2f2f2f;">
                <div style="max-width: 768px; margin: 0 auto; padding: 0 24px; display: flex; gap: 16px;">
                    <div style="width: 32px; height: 32px; background-color: #10a37f; color: white; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px;">You</div>
                    <div style="flex: 1; color: #ececf1; font-size: 16px; line-height: 1.6;">{message['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 24px 0; background-color: #212121;">
                <div style="max-width: 768px; margin: 0 auto; padding: 0 24px; display: flex; gap: 16px;">
                    <div style="width: 32px; height: 32px; background-color: #ab68ff; color: white; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px;">AI</div>
                    <div style="flex: 1; color: #ececf1; font-size: 16px; line-height: 1.6;">{message['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_input_area():
    """Render the ChatGPT-style input area"""
    # Initialize question counter
    if 'question_counter' not in st.session_state:
        st.session_state.question_counter = 0

    # Input form
    with st.form(key=f"chat_form_{st.session_state.question_counter}", clear_on_submit=True):
        if not st.session_state.get('document_text'):
            placeholder_text = "Upload a document to start analyzing..."
            disabled = True
        else:
            placeholder_text = "Message Document Analyzer AI..."
            disabled = False

        user_question = st.text_area(
            "Message",
            key=f"question_input_{st.session_state.question_counter}",
            height=60,
            placeholder=placeholder_text,
            label_visibility="collapsed",
            disabled=disabled
        )

        submitted = st.form_submit_button("Send", disabled=disabled)

        if submitted and user_question.strip() and st.session_state.get('document_text'):
            st.session_state.question_counter += 1
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            # Get AI response
            with st.spinner("🤖 AI is thinking..."):
                layoutlm_api = st.session_state.get('layoutlm_api')
                if layoutlm_api:
                    first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                    answer = layoutlm_api.answer_query(st.session_state.document_text, user_question, first_image)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "❌ AI models not initialized."})

            st.rerun()

def process_file_upload(uploaded_file):
    """Process uploaded file"""
    if uploaded_file is not None:
        st.success(f"📄 File uploaded: {uploaded_file.name}")

        with st.spinner("🔄 Processing document..."):
            doc_processor = st.session_state.get('doc_processor')
            if doc_processor:
                file_buffer = io.BytesIO(uploaded_file.getvalue())
                file_extension = uploaded_file.name.split('.')[-1].lower()

                try:
                    if file_extension == 'pdf':
                        text, images = doc_processor.extract_text_from_pdf(file_buffer, extract_images=False)
                    elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif']:
                        text, images = doc_processor.extract_text_from_image(file_buffer)
                    elif file_extension == 'docx':
                        text, images = doc_processor.extract_text_from_docx(file_buffer)
                    elif file_extension == 'txt':
                        text = uploaded_file.getvalue().decode('utf-8')
                        images = []
                    elif file_extension == 'csv':
                        text, images = doc_processor.extract_text_from_csv(file_buffer)
                    elif file_extension == 'json':
                        text, images = doc_processor.extract_text_from_json(file_buffer)
                    else:
                        st.error(f"Unsupported file type: {file_extension}")
                        return

                    # Store in session state
                    st.session_state.document_text = text
                    st.session_state.document_images = images
                    st.session_state.document_name = uploaded_file.name

                    st.success("✅ Document processed successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

def add_analysis_to_chat(analysis_type):
    """Add analysis request to chat"""
    layoutlm_api = st.session_state.get('layoutlm_api')
    if not layoutlm_api:
        st.error("AI models not initialized")
        return

    # Add user request to chat
    requests = {
        "summary": "Please provide a summary of this document.",
        "legal": "Please analyze the legal aspects of this document.",
        "entities": "Please extract the named entities from this document.",
        "complete": "Please provide a complete analysis of this document."
    }

    st.session_state.chat_history.append({"role": "user", "content": requests[analysis_type]})

    # Process analysis
    with st.spinner("🤖 AI is analyzing..."):
        try:
            if analysis_type == "summary":
                first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                result = layoutlm_api.summarize_document(st.session_state.document_text, first_image)
            elif analysis_type == "legal":
                result = layoutlm_api.legal_document_analysis(st.session_state.document_text)
            elif analysis_type == "entities":
                result = layoutlm_api.perform_ner_analysis(st.session_state.document_text)
            elif analysis_type == "complete":
                first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                summary = layoutlm_api.summarize_document(st.session_state.document_text, first_image)
                comprehensive_results = layoutlm_api.comprehensive_legal_analysis(st.session_state.document_text)

                result = f"""**Complete Document Analysis:**

**📄 Summary:**
{summary}

**⚖️ Legal Analysis:**
{comprehensive_results['legal_analysis']}

**👥 Named Entities:**
{comprehensive_results['ner_analysis']}

**📋 Clauses:**
{comprehensive_results['clause_analysis']}

**🏷️ Classification:**
{comprehensive_results['document_classification']}
"""

            st.session_state.chat_history.append({"role": "assistant", "content": result})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"❌ Error during analysis: {str(e)}"})

    st.rerun()

def initialize_session_state():
    """Initialize session state variables"""
    # Document-related state
    if 'document_text' not in st.session_state:
        st.session_state.document_text = None
    if 'document_images' not in st.session_state:
        st.session_state.document_images = []
    if 'document_name' not in st.session_state:
        st.session_state.document_name = ""

    # Chat-related state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'question_counter' not in st.session_state:
        st.session_state.question_counter = 0
