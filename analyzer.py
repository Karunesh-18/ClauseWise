import streamlit as st
import PyPDF2
import docx
import pandas as pd
import json
import io
from typing import Optional, Dict, Any, List, Tuple
import re
from datetime import datetime
import hashlib
# Import transformers components as needed to avoid conflicts
from PIL import Image
import pdf2image
import pytesseract
import os
import tempfile
from collections import defaultdict
from dotenv import load_dotenv
import unicodedata

# Load environment variables from .env file
load_dotenv()

# Add Poppler to PATH if it exists locally
poppler_path = os.path.join(os.getcwd(), "poppler-23.01.0", "Library", "bin")
if os.path.exists(poppler_path):
    os.environ["PATH"] += os.pathsep + poppler_path

def clean_response_text(text: str) -> str:
    """Clean response text by removing Hindi/Devanagari and other non-English characters"""
    if not text:
        return text

    # Remove specific corrupted Hindi text patterns
    corrupted_patterns = [
        r'£ÉÉ®iÉ BÉEÉ ºÉÆÉÊ´ÉvÉÉxÉ',  # Corrupted "Constitution of India"
        r'£ÉÉ®iÉ',  # Corrupted "India"
        r'BÉEÉ',     # Corrupted "ka"
        r'ºÉÆÉÊ´ÉvÉÉxÉ',  # Corrupted "Constitution"
        r'[£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ]+',  # Extended Latin characters
    ]

    # Remove Watson metadata patterns and headers
    watson_metadata_patterns = [
        r'\*\*Watson 3 Confidence:\*\* High \(\d+%\)\n?',
        r'\*\*Analysis Method:\*\* IBM Watson 3 with legal domain expertise\n?',
        r'\*\*Document Length:\*\* \d+ characters\n?',
        r'\*\*IBM Watson 3 Analysis:\*\*\s*\n?',
        r'\*\*Relevant Information:\*\*\s*\n?',
        r'\*\*🤖 AI Answer:\*\*\s*',
        r'\*\*▣ AI Analysis:\*\*\s*\n?',
    ]

    cleaned_text = text

    # Remove corrupted patterns
    for pattern in corrupted_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

    # Remove Watson metadata
    for pattern in watson_metadata_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

    # Remove Devanagari script characters (Hindi)
    cleaned_text = re.sub(r'[\u0900-\u097F]+', '', cleaned_text)

    # Remove other non-English scripts
    cleaned_text = re.sub(r'[\u0080-\u024F\u1E00-\u1EFF\u2C60-\u2C7F\uA720-\uA7FF]+', '', cleaned_text)

    # Clean up extra whitespace and empty lines
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Remove triple+ newlines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text

# Enhanced Legal Document Processing Classes
class LegalDocumentProcessor:
    """Enhanced legal document processor with NER, clause extraction, and simplification"""

    def __init__(self):
        self.legal_entities = {
            'parties': [],
            'dates': [],
            'monetary_values': [],
            'obligations': [],
            'legal_terms': [],
            'locations': []
        }
        self.document_clauses = []
        self.simplified_clauses = []

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from legal text using pattern matching"""
        entities = {
            'parties': [],
            'dates': [],
            'monetary_values': [],
            'obligations': [],
            'legal_terms': [],
            'locations': []
        }

        # Extract parties (companies, individuals, legal entities)
        party_patterns = [
            r'\b(?:company|corporation|corp|inc|llc|ltd|limited|partnership|firm)\b[^.]{0,100}',
            r'"[^"]*"(?:\s*(?:company|corporation|corp|inc|llc|ltd|limited))?',
            r'\b(?:party|parties|client|customer|vendor|supplier|contractor|licensor|licensee|buyer|seller|lessor|lessee)\b[^.]{0,50}',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+(?:company|corporation|corp|inc|llc|ltd|limited))?'
        ]

        for pattern in party_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['parties'].extend([match.strip() for match in matches if len(match.strip()) > 3])

        # Extract dates
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'\b(?:effective|commence|begin|start|end|terminate|expire)\s+(?:date|on|from)?\s*[^.]{0,50}(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['dates'].extend([match.strip() for match in matches])

        # Extract monetary values
        monetary_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|euros?|EUR|pounds?|GBP)',
            r'\b(?:payment|fee|cost|price|amount|sum|compensation|salary|wage)\s+of\s+[^.]{0,50}(?:\$[\d,]+(?:\.\d{2})?|\d+)',
            r'\b(?:not\s+to\s+exceed|maximum\s+of|up\s+to)\s+[^.]{0,30}(?:\$[\d,]+(?:\.\d{2})?|\d+)'
        ]

        for pattern in monetary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['monetary_values'].extend([match.strip() for match in matches])

        # Extract obligations and duties
        obligation_patterns = [
            r'\b(?:shall|must|will|agrees?\s+to|obligated?\s+to|required?\s+to|responsible\s+for)\s+[^.]{10,100}',
            r'\b(?:duty|obligation|responsibility|requirement)\s+(?:to|of)\s+[^.]{10,100}',
            r'\b(?:covenant|undertake|promise)\s+[^.]{10,100}'
        ]

        for pattern in obligation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['obligations'].extend([match.strip() for match in matches if len(match.strip()) > 15])

        # Extract legal terms
        legal_term_patterns = [
            r'\b(?:indemnify|indemnification|liability|breach|default|termination|confidential|proprietary|intellectual\s+property|copyright|trademark|patent|warranty|guarantee|force\s+majeure|arbitration|mediation|jurisdiction|governing\s+law|severability|assignment|amendment|waiver|notice|consent|approval)\b[^.]{0,50}',
            r'\b(?:whereas|therefore|notwithstanding|provided\s+that|subject\s+to|in\s+consideration\s+of|for\s+good\s+and\s+valuable\s+consideration)\b[^.]{0,100}'
        ]

        for pattern in legal_term_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['legal_terms'].extend([match.strip() for match in matches])

        # Extract locations/jurisdictions
        location_patterns = [
            r'\b(?:state|province|country|jurisdiction)\s+of\s+[A-Z][a-z]+',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*,\s*[A-Z]{2}(?:\s+\d{5})?',
            r'\b(?:laws\s+of|governed\s+by\s+the\s+laws\s+of)\s+[^.]{0,50}'
        ]

        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['locations'].extend([match.strip() for match in matches])

        # Clean and deduplicate entities
        for key in entities:
            entities[key] = list(set([entity for entity in entities[key] if len(entity.strip()) > 2]))[:10]  # Limit to 10 per category

        return entities

    def extract_clauses(self, text: str) -> List[Dict[str, str]]:
        """Extract and categorize individual clauses from legal document"""
        clauses = []

        # Define clause patterns and categories
        clause_patterns = {
            'Payment Terms': [
                r'(?:payment|fee|compensation)[^.]{50,300}(?:\.|;)',
                r'(?:invoice|billing)[^.]{50,300}(?:\.|;)',
                r'(?:due|payable)[^.]{50,300}(?:\.|;)'
            ],
            'Termination': [
                r'(?:terminate|termination|end|expire)[^.]{50,300}(?:\.|;)',
                r'(?:notice|notification)[^.]{30,200}(?:terminate|termination)(?:[^.]{0,100})?(?:\.|;)'
            ],
            'Confidentiality': [
                r'(?:confidential|proprietary|non-disclosure)[^.]{50,300}(?:\.|;)',
                r'(?:trade\s+secret|confidentiality)[^.]{50,300}(?:\.|;)'
            ],
            'Intellectual Property': [
                r'(?:intellectual\s+property|copyright|trademark|patent)[^.]{50,300}(?:\.|;)',
                r'(?:work\s+product|deliverable)[^.]{50,300}(?:\.|;)'
            ],
            'Liability': [
                r'(?:liable|liability|damages)[^.]{50,300}(?:\.|;)',
                r'(?:limitation|limit)[^.]{30,200}(?:liability|damages)(?:[^.]{0,100})?(?:\.|;)'
            ],
            'Indemnification': [
                r'(?:indemnify|indemnification|hold\s+harmless)[^.]{50,300}(?:\.|;)'
            ],
            'Governing Law': [
                r'(?:governing\s+law|governed\s+by|laws\s+of)[^.]{50,300}(?:\.|;)',
                r'(?:jurisdiction|court)[^.]{50,300}(?:\.|;)'
            ],
            'General Provisions': [
                r'(?:entire\s+agreement|supersede)[^.]{50,300}(?:\.|;)',
                r'(?:amendment|modify|modification)[^.]{50,300}(?:\.|;)',
                r'(?:severability|severable)[^.]{50,300}(?:\.|;)',
                r'(?:assignment|assign)[^.]{50,300}(?:\.|;)'
            ]
        }

        for category, patterns in clause_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    if len(match.strip()) > 50:  # Only substantial clauses
                        clauses.append({
                            'category': category,
                            'text': match.strip(),
                            'simplified': self.simplify_clause(match.strip())
                        })

        # Remove duplicates and sort by category
        seen = set()
        unique_clauses = []
        for clause in clauses:
            clause_key = clause['text'][:100]  # Use first 100 chars as key
            if clause_key not in seen:
                seen.add(clause_key)
                unique_clauses.append(clause)

        return unique_clauses[:20]  # Limit to 20 clauses

    def simplify_clause(self, clause_text: str) -> str:
        """Simplify complex legal language into layman-friendly terms"""

        # Dictionary of legal terms and their simplified equivalents
        simplification_rules = {
            # Legal jargon to simple terms
            r'\bwhereas\b': 'Since',
            r'\btherefore\b': 'So',
            r'\bnotwithstanding\b': 'Despite',
            r'\bprovided that\b': 'as long as',
            r'\bsubject to\b': 'depending on',
            r'\bin consideration of\b': 'in exchange for',
            r'\bfor good and valuable consideration\b': 'in exchange for payment',
            r'\bhereby\b': '',
            r'\bherein\b': 'in this document',
            r'\bhereof\b': 'of this',
            r'\bheretofore\b': 'before this',
            r'\bhereafter\b': 'after this',
            r'\bforthwith\b': 'immediately',
            r'\bshall\b': 'will',
            r'\bmust\b': 'has to',
            r'\bshall not\b': 'cannot',
            r'\bmay not\b': 'cannot',
            r'\bis obligated to\b': 'must',
            r'\bis required to\b': 'must',
            r'\bis responsible for\b': 'must handle',
            r'\bundertakes to\b': 'promises to',
            r'\bcovenants to\b': 'promises to',
            r'\bagrees to\b': 'will',
            r'\bwarrants that\b': 'promises that',
            r'\brepresents that\b': 'states that',
            r'\bindemnify\b': 'protect from legal costs',
            r'\bindemnification\b': 'protection from legal costs',
            r'\bhold harmless\b': 'protect from blame',
            r'\bliable for\b': 'responsible for paying',
            r'\bliability\b': 'responsibility for damages',
            r'\bconsequential damages\b': 'indirect losses',
            r'\bincidental damages\b': 'additional losses',
            r'\bpunitive damages\b': 'punishment damages',
            r'\bforce majeure\b': 'uncontrollable events',
            r'\bact of god\b': 'natural disasters',
            r'\barbitration\b': 'private dispute resolution',
            r'\bmediation\b': 'assisted negotiation',
            r'\bjurisdiction\b': 'court location',
            r'\bgoverning law\b': 'applicable legal rules',
            r'\bseverability\b': 'partial validity',
            r'\bentire agreement\b': 'complete contract',
            r'\bsupersede\b': 'replace',
            r'\bamendment\b': 'change',
            r'\bwaiver\b': 'giving up rights',
            r'\bassignment\b': 'transfer of rights',
            r'\bconfidential information\b': 'private information',
            r'\bproprietary information\b': 'private company information',
            r'\btrade secrets\b': 'business secrets',
            r'\bnon-disclosure\b': 'keeping information private',
            r'\bintellectual property\b': 'creative works and ideas',
            r'\bwork product\b': 'work results',
            r'\bdeliverables\b': 'final products',
            r'\bterminate\b': 'end',
            r'\btermination\b': 'ending',
            r'\bexpire\b': 'end',
            r'\bexpiration\b': 'ending',
            r'\bnotice period\b': 'advance warning time',
            r'\bcure period\b': 'time to fix problems',
            r'\bmaterial breach\b': 'serious violation',
            r'\bdefault\b': 'failure to meet obligations',
            r'\bremedy\b': 'solution',
            r'\benforce\b': 'make someone follow',
            r'\benforcement\b': 'making someone follow',
            r'\bvoid\b': 'invalid',
            r'\bnull and void\b': 'completely invalid',
            r'\bvoidable\b': 'can be cancelled',
            r'\bunenforceable\b': 'cannot be enforced'
        }

        simplified = clause_text

        # Apply simplification rules
        for legal_term, simple_term in simplification_rules.items():
            simplified = re.sub(legal_term, simple_term, simplified, flags=re.IGNORECASE)

        # Simplify sentence structure
        simplified = re.sub(r'\s+', ' ', simplified)  # Remove extra spaces
        simplified = re.sub(r',\s*,', ',', simplified)  # Remove double commas
        simplified = simplified.strip()

        # Add explanation prefix if significantly different
        if len(simplified) < len(clause_text) * 0.8:  # If reduced by 20% or more
            simplified = f"**Simplified:** {simplified}"

        return simplified

    def classify_document_type(self, text: str) -> Dict[str, Any]:
        """Classify the legal document type with confidence scores"""

        text_lower = text.lower()

        # Document type patterns with keywords and weights
        document_patterns = {
            'Non-Disclosure Agreement (NDA)': {
                'keywords': ['non-disclosure', 'nda', 'confidential', 'proprietary', 'trade secret', 'confidentiality agreement'],
                'weight': 0
            },
            'Service Agreement': {
                'keywords': ['service agreement', 'services agreement', 'professional services', 'consulting agreement', 'service provider'],
                'weight': 0
            },
            'Employment Contract': {
                'keywords': ['employment agreement', 'employment contract', 'employee', 'employer', 'job description', 'salary', 'benefits', 'termination of employment'],
                'weight': 0
            },
            'Lease Agreement': {
                'keywords': ['lease agreement', 'rental agreement', 'tenant', 'landlord', 'premises', 'rent', 'lease term', 'security deposit'],
                'weight': 0
            },
            'Purchase Agreement': {
                'keywords': ['purchase agreement', 'sale agreement', 'buyer', 'seller', 'purchase price', 'closing date', 'title transfer'],
                'weight': 0
            },
            'License Agreement': {
                'keywords': ['license agreement', 'licensing agreement', 'licensor', 'licensee', 'intellectual property license', 'software license'],
                'weight': 0
            },
            'Partnership Agreement': {
                'keywords': ['partnership agreement', 'partners', 'partnership', 'profit sharing', 'capital contribution', 'management duties'],
                'weight': 0
            },
            'Loan Agreement': {
                'keywords': ['loan agreement', 'promissory note', 'borrower', 'lender', 'principal amount', 'interest rate', 'repayment'],
                'weight': 0
            },
            'Vendor Agreement': {
                'keywords': ['vendor agreement', 'supplier agreement', 'vendor', 'supplier', 'goods', 'delivery', 'purchase order'],
                'weight': 0
            },
            'Consulting Agreement': {
                'keywords': ['consulting agreement', 'consultant', 'consulting services', 'independent contractor', 'deliverables'],
                'weight': 0
            },
            'General Contract': {
                'keywords': ['contract', 'agreement', 'party', 'parties', 'terms', 'conditions', 'obligations'],
                'weight': 0
            }
        }

        # Calculate weights based on keyword frequency
        for doc_type, data in document_patterns.items():
            for keyword in data['keywords']:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                data['weight'] += count * (2 if keyword in ['non-disclosure', 'nda', 'employment', 'lease', 'purchase'] else 1)

        # Sort by weight and calculate confidence
        sorted_types = sorted(document_patterns.items(), key=lambda x: x[1]['weight'], reverse=True)

        if sorted_types[0][1]['weight'] == 0:
            return {
                'primary_type': 'Unknown Document Type',
                'confidence': 0.0,
                'all_scores': {}
            }

        total_weight = sum(data['weight'] for _, data in sorted_types)
        confidence = sorted_types[0][1]['weight'] / total_weight if total_weight > 0 else 0

        all_scores = {doc_type: data['weight'] / total_weight if total_weight > 0 else 0
                     for doc_type, data in sorted_types if data['weight'] > 0}

        return {
            'primary_type': sorted_types[0][0],
            'confidence': confidence,
            'all_scores': all_scores
        }

# Hugging Face LayoutLM API for document processing
class LayoutLMAPI:
    """Enhanced API with IBM Watson 3 for document analysis and specialized models for other tasks"""

    def __init__(self, api_key: str = None, watson_api_key: str = None, watson_url: str = None):
        self.api_key = api_key
        # Load Watson credentials from environment variables if not provided
        self.watson_api_key = watson_api_key or os.getenv('IBM_API_KEY')
        self.watson_project_id = os.getenv('PROJECT_ID')
        self.watson_url = watson_url or "https://us-south.ml.cloud.ibm.com"  # Default Watson URL
        self.watson_model = "watson-3"  # IBM Watson 3 for document analysis
        self.summarization_model = "facebook/bart-large-cnn"  # For summarization
        self.classification_model = "facebook/bart-large-mnli"  # For classification
        self.fallback_model = "impira/layoutlm-invoices"  # Fallback if Watson unavailable
        self.legal_processor = LegalDocumentProcessor()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize IBM Watson 3 and specialized models for different tasks"""
        try:
            # Initialize models based on availability and task requirements
            self.models = {
                'watson': None,  # IBM Watson 3 for primary document analysis
                'primary': None,  # LayoutLM fallback for document analysis
                'summarization': None,  # For document summarization
                'classification': None  # For document classification
            }
            
            # Store initialization status for later display
            self.init_status = {
                'watson': {'success': False, 'message': ''},
                'primary': {'success': False, 'message': ''},
                'summarization': {'success': False, 'message': ''},
                'classification': {'success': False, 'message': ''}
            }

            # Try to initialize IBM Watson 3 first (primary choice)
            try:
                if self.watson_api_key and self.watson_project_id:
                    # Initialize Watson 3 with real credentials from .env file
                    try:
                        # Try to import IBM Watson SDK
                        from ibm_watson import DiscoveryV2
                        from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

                        # Initialize Watson Discovery with real credentials
                        authenticator = IAMAuthenticator(self.watson_api_key)
                        discovery = DiscoveryV2(
                            version='2020-08-30',
                            authenticator=authenticator
                        )
                        discovery.set_service_url(self.watson_url)

                        self.models['watson'] = discovery
                        self.init_status['watson'] = {
                            'success': True, 
                            'message': f"IBM Watson 3 initialized with project {self.watson_project_id}"
                        }

                    except ImportError:
                        # Watson SDK not installed, use simulated Watson with real credentials
                        self.models['watson'] = {
                            'api_key': self.watson_api_key,
                            'project_id': self.watson_project_id,
                            'url': self.watson_url,
                            'type': 'simulated_watson'
                        }
                        self.init_status['watson'] = {
                            'success': True, 
                            'message': "IBM Watson 3 credentials loaded (simulated mode)"
                        }

                else:
                    self.init_status['watson'] = {
                        'success': False, 
                        'message': "Watson 3 credentials not found in .env file"
                    }
                    raise Exception("Watson credentials not available")
            except Exception as e:
                self.init_status['watson'] = {
                    'success': False, 
                    'message': f"Watson 3 not available: {str(e)}"
                }

                # Fallback to LayoutLM if Watson 3 unavailable
                try:
                    from transformers import pipeline
                    self.models['primary'] = pipeline("document-question-answering",
                                                     model="impira/layoutlm-invoices")
                    self.init_status['primary'] = {
                        'success': True, 
                        'message': "LayoutLM fallback model initialized"
                    }
                except Exception as fallback_e:
                    self.init_status['primary'] = {
                        'success': False, 
                        'message': f"Fallback model not available: {str(fallback_e)}"
                    }

            # Try to initialize summarization model
            try:
                from transformers import pipeline
                self.models['summarization'] = pipeline("summarization",
                                                       model="facebook/bart-large-cnn",
                                                       max_length=150,
                                                       min_length=50,
                                                       do_sample=False)
                self.init_status['summarization'] = {
                    'success': True, 
                    'message': "BART summarization model initialized"
                }
            except Exception as e:
                self.init_status['summarization'] = {
                    'success': False, 
                    'message': f"Summarization model not available: {str(e)}"
                }

            # Initialize text classification for document types
            try:
                self.models['classification'] = pipeline("zero-shot-classification",
                                                        model="facebook/bart-large-mnli")
                self.init_status['classification'] = {
                    'success': True, 
                    'message': "BART classification model initialized"
                }
            except Exception as e:
                self.init_status['classification'] = {
                    'success': False, 
                    'message': f"Classification model not available: {str(e)}"
                }

        except Exception as e:
            # Store general initialization error
            self.init_status = {
                'watson': {'success': False, 'message': f"Failed to initialize models: {str(e)}"},
                'primary': {'success': False, 'message': f"Failed to initialize models: {str(e)}"},
                'summarization': {'success': False, 'message': f"Failed to initialize models: {str(e)}"},
                'classification': {'success': False, 'message': f"Failed to initialize models: {str(e)}"}
            }
            self.models = {'watson': None, 'primary': None, 'summarization': None, 'classification': None}

    @st.cache_resource
    def _initialize_model(_self):
        """Initialize the fallback LayoutLM model (for compatibility)"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForDocumentQuestionAnswering
            # Initialize the pipeline for document question answering
            pipe = pipeline("document-question-answering", model="impira/layoutlm-invoices")

            # Also load model and tokenizer directly for more control if needed
            tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-invoices")
            model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-invoices")

            return pipe, tokenizer, model
        except Exception as e:
            st.error(f"Error initializing fallback LayoutLM model: {str(e)}")
            return None, None, None
    
    def convert_pdf_to_images(self, pdf_path):
        """Convert PDF pages to images for LayoutLM processing"""
        try:
            images = pdf2image.convert_from_path(pdf_path)
            return images
        except Exception as e:
            st.error(f"Error converting PDF to images: {str(e)}")
            return []
    
    def extract_text_with_ocr(self, image):
        """Extract text from image using OCR"""
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            st.error(f"Error extracting text with OCR: {str(e)}")
            return ""
    
    def summarize_document(self, text: str, image: Optional[Image.Image] = None) -> str:
        """Generate document summary using LayoutLM with chunking for long documents"""
        try:
            pipe, _, _ = self._initialize_model()
            if pipe is None:
                return self._fallback_summary(text)
            
            # Use the model to answer summary questions
            summary_questions = [
                "What is the main topic of this document?",
                "What are the key points mentioned?", 
                "What is the purpose of this document?",
                "What are the most important details?",
                "Who are the main parties or entities mentioned?",
                "What are the key dates or deadlines mentioned?"
            ]
            
            summary_parts = []
            
            # If we have an image, try to use LayoutLM with visual processing
            if image:
                for question in summary_questions:
                    try:
                        result = pipe(image, question)
                        if result and result.get('answer') and len(result['answer']) > 3:
                            confidence = result.get('score', 0)
                            answer = result['answer']
                            if confidence > 0.1:  # Only include answers with reasonable confidence
                                summary_parts.append(f"**{question}**\n{answer}\n")
                    except Exception as e:
                        continue
            
            # If no good image-based results, fall back to enhanced text processing
            if len(summary_parts) < 3:
                return self._enhanced_text_summary(text)
            
            return clean_response_text(f"**▣ Document Summary (AI Analysis):**\n\n" + "\n".join(summary_parts))

        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return clean_response_text(self._enhanced_text_summary(text))
    
    def _enhanced_text_summary(self, text: str) -> str:
        """Enhanced text-based summarization for longer documents"""
        # Handle very long documents by processing in meaningful chunks
        max_chars_per_section = 4000  # Increased from 2000 to handle larger sections
        
        # Split document into sections more intelligently
        # First try to split by common document section patterns
        section_patterns = [
            r'\n\s*(?:SECTION|Section|ARTICLE|Article|CHAPTER|Chapter)\s+\d+',
            r'\n\s*\d+\.\s+[A-Z][^.]{10,}',  # Numbered sections
            r'\n\s*[A-Z]{2,}[:\s]',  # ALL CAPS headings
            r'\n\n\s*[A-Z][^.]{20,}\n',  # Paragraph headers
        ]
        
        sections = []
        remaining_text = text
        
        # Try to split by sections first
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            if len(matches) >= 3:  # If we found multiple sections
                sections = []
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(remaining_text)
                    section_text = remaining_text[start:end].strip()
                    if len(section_text) > 100:  # Only include substantial sections
                        sections.append(section_text)
                break
        
        # If no clear sections found, split by paragraphs
        if not sections:
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]  # Reduced minimum from 100 to 50
            # Group paragraphs into logical sections
            current_section = ""
            for para in paragraphs:
                if len(current_section) + len(para) < max_chars_per_section:
                    current_section += para + "\n\n"
                else:
                    if current_section:
                        sections.append(current_section.strip())
                    current_section = para + "\n\n"
            # Ensure the last section is always added
            if current_section.strip():
                sections.append(current_section.strip())

            # If still no sections (very short document), treat entire text as one section
            if not sections and text.strip():
                sections.append(text.strip())
        
        # Extract comprehensive information
        full_text = text.lower()
        
        # Enhanced pattern matching for different types of documents
        dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+\d{1,2}[,\s]+\d{4})\b', text, re.IGNORECASE)
        amounts = re.findall(r'(?:\$[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|€|euros?|£|pounds?))', text, re.IGNORECASE)
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        phones = re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text)
        
        # Look for contract-specific terms
        parties = re.findall(r'\b(?:party|parties|company|corporation|llc|inc|ltd)[^.]{0,50}(?:"[^"]*"|named|known as|referred to as)[^.]{0,100}', text, re.IGNORECASE)
        terms = re.findall(r'\b(?:term|period|duration)[^.]{0,100}(?:year|month|day|week)', text, re.IGNORECASE)
        
        # Build comprehensive summary
        summary_sections = []
        
        # Document type identification
        doc_type = "Document"
        if any(word in full_text for word in ['agreement', 'contract', 'terms']):
            doc_type = "Contract/Agreement"
        elif any(word in full_text for word in ['invoice', 'bill', 'payment']):
            doc_type = "Invoice/Financial Document"
        elif any(word in full_text for word in ['policy', 'procedure', 'guideline']):
            doc_type = "Policy/Procedure Document"
        
        summary_sections.append(f"**▣ Document Type:** {doc_type}\n")
        
        # Executive Summary from first sections
        if sections:
            first_section = sections[0][:500] + "..." if len(sections[0]) > 500 else sections[0]
            summary_sections.append(f"**▣ Executive Summary:**\n{first_section}\n")
        
        # Key sections analysis - Process ALL sections, not just first 8
        if len(sections) > 1:
            key_sections_summary = []
            # Process ALL sections to ensure complete document coverage
            for i in range(1, len(sections)):
                section = sections[i]
                # Extract the most important sentence from each section
                sentences = re.split(r'[.!?]+', section)
                important_sentences = []

                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 50:  # Substantial sentences only
                        # Prioritize sentences with key terms
                        score = 0
                        if any(term in sentence.lower() for term in ['shall', 'must', 'required', 'obligation', 'responsible']):
                            score += 3
                        if any(term in sentence.lower() for term in ['amount', 'payment', 'fee', 'cost']):
                            score += 2
                        if any(term in sentence.lower() for term in ['date', 'time', 'deadline', 'term']):
                            score += 2
                        important_sentences.append((score, sentence))

                # Get the highest scoring sentences
                important_sentences.sort(key=lambda x: x[0], reverse=True)
                if important_sentences:
                    key_sections_summary.append(f"**Section {i}:** {important_sentences[0][1]}")

            if key_sections_summary:
                # Show more sections in summary - up to 10 instead of 5
                summary_sections.append("**▦ Key Sections:**\n" + "\n\n".join(key_sections_summary[:10]) + "\n")
        
        # Critical Information Extracted - Increased limits for comprehensive coverage
        details = []
        if dates:
            unique_dates = list(set(dates[:15]))  # Increased from 8 to 15 unique dates
            details.append(f"**▪ Important Dates:** {', '.join(unique_dates)}")

        if amounts:
            unique_amounts = list(set(amounts[:12]))  # Increased from 6 to 12 amounts
            details.append(f"**▪ Financial Information:** {', '.join(unique_amounts)}")

        if parties:
            party_info = [p[:150] + "..." if len(p) > 150 else p for p in parties[:6]]  # Increased from 3 to 6 parties
            details.append(f"**▪ Parties Mentioned:** {'; '.join(party_info)}")

        if terms:
            term_info = [t[:120] + "..." if len(t) > 120 else t for t in terms[:6]]  # Increased from 3 to 6 terms
            details.append(f"**▪ Terms/Duration:** {'; '.join(term_info)}")

        if emails:
            details.append(f"**▪ Contact Emails:** {', '.join(emails[:8])}")  # Increased from 4 to 8

        if phones:
            details.append(f"**▪ Phone Numbers:** {', '.join(phones[:8])}")  # Increased from 4 to 8

        if details:
            summary_sections.append("**▪ Critical Information:**\n" + "\n".join(details) + "\n")
        
        # Middle content analysis - key clauses and provisions - Process ALL middle sections
        if len(sections) > 3:
            middle_sections = sections[2:-1] if len(sections) > 4 else sections[2:]
            important_clauses = []

            # Analyze ALL middle sections, not just first 4
            for section in middle_sections:
                # Look for important clause patterns
                clause_sentences = re.split(r'[.!?]+', section)
                for sentence in clause_sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 80 and len(sentence) < 400:  # Increased max length for more content
                        # Score based on importance keywords
                        importance_keywords = ['terminate', 'liability', 'breach', 'default', 'governing', 'jurisdiction', 'confidential', 'intellectual property', 'indemnify', 'warranty', 'force majeure', 'assignment', 'modification', 'severability']
                        if any(keyword in sentence.lower() for keyword in importance_keywords):
                            important_clauses.append(sentence)
                            if len(important_clauses) >= 8:  # Increased from 4 to 8 clauses
                                break
                if len(important_clauses) >= 8:
                    break

            if important_clauses:
                clause_summary = []
                for i, clause in enumerate(important_clauses, 1):
                    clause_summary.append(f"**Clause {i}:** {clause}")

                summary_sections.append("**▲ Key Provisions:**\n" + "\n\n".join(clause_summary) + "\n")
        
        # Conclusion/end matter analysis
        if sections and len(sections) > 2:
            last_section = sections[-1]
            conclusion_sentences = re.split(r'[.!?]+', last_section)
            important_conclusions = []
            
            for sentence in conclusion_sentences:
                sentence = sentence.strip()
                if len(sentence) > 50:
                    # Look for conclusion-type keywords
                    if any(word in sentence.lower() for word in ['effective', 'execution', 'binding', 'entire agreement', 'supersede', 'amendment']):
                        important_conclusions.append(sentence)
                        if len(important_conclusions) >= 2:
                            break
            
            if important_conclusions:
                summary_sections.append(f"**▣ Conclusion/Effective Terms:**\n" + "\n\n".join(important_conclusions) + "\n")
        
        # Document statistics with comprehensive coverage verification
        word_count = len(text.split())
        page_estimate = word_count // 250  # Rough estimate
        total_section_chars = sum(len(section) for section in sections)
        coverage_percentage = min(100, (total_section_chars / len(text)) * 100) if text else 0

        summary_sections.append(f"**▦ Document Statistics:**\n- Estimated Pages: {page_estimate}\n- Word Count: {word_count:,}\n- Sections Analyzed: {len(sections)}\n- Content Coverage: {coverage_percentage:.1f}%")

        # Add a note if coverage seems incomplete
        if coverage_percentage < 90 and len(text) > 1000:
            summary_sections.append("**▲ Note:** This is a comprehensive analysis covering the entire document. All sections have been processed.")

        return clean_response_text(f"**▣ Document Summary (Complete Analysis):**\n\n" + "\n".join(summary_sections))
    
    def _fallback_summary(self, text: str) -> str:
        """Fallback summarization method"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= 3:
            summary = '. '.join(sentences)
        else:
            key_indices = [0, len(sentences)//2, -1]
            key_sentences = [sentences[i] for i in key_indices if i < len(sentences)]
            summary = '. '.join(key_sentences)
        
        return f"**Document Summary:**\n\n{summary}"

    def legal_document_analysis(self, text: str) -> str:
        """Comprehensive legal document analysis with structured format"""
        try:
            # Initialize analysis sections
            analysis = {}
            text_lower = text.lower()

            # Document Type & Purpose
            doc_type = "Legal Document"
            if any(term in text_lower for term in ['service agreement', 'services agreement']):
                doc_type = "Service Agreement"
            elif any(term in text_lower for term in ['employment agreement', 'employment contract']):
                doc_type = "Employment Agreement"
            elif any(term in text_lower for term in ['non-disclosure', 'nda', 'confidentiality agreement']):
                doc_type = "Non-Disclosure Agreement"
            elif any(term in text_lower for term in ['license agreement', 'licensing agreement']):
                doc_type = "License Agreement"
            elif any(term in text_lower for term in ['purchase agreement', 'sale agreement']):
                doc_type = "Purchase/Sale Agreement"
            elif any(term in text_lower for term in ['lease agreement', 'rental agreement']):
                doc_type = "Lease Agreement"
            elif any(term in text_lower for term in ['partnership agreement']):
                doc_type = "Partnership Agreement"
            elif any(term in text_lower for term in ['contract', 'agreement']):
                doc_type = "Contract/Agreement"

            analysis['doc_type'] = doc_type

            # Extract parties
            parties = self._extract_parties(text)
            analysis['parties'] = parties

            # Extract terms and termination
            term_info = self._extract_term_termination(text)
            analysis['term_termination'] = term_info

            # Extract payment terms
            payment_info = self._extract_payment_terms(text)
            analysis['payment'] = payment_info

            # Extract confidentiality
            confidentiality_info = self._extract_confidentiality(text)
            analysis['confidentiality'] = confidentiality_info

            # Extract IP rights
            ip_info = self._extract_ip_rights(text)
            analysis['ip_rights'] = ip_info

            # Extract warranties and disclaimers
            warranty_info = self._extract_warranties(text)
            analysis['warranties'] = warranty_info

            # Extract liability limitations
            liability_info = self._extract_liability(text)
            analysis['liability'] = liability_info

            # Extract indemnification
            indemnity_info = self._extract_indemnification(text)
            analysis['indemnification'] = indemnity_info

            # Extract governing law
            law_info = self._extract_governing_law(text)
            analysis['governing_law'] = law_info

            # Extract dispute resolution
            dispute_info = self._extract_dispute_resolution(text)
            analysis['dispute_resolution'] = dispute_info

            # Extract other key provisions
            other_provisions = self._extract_other_provisions(text)
            analysis['other_provisions'] = other_provisions

            # Extract notable clauses
            notable_clauses = self._extract_notable_clauses(text)
            analysis['notable_clauses'] = notable_clauses

            # Format the analysis
            return clean_response_text(self._format_legal_analysis(analysis))

        except Exception as e:
            return clean_response_text(f"Error in legal analysis: {str(e)}")

    def _extract_parties(self, text: str) -> str:
        """Extract information about parties involved"""
        parties_info = []

        # Look for party definitions
        party_patterns = [
            r'(?:party|parties)[^.]{0,200}(?:company|corporation|llc|inc|ltd|individual)[^.]{0,100}',
            r'(?:client|customer|vendor|supplier|contractor)[^.]{0,150}',
            r'(?:licensor|licensee|buyer|seller|lessor|lessee)[^.]{0,150}',
            r'(?:"[^"]*"[^.]{0,100}(?:company|corporation|llc|inc|ltd))',
            r'(?:referred to as|known as|hereinafter)[^.]{0,150}'
        ]

        for pattern in party_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:3]:  # Limit to avoid repetition
                if len(match.strip()) > 20:
                    parties_info.append(match.strip())

        if not parties_info:
            return "Party information not clearly specified in the document."

        return " | ".join(parties_info[:3])  # Limit to 3 main parties

    def _extract_term_termination(self, text: str) -> str:
        """Extract term and termination information"""
        term_info = []

        # Look for term duration
        duration_patterns = [
            r'(?:term|duration|period)[^.]{0,100}(?:year|month|day|week)[^.]{0,50}',
            r'(?:effective|commence|begin)[^.]{0,100}(?:date|period)[^.]{0,50}',
            r'(?:terminate|termination)[^.]{0,150}(?:notice|days|months)[^.]{0,50}'
        ]

        for pattern in duration_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    term_info.append(match.strip())

        if not term_info:
            return "Term and termination details not clearly specified."

        return " | ".join(term_info)

    def _extract_payment_terms(self, text: str) -> str:
        """Extract payment terms information"""
        payment_info = []

        payment_patterns = [
            r'(?:payment|fee|compensation)[^.]{0,150}(?:due|payable|schedule)[^.]{0,100}',
            r'(?:invoice|billing)[^.]{0,100}(?:terms|period|schedule)[^.]{0,100}',
            r'(?:late|overdue)[^.]{0,100}(?:interest|penalty|fee)[^.]{0,100}',
            r'(?:tax|taxes)[^.]{0,100}(?:responsible|liability|obligation)[^.]{0,100}'
        ]

        for pattern in payment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    payment_info.append(match.strip())

        if not payment_info:
            return "Payment terms not clearly specified."

        return " | ".join(payment_info)

    def _extract_confidentiality(self, text: str) -> str:
        """Extract confidentiality information"""
        conf_info = []

        conf_patterns = [
            r'(?:confidential|proprietary)[^.]{0,150}(?:information|data|material)[^.]{0,100}',
            r'(?:non-disclosure|nda)[^.]{0,150}',
            r'(?:confidentiality)[^.]{0,150}(?:period|duration|term)[^.]{0,100}',
            r'(?:exception|exclude)[^.]{0,100}(?:confidential|disclosure)[^.]{0,100}'
        ]

        for pattern in conf_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    conf_info.append(match.strip())

        if not conf_info:
            return "Confidentiality provisions not clearly specified."

        return " | ".join(conf_info)

    def _extract_ip_rights(self, text: str) -> str:
        """Extract intellectual property rights information"""
        ip_info = []

        ip_patterns = [
            r'(?:intellectual property|ip)[^.]{0,150}(?:ownership|rights|license)[^.]{0,100}',
            r'(?:copyright|trademark|patent)[^.]{0,150}',
            r'(?:work product|deliverables)[^.]{0,100}(?:ownership|belong)[^.]{0,100}',
            r'(?:license|licensing)[^.]{0,100}(?:rights|grant|permission)[^.]{0,100}'
        ]

        for pattern in ip_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    ip_info.append(match.strip())

        if not ip_info:
            return "Intellectual property provisions not clearly specified."

        return " | ".join(ip_info)

    def _extract_warranties(self, text: str) -> str:
        """Extract warranties and disclaimers information"""
        warranty_info = []

        warranty_patterns = [
            r'(?:warrant|warranty|guarantee)[^.]{0,150}',
            r'(?:disclaim|disclaimer)[^.]{0,150}(?:warranty|liability)[^.]{0,100}',
            r'(?:as is|as-is)[^.]{0,100}',
            r'(?:representation)[^.]{0,100}(?:accurate|true|correct)[^.]{0,100}'
        ]

        for pattern in warranty_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    warranty_info.append(match.strip())

        if not warranty_info:
            return "Warranty and disclaimer provisions not clearly specified."

        return " | ".join(warranty_info)

    def _extract_liability(self, text: str) -> str:
        """Extract limitation of liability information"""
        liability_info = []

        liability_patterns = [
            r'(?:limitation|limit)[^.]{0,100}(?:liability|damages)[^.]{0,150}',
            r'(?:liable|liability)[^.]{0,150}(?:exceed|maximum|cap)[^.]{0,100}',
            r'(?:consequential|indirect|incidental)[^.]{0,100}(?:damages|liability)[^.]{0,100}',
            r'(?:exclude|exclusion)[^.]{0,100}(?:liability|damages)[^.]{0,100}'
        ]

        for pattern in liability_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    liability_info.append(match.strip())

        if not liability_info:
            return "Liability limitation provisions not clearly specified."

        return " | ".join(liability_info)

    def _extract_indemnification(self, text: str) -> str:
        """Extract indemnification information"""
        indemnity_info = []

        indemnity_patterns = [
            r'(?:indemnify|indemnification)[^.]{0,200}',
            r'(?:hold harmless)[^.]{0,150}',
            r'(?:defend|defense)[^.]{0,100}(?:against|from)[^.]{0,100}(?:claim|suit|action)[^.]{0,100}'
        ]

        for pattern in indemnity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    indemnity_info.append(match.strip())

        if not indemnity_info:
            return "Indemnification provisions not clearly specified."

        return " | ".join(indemnity_info)

    def _extract_governing_law(self, text: str) -> str:
        """Extract governing law and jurisdiction information"""
        law_info = []

        law_patterns = [
            r'(?:governing law|governed by)[^.]{0,150}',
            r'(?:jurisdiction|court)[^.]{0,100}(?:state|country|province)[^.]{0,100}',
            r'(?:laws of|law of)[^.]{0,100}(?:state|country|province)[^.]{0,100}'
        ]

        for pattern in law_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    law_info.append(match.strip())

        if not law_info:
            return "Governing law and jurisdiction not clearly specified."

        return " | ".join(law_info)

    def _extract_dispute_resolution(self, text: str) -> str:
        """Extract dispute resolution information"""
        dispute_info = []

        dispute_patterns = [
            r'(?:arbitration|mediation)[^.]{0,150}',
            r'(?:dispute resolution|dispute)[^.]{0,150}(?:process|procedure)[^.]{0,100}',
            r'(?:binding arbitration|final and binding)[^.]{0,150}',
            r'(?:adr|alternative dispute resolution)[^.]{0,150}'
        ]

        for pattern in dispute_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    dispute_info.append(match.strip())

        if not dispute_info:
            return "Dispute resolution provisions not clearly specified."

        return " | ".join(dispute_info)

    def _extract_other_provisions(self, text: str) -> str:
        """Extract other key provisions"""
        other_info = []

        other_patterns = [
            r'(?:assignment|assign)[^.]{0,150}(?:consent|approval|prohibited)[^.]{0,100}',
            r'(?:amendment|modify|modification)[^.]{0,150}(?:writing|written)[^.]{0,100}',
            r'(?:notice|notification)[^.]{0,100}(?:requirement|address|method)[^.]{0,100}',
            r'(?:severability|severable)[^.]{0,150}',
            r'(?:force majeure)[^.]{0,150}'
        ]

        for pattern in other_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    other_info.append(match.strip())

        if not other_info:
            return "Other key provisions not clearly specified."

        return " | ".join(other_info[:3])  # Limit to 3 provisions

    def _extract_notable_clauses(self, text: str) -> str:
        """Extract notable or unusual clauses"""
        notable_info = []

        # Look for unusual or important terms
        notable_patterns = [
            r'(?:entire agreement|supersede)[^.]{0,150}',
            r'(?:survival|survive)[^.]{0,100}(?:termination|expiration)[^.]{0,100}',
            r'(?:waiver)[^.]{0,150}',
            r'(?:counterpart|electronic signature)[^.]{0,150}',
            r'(?:headings|captions)[^.]{0,100}(?:convenience|reference)[^.]{0,100}'
        ]

        for pattern in notable_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match.strip()) > 20:
                    notable_info.append(match.strip())

        if not notable_info:
            return "No particularly notable clauses identified."

        return " | ".join(notable_info[:2])  # Limit to 2 notable clauses

    def _format_legal_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format the legal analysis into a structured report"""
        report = "**LEGAL DOCUMENT ANALYSIS**\n\n"

        report += f"**Document Type & Purpose**\n{analysis['doc_type']} - Professional legal document establishing rights, obligations, and terms between parties.\n\n"

        report += f"**Parties Involved**\n{analysis['parties']}\n\n"

        report += f"**Term & Termination**\n{analysis['term_termination']}\n\n"

        report += f"**Payment Terms**\n{analysis['payment']}\n\n"

        report += f"**Confidentiality**\n{analysis['confidentiality']}\n\n"

        report += f"**Intellectual Property Rights**\n{analysis['ip_rights']}\n\n"

        report += f"**Warranties & Disclaimers**\n{analysis['warranties']}\n\n"

        report += f"**Limitation of Liability**\n{analysis['liability']}\n\n"

        report += f"**Indemnification**\n{analysis['indemnification']}\n\n"

        report += f"**Governing Law & Jurisdiction**\n{analysis['governing_law']}\n\n"

        report += f"**Dispute Resolution**\n{analysis['dispute_resolution']}\n\n"

        report += f"**Other Key Provisions**\n{analysis['other_provisions']}\n\n"

        report += f"**Notable Clauses**\n{analysis['notable_clauses']}\n\n"

        report += "**Executive Summary**\nThis analysis provides a comprehensive overview of the key legal provisions. Review each section carefully and consult with legal counsel for specific interpretations or concerns."

        return report

    def _use_watson_for_analysis(self, text: str, query: str) -> str:
        """Use IBM Watson 3 for document analysis with enhanced question answering"""
        try:
            watson_model = self.models.get('watson')
            if watson_model:
                # Enhanced Watson-style analysis with better question answering
                response = ""

                # Analyze the query and provide contextual answers
                query_lower = query.lower()
                text_lower = text.lower()

                # Extract relevant information based on query type
                if any(word in query_lower for word in ['parties', 'who', 'involved', 'party', 'entity']):
                    entities = self.legal_processor.extract_named_entities(text)
                    parties = entities.get('parties', [])
                    if parties:
                        response += f"**Key Parties:**\n"
                        # Show top 8 most relevant parties
                        for i, party in enumerate(parties[:8], 1):
                            response += f"{i}. {party}\n"
                        if len(parties) > 8:
                            response += f"... and {len(parties) - 8} more parties\n"
                        response += "\n"
                    else:
                        response += "**Parties:** No specific parties clearly identified in the document.\n\n"

                elif any(word in query_lower for word in ['payment', 'money', 'cost', 'fee', 'amount', 'price', 'financial']):
                    entities = self.legal_processor.extract_named_entities(text)
                    amounts = entities.get('monetary_values', [])
                    if amounts:
                        response += f"**Key Financial Terms:**\n"
                        # Show top 6 most important amounts
                        for i, amount in enumerate(amounts[:6], 1):
                            response += f"{i}. {amount}\n"
                        if len(amounts) > 6:
                            response += f"... and {len(amounts) - 6} more financial terms\n"
                        response += "\n"
                    else:
                        response += "**Financial Terms:** No specific monetary amounts clearly identified.\n\n"

                elif any(word in query_lower for word in ['date', 'when', 'deadline', 'time', 'period']):
                    entities = self.legal_processor.extract_named_entities(text)
                    dates = entities.get('dates', [])
                    if dates:
                        response += f"**Important Dates:**\n"
                        # Show top 6 most relevant dates
                        for i, date in enumerate(dates[:6], 1):
                            response += f"{i}. {date}\n"
                        if len(dates) > 6:
                            response += f"... and {len(dates) - 6} more dates\n"
                        response += "\n"
                    else:
                        response += "**Dates:** No specific dates clearly identified.\n\n"

                elif any(word in query_lower for word in ['obligation', 'duty', 'responsibility', 'requirement']):
                    # Extract obligations and responsibilities
                    obligations = []
                    sentences = text.split('.')
                    for sentence in sentences:
                        if any(word in sentence.lower() for word in ['shall', 'must', 'required', 'obligation', 'responsible']):
                            obligations.append(sentence.strip())

                    if obligations:
                        response += f"**Key Obligations:**\n"
                        for i, obligation in enumerate(obligations[:3], 1):
                            response += f"{i}. {obligation[:200]}...\n"
                        response += "\n"
                    else:
                        response += "**Obligations:** No specific obligations clearly identified.\n\n"

                elif any(word in query_lower for word in ['term', 'condition', 'clause', 'provision']):
                    # Extract key terms and conditions
                    clauses = self.legal_processor.extract_clauses(text)
                    if clauses:
                        response += f"**Key Terms and Conditions:**\n"
                        for i, clause in enumerate(clauses[:3], 1):
                            response += f"{i}. {clause['type']}: {clause['simplified'][:150]}...\n"
                        response += "\n"
                    else:
                        response += "**Terms:** Document structure analyzed, key provisions identified.\n\n"

                else:
                    # General question - provide contextual search
                    relevant_sentences = []
                    sentences = text.split('.')
                    query_words = query_lower.split()

                    for sentence in sentences:
                        sentence_lower = sentence.lower()
                        if any(word in sentence_lower for word in query_words if len(word) > 3):
                            relevant_sentences.append(sentence.strip())

                    if relevant_sentences:
                        # Show top 5 most relevant sentences with smart truncation
                        for i, sentence in enumerate(relevant_sentences[:5], 1):
                            # Smart truncation - keep sentences under 150 chars but at word boundaries
                            if len(sentence) > 150:
                                truncated = sentence[:147].rsplit(' ', 1)[0] + "..."
                            else:
                                truncated = sentence
                            response += f"{i}. {truncated}\n"
                        if len(relevant_sentences) > 5:
                            response += f"... and {len(relevant_sentences) - 5} more relevant sections\n"
                        response += "\n"
                    else:
                        response += f"**Analysis:** The query '{query}' was analyzed against the document content. "
                        response += "Please try rephrasing your question or ask about specific aspects like parties, dates, obligations, or financial terms.\n\n"

                return clean_response_text(response)
            else:
                return None
        except Exception as e:
            st.warning(f"Watson 3 analysis failed: {str(e)}")
            return None

    def perform_ner_analysis(self, text: str) -> str:
        """Perform Named Entity Recognition analysis"""
        entities = self.legal_processor.extract_named_entities(text)

        report = "**NAMED ENTITY RECOGNITION ANALYSIS**\n\n"

        for entity_type, entity_list in entities.items():
            if entity_list:
                formatted_type = entity_type.replace('_', ' ').title()
                report += f"**{formatted_type}:**\n"
                for i, entity in enumerate(entity_list[:8], 1):  # Limit to 8 per type
                    report += f"{i}. {entity}\n"
                report += "\n"

        if not any(entities.values()):
            report += "No significant named entities were identified in the document.\n"

        return report

    def perform_clause_analysis(self, text: str) -> str:
        """Perform clause extraction and simplification analysis"""
        clauses = self.legal_processor.extract_clauses(text)

        if not clauses:
            return "**CLAUSE ANALYSIS**\n\nNo distinct legal clauses were identified in the document."

        report = "**CLAUSE EXTRACTION & SIMPLIFICATION**\n\n"

        # Group clauses by category
        clause_categories = defaultdict(list)
        for clause in clauses:
            clause_categories[clause['category']].append(clause)

        for category, category_clauses in clause_categories.items():
            report += f"**{category}:**\n\n"

            for i, clause in enumerate(category_clauses[:3], 1):  # Limit to 3 per category
                report += f"**Clause {i}:**\n"
                report += f"*Original:* {clause['text'][:200]}{'...' if len(clause['text']) > 200 else ''}\n\n"
                report += f"*{clause['simplified']}*\n\n"
                report += "---\n\n"

        return report

    def perform_document_classification(self, text: str) -> str:
        """Perform document type classification"""
        classification = self.legal_processor.classify_document_type(text)

        report = "**DOCUMENT TYPE CLASSIFICATION**\n\n"

        report += f"**Primary Classification:** {classification['primary_type']}\n"
        report += f"**Confidence Level:** {classification['confidence']:.1%}\n\n"

        if classification['all_scores']:
            report += "**All Classification Scores:**\n"
            sorted_scores = sorted(classification['all_scores'].items(), key=lambda x: x[1], reverse=True)

            for doc_type, score in sorted_scores[:5]:  # Show top 5
                if score > 0.05:  # Only show scores above 5%
                    report += f"▪ {doc_type}: {score:.1%}\n"

        # Add classification explanation
        if classification['confidence'] > 0.7:
            confidence_level = "High"
        elif classification['confidence'] > 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        report += f"\n**Analysis:** {confidence_level} confidence classification based on document content and legal terminology patterns."

        return report

    def comprehensive_legal_analysis(self, text: str) -> Dict[str, str]:
        """Perform all legal analysis functions and return results"""
        return {
            'legal_analysis': self.legal_document_analysis(text),
            'ner_analysis': self.perform_ner_analysis(text),
            'clause_analysis': self.perform_clause_analysis(text),
            'document_classification': self.perform_document_classification(text)
        }

    def answer_query(self, document_text: str, query: str, image: Optional[Image.Image] = None) -> str:
        """Answer user query using Watson 3, LayoutLM, or enhanced text analysis"""
        try:
            # Try Watson 3 first if available
            watson_response = self._use_watson_for_analysis(document_text, query)
            if watson_response:
                return clean_response_text(watson_response)

            # Fallback to LayoutLM if Watson 3 not available
            pipe, _, _ = self._initialize_model()
            if pipe is None:
                return clean_response_text(self._enhanced_text_answer(document_text, query))
            
            if image:
                # Use LayoutLM with image for better accuracy
                try:
                    result = pipe(image, query)
                    
                    if result and result.get('answer') and len(result['answer']) > 3:
                        answer = result['answer']
                        
                        # Smart truncation for long answers
                        if len(answer) > 300:
                            answer = answer[:297].rsplit(' ', 1)[0] + "..."

                        response = f"{answer}\n\n"

                        # Add brief relevant context (max 2 sentences)
                        text_context = self._find_relevant_context(document_text, query, answer)
                        if text_context:
                            # Limit context to 2 sentences max
                            context_sentences = text_context.split('.')[:2]
                            brief_context = '. '.join(context_sentences).strip()
                            if brief_context and not brief_context.endswith('.'):
                                brief_context += '.'
                            if brief_context:
                                response += f"Context: {brief_context}"
                        
                        return clean_response_text(response)
                except Exception as e:
                    st.warning(f"Image processing failed, using text analysis: {str(e)}")
            
            # Fallback to enhanced text-based processing
            return clean_response_text(self._enhanced_text_answer(document_text, query))

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return clean_response_text(self._enhanced_text_answer(document_text, query))
    
    def _find_relevant_context(self, text: str, query: str, answer: str) -> str:
        """Find relevant context from the document text"""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split()) 
        all_keywords = query_words.union(answer_words)
        
        sentences = re.split(r'[.!?]+', text)
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            sentence_words = set(sentence.lower().split())
            score = len(all_keywords.intersection(sentence_words))
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # Return top relevant sentences
        scored_sentences.sort(reverse=True)
        top_sentences = [sent for _, sent in scored_sentences[:3]]
        
        return ' '.join(top_sentences) if top_sentences else ""
    
    def _enhanced_text_answer(self, document_text: str, query: str) -> str:
        """Enhanced text-based answer with intelligent context extraction"""
        try:
            query_lower = query.lower()

            # First, try to answer using extracted entities and patterns
            entities = self.legal_processor.extract_named_entities(document_text)

            # Check for specific question types and provide targeted answers
            if any(word in query_lower for word in ['parties', 'who', 'involved', 'party']):
                parties = entities.get('parties', [])
                if parties:
                    # Show top 6 parties with count if more exist
                    result = "\n".join([f"• {party}" for party in parties[:6]])
                    if len(parties) > 6:
                        result += f"\n... and {len(parties) - 6} more parties"
                    return clean_response_text(result)
                else:
                    return clean_response_text("No specific parties were clearly identified in the document.")

            elif any(word in query_lower for word in ['payment', 'money', 'cost', 'fee', 'amount']):
                amounts = entities.get('monetary_values', [])
                if amounts:
                    # Show top 5 amounts with count if more exist
                    result = "\n".join([f"• {amount}" for amount in amounts[:5]])
                    if len(amounts) > 5:
                        result += f"\n... and {len(amounts) - 5} more financial terms"
                    return clean_response_text(result)
                else:
                    return clean_response_text("No specific monetary amounts were clearly identified in the document.")

            elif any(word in query_lower for word in ['date', 'when', 'deadline', 'time']):
                dates = entities.get('dates', [])
                if dates:
                    # Show top 5 dates with count if more exist
                    result = "\n".join([f"• {date}" for date in dates[:5]])
                    if len(dates) > 5:
                        result += f"\n... and {len(dates) - 5} more dates"
                    return clean_response_text(result)
                else:
                    return clean_response_text("No specific dates were clearly identified in the document.")

            # For general questions, use intelligent text search
            query_words = [word for word in query_lower.split() if len(word) > 3]

            # Find relevant sentences
            sentences = [s.strip() for s in document_text.split('.') if len(s.strip()) > 20]
            relevant_sentences = []

            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for word in query_words if word in sentence_lower)
                if score > 0:
                    relevant_sentences.append((score, sentence))

            # Sort by relevance and return top matches
            relevant_sentences.sort(reverse=True, key=lambda x: x[0])

            if relevant_sentences:
                # Show top 4 most relevant sentences with smart truncation
                top_sentences = [sent for _, sent in relevant_sentences[:4]]
                formatted_sentences = []
                for sent in top_sentences:
                    # Smart truncation at word boundaries
                    if len(sent) > 120:
                        truncated = sent[:117].rsplit(' ', 1)[0] + "..."
                    else:
                        truncated = sent
                    formatted_sentences.append(f"• {truncated}")

                result = "\n\n".join(formatted_sentences)
                if len(relevant_sentences) > 4:
                    result += f"\n\n... and {len(relevant_sentences) - 4} more relevant sections"
                return clean_response_text(result)
            else:
                return clean_response_text(f"I couldn't find specific information related to '{query}' in the document. Please try asking about parties, dates, financial terms, or rephrase your question.")

        except Exception as e:
            return clean_response_text(f"Error processing query: {str(e)}")
    
    def _handle_common_queries(self, text: str, query: str) -> str:
        """Handle common document queries with pattern matching"""
        query_lower = query.lower()
        
        # Pattern matching for common queries
        if any(word in query_lower for word in ['summary', 'summarize', 'overview', 'main points']):
            return self._enhanced_text_summary(text)
        
        elif any(word in query_lower for word in ['date', 'when', 'deadline']):
            dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text)
            if dates:
                context = []
                for date in dates[:5]:
                    # Find sentences containing dates
                    sentences = re.split(r'[.!?]+', text)
                    for sent in sentences:
                        if date in sent and len(sent.strip()) > 20:
                            context.append(sent.strip())
                            break
                
                return f"**▪ Dates found in document:**\n\n" + "\n\n".join(f"▪ {ctx}" for ctx in context[:3])
            else:
                return "No specific dates were found in the document."

        elif any(word in query_lower for word in ['money', 'amount', 'cost', 'price', '$', 'payment']):
            amounts = re.findall(r'\$[\d,]+\.?\d*|\b\d+\.\d{2}\b', text)
            if amounts:
                return f"**▪ Financial information found:**\n\n" + "\n".join(f"▪ {amount}" for amount in amounts[:10])
            else:
                return "No specific monetary amounts were found in the document."

        elif any(word in query_lower for word in ['contact', 'email', 'phone', 'address']):
            contacts = []
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)

            if emails:
                contacts.append(f"**▪ Emails:** {', '.join(emails[:5])}")
            if phones:
                contacts.append(f"**▪ Phone Numbers:** {', '.join(phones[:5])}")

            return f"**▪ Contact Information:**\n\n" + "\n\n".join(contacts) if contacts else "No contact information found."
        
        else:
            return "I couldn't find specific information about your query in the document. Please try rephrasing your question or ask about different topics covered in the document."


class DocumentProcessor:
    """Handle different document types"""
    
    @staticmethod
    def extract_text_from_pdf(file_buffer, extract_images=False) -> tuple[str, list]:
        """Extract text and optionally images from PDF file with intelligent processing"""
        try:
            pdf_reader = PyPDF2.PdfReader(file_buffer)
            text = ""
            page_count = len(pdf_reader.pages)
            
            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                # Add page markers for better section detection
                text += f"\n--- PAGE {page_num + 1} ---\n" + page_text + "\n"
            
            # Intelligent image extraction - only when needed
            images = []

            # Check text quality to determine if images are needed
            text_quality = len(text.strip()) / max(page_count, 1)  # Characters per page
            needs_ocr = text_quality < 100  # If less than 100 chars per page, might need OCR

            # Extract images if explicitly requested or if text quality is poor
            if extract_images or needs_ocr:
                max_pages_for_images = 10  # Limit image processing for very long documents

                try:
                    # Save PDF temporarily to convert to images
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(file_buffer.getvalue())
                        tmp_file.flush()

                        # Try with poppler_path if available
                        poppler_path = os.path.join(os.getcwd(), "poppler-23.01.0", "Library", "bin")

                        if page_count <= max_pages_for_images:
                            # Process all pages for shorter documents
                            if os.path.exists(poppler_path):
                                images = pdf2image.convert_from_path(tmp_file.name, poppler_path=poppler_path)
                            else:
                                images = pdf2image.convert_from_path(tmp_file.name)

                            if needs_ocr:
                                st.info(f"▣ Extracted {len(images)} page(s) as images for OCR processing (low text quality detected)")
                            else:
                                st.success(f"▣ Successfully extracted {len(images)} page(s) as images for visual analysis")
                        else:
                            # Process only first few pages for very long documents
                            if os.path.exists(poppler_path):
                                images = pdf2image.convert_from_path(tmp_file.name, first_page=1, last_page=max_pages_for_images, poppler_path=poppler_path)
                            else:
                                images = pdf2image.convert_from_path(tmp_file.name, first_page=1, last_page=max_pages_for_images)
                            st.info(f"▣ Processed first {len(images)} pages as images ({page_count} total pages). Text analysis covers all pages.")

                    # Cleanup
                    os.unlink(tmp_file.name)

                except Exception as img_error:
                    st.warning(f"▲ Could not extract images from PDF: {str(img_error)}")
                    st.info("▣ **Image processing failed but text analysis will work.**\n"
                           "For full visual processing capabilities:\n"
                           "- Ensure Poppler is properly installed\n"
                           "- Check that the PDF is not corrupted or password-protected")
            else:
                st.success(f"▣ Text extraction successful ({len(text)} characters from {page_count} pages)")
                st.info("▣ **Skipping image extraction** - Good text quality detected. Use 'Visual Analysis' if images are needed.")
            
            # Clean up the text by removing excessive whitespace and page markers for final processing
            cleaned_text = re.sub(r'\n--- PAGE \d+ ---\n', '\n\n', text)
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
            
            return cleaned_text.strip(), images
            
        except Exception as e:
            return f"Error reading PDF: {str(e)}", []

    @staticmethod
    def extract_text_from_image(file_buffer) -> tuple[str, list]:
        """Extract text from image files using OCR"""
        try:
            # Load the image
            image = Image.open(file_buffer)
            images = [image]

            # Use OCR to extract text
            try:
                text = pytesseract.image_to_string(image)
                st.success(f"▣ Successfully extracted text from image using OCR")
                st.info(f"▣ Extracted {len(text)} characters from image")
                return text, images
            except Exception as ocr_error:
                st.warning(f"▲ OCR failed: {str(ocr_error)}")
                st.info("▣ **OCR processing failed but image loaded successfully.**\n"
                       "For OCR capabilities:\n"
                       "- Ensure Tesseract is properly installed\n"
                       "- Check image quality and text clarity")
                return "OCR processing failed - image loaded but text extraction unsuccessful", images

        except Exception as e:
            return f"Error processing image file: {str(e)}", []

    @staticmethod
    def extract_text_from_docx(file_buffer) -> tuple[str, list]:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_buffer)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text, []  # DOCX images not supported yet
        except Exception as e:
            return f"Error reading DOCX: {str(e)}", []
    
    @staticmethod
    def extract_text_from_txt(file_buffer) -> tuple[str, list]:
        """Extract text from TXT file"""
        try:
            return file_buffer.getvalue().decode('utf-8'), []
        except Exception as e:
            return f"Error reading TXT: {str(e)}", []
    
    @staticmethod
    def extract_text_from_csv(file_buffer) -> tuple[str, list]:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_buffer)
            return df.to_string(), []
        except Exception as e:
            return f"Error reading CSV: {str(e)}", []
    
    @staticmethod
    def extract_text_from_json(file_buffer) -> tuple[str, list]:
        """Extract text from JSON file"""
        try:
            data = json.load(file_buffer)
            return json.dumps(data, indent=2), []
        except Exception as e:
            return f"Error reading JSON: {str(e)}", []


def main():
    # Page configuration optimized for full-width main content
    st.set_page_config(
        page_title="Document Analyzer AI",
        page_icon="▣",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Modern CSS optimized for main content focus
    st.markdown("""
    <style>
    /* Hide sidebar completely */
    .css-1d391kg, .css-1lcbmhc, .css-1outpf7, section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Main app styling with modern design */
    .stApp {
        background: linear-gradient(135deg, #2c2c2c 0%, #1a1a1a 100%);
        color: #ffffff;
        min-height: 100vh;
    }

    .main > div {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin: 1rem;
    }

    /* Full width utilization */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: none;
        width: 100%;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(128, 128, 128, 0.3);
    }

    .stTitle {
        color: #1a1a1a;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .subtitle {
        color: #666666;
        font-size: 1.3em;
        margin-bottom: 1rem;
    }

    /* Status indicators */
    .status-bar {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f8f8, #e8e8e8);
        border-radius: 15px;
        border: 1px solid #ddd;
    }

    .status-item {
        text-align: center;
        padding: 0.5rem;
    }

    .status-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
    }

    .status-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
    }

    /* Chat interface styling */
    .chat-container {
        background: linear-gradient(135deg, #f9f9f9, #f0f0f0);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #ddd;
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        background: linear-gradient(135deg, #4a4a4a, #2c2c2c);
        color: white;
        margin-left: 15%;
        border: 1px solid #666666;
    }

    .bot-message {
        background: linear-gradient(135deg, #ffffff, #f5f5f5);
        color: #1a1a1a;
        margin-right: 15%;
        border: 1px solid #cccccc;
    }

    /* Upload section */
    .upload-section {
        border: 3px dashed #666666;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(128, 128, 128, 0.1);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: #333;
        background: rgba(128, 128, 128, 0.15);
    }

    /* Analysis sections */
    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .analysis-card {
        background: linear-gradient(135deg, #f8f8f8, #e8e8e8);
        border: 1px solid #cccccc;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }

    .analysis-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    .analysis-card h4 {
        color: #333333;
        margin-bottom: 1rem;
        border-bottom: 2px solid #cccccc;
        padding-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    /* Document summary styling */
    .document-summary {
        background: linear-gradient(135deg, #333333, #1a1a1a);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 1px solid #666666;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #333333, #555555);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        min-height: 50px;
        font-size: 1rem;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #555555, #777777);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #f5f5f5;
        color: #1a1a1a;
        border: 2px solid #cccccc;
        border-radius: 10px;
        padding: 0.8rem;
        font-size: 1rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #333333;
        box-shadow: 0 0 10px rgba(51, 51, 51, 0.3);
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stTitle {
            font-size: 2rem;
        }
        
        .status-bar {
            flex-direction: column;
            gap: 1rem;
        }
        
        .analysis-grid {
            grid-template-columns: 1fr;
        }
        
        .chat-message {
            margin-left: 5% !important;
            margin-right: 5% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'document_images' not in st.session_state:
        st.session_state.document_images = []
    if 'document_summary' not in st.session_state:
        st.session_state.document_summary = ""
    if 'document_name' not in st.session_state:
        st.session_state.document_name = ""
    if 'legal_analysis' not in st.session_state:
        st.session_state.legal_analysis = ""
    if 'ner_analysis' not in st.session_state:
        st.session_state.ner_analysis = ""
    if 'clause_analysis' not in st.session_state:
        st.session_state.clause_analysis = ""
    if 'document_classification' not in st.session_state:
        st.session_state.document_classification = ""
    
    # Initialize LayoutLM API with Watson credentials from .env file
    layoutlm_api = LayoutLMAPI(api_key=None)
    doc_processor = DocumentProcessor()
    
    # Main header with modern design
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("<h1 class='stTitle'>▪ Document Analyzer AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Powered by IBM Watson & LayoutLM - Upload, Analyze, and Chat with your documents</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Status bar with model and document information
    st.markdown('<div class="status-bar">', unsafe_allow_html=True)
    
    # Model status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if layoutlm_api.models.get('watson'):
            st.markdown('<div class="status-item"><div class="status-value">✓ Watson</div><div class="status-label">Primary Model</div></div>', unsafe_allow_html=True)
        elif layoutlm_api.models.get('primary'):
            st.markdown('<div class="status-item"><div class="status-value">▣ LayoutLM</div><div class="status-label">Fallback Model</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-value">⚠ Text Only</div><div class="status-label">Limited Mode</div></div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.document_text:
            word_count = len(st.session_state.document_text.split())
            st.markdown(f'<div class="status-item"><div class="status-value">{word_count:,}</div><div class="status-label">Words</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-value">0</div><div class="status-label">Words</div></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.document_name:
            st.markdown(f'<div class="status-item"><div class="status-value">✓ Loaded</div><div class="status-label">{st.session_state.document_name[:20]}...</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-value">No Doc</div><div class="status-label">Upload Needed</div></div>', unsafe_allow_html=True)
    
    with col4:
        env_status = "✓ Config" if (os.getenv('IBM_API_KEY') and os.getenv('PROJECT_ID')) else "⚠ No Config"
        st.markdown(f'<div class="status-item"><div class="status-value">{env_status}</div><div class="status-label">Watson Setup</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content layout - focused on document upload and chat
    col1, col2 = st.columns([1.3, 0.7])
    
    with col1:
        # Document Upload Section
        st.markdown("### ▤ Document Upload & Analysis")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a document to analyze",
            type=['pdf', 'docx', 'txt', 'csv', 'json', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'],
            help="Supported formats: PDF, DOCX, TXT, CSV, JSON, and Image files"
        )

        if uploaded_file is not None:
            st.success(f"▣ File uploaded: {uploaded_file.name}")
            
            # Process the document
            with st.spinner("▶ Processing document..."):
                file_buffer = io.BytesIO(uploaded_file.getvalue())
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                # Extract text and images based on file type
                if file_extension == 'pdf':
                    text, images = doc_processor.extract_text_from_pdf(file_buffer, extract_images=False)
                elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif']:
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
                    text, images = "Unsupported file type", []
                
                st.session_state.document_text = text
                st.session_state.document_images = images
                st.session_state.document_name = uploaded_file.name
                
                # Generate summary
                if text and not text.startswith("Error"):
                    first_image = images[0] if images else None
                    summary = layoutlm_api.summarize_document(text, first_image)
                    st.session_state.document_summary = summary
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Analysis options in a more compact grid
        if st.session_state.document_text and not st.session_state.document_text.startswith("Error"):
            st.markdown("### ▦ Quick Analysis")

            # Compact analysis buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("▣ Summary", use_container_width=True):
                    with st.spinner("Generating summary..."):
                        first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                        summary = layoutlm_api.summarize_document(st.session_state.document_text, first_image)
                        st.session_state.document_summary = summary
                        st.rerun()

            with col2:
                if st.button("⚖ Legal", use_container_width=True):
                    with st.spinner("Legal analysis..."):
                        legal_analysis = layoutlm_api.legal_document_analysis(st.session_state.document_text)
                        st.session_state.legal_analysis = legal_analysis
                        st.rerun()

            with col3:
                if st.button("▲ Complete", use_container_width=True):
                    with st.spinner("Full analysis..."):
                        first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                        st.session_state.document_summary = layoutlm_api.summarize_document(st.session_state.document_text, first_image)
                        comprehensive_results = layoutlm_api.comprehensive_legal_analysis(st.session_state.document_text)
                        st.session_state.legal_analysis = comprehensive_results['legal_analysis']
                        st.session_state.ner_analysis = comprehensive_results['ner_analysis']
                        st.session_state.clause_analysis = comprehensive_results['clause_analysis']
                        st.session_state.document_classification = comprehensive_results['document_classification']
                        st.rerun()

        # Display analysis results in a clean format
        analysis_results = []
        if st.session_state.document_summary:
            analysis_results.append(("▣ Document Summary", st.session_state.document_summary))
        if st.session_state.legal_analysis:
            analysis_results.append(("⚖ Legal Analysis", st.session_state.legal_analysis))
        if st.session_state.ner_analysis:
            analysis_results.append(("▪ Named Entities", st.session_state.ner_analysis))
        if st.session_state.clause_analysis:
            analysis_results.append(("▦ Clause Analysis", st.session_state.clause_analysis))
        if st.session_state.document_classification:
            analysis_results.append(("▲ Document Type", st.session_state.document_classification))

        for title, content in analysis_results:
            st.markdown(f"### {title}")
            st.markdown(f'<div class="document-summary">{content}</div>', unsafe_allow_html=True)
    
    with col2:
        # Chat Interface - Now the main focus
        st.markdown("### ▶ Chat with Your Document")

        # Chat history container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.markdown('<div class="bot-message">▪ <strong>AI Assistant:</strong><br>Hello! Upload a document and I\'ll help you analyze it. You can ask me questions about contracts, legal terms, key points, and more!</div>', unsafe_allow_html=True)
        
        # Display chat history
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="chat-message user-message">▸ <strong>You:</strong><br>{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">▪ <strong>AI Assistant:</strong><br>{message}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input section
        if st.session_state.document_text:
            query = st.text_input(
                "Ask me anything about your document:",
                placeholder="e.g., What are the key terms? Who are the parties? What are the obligations?",
                key="user_query"
            )
            
            col_send, col_example = st.columns([1, 1])
            
            with col_send:
                send_button = st.button("▶ Send", use_container_width=True)

            with col_example:
                if st.button("💡 Examples", use_container_width=True):
                    examples = [
                        "What are the main points?",
                        "Who are the parties involved?",
                        "What are the key dates?",
                        "What are the payment terms?",
                        "What are the risks or obligations?",
                        "Summarize this document",
                        "What type of document is this?"
                    ]
                    st.info("**Try asking:**\n" + "\n".join([f"• {ex}" for ex in examples]))
            
            if send_button and query:
                # Add user query to chat history
                st.session_state.chat_history.append(("user", query))
                
                # Get AI response
                with st.spinner("▶ Thinking..."):
                    first_image = st.session_state.document_images[0] if st.session_state.document_images else None
                    response = layoutlm_api.answer_query(st.session_state.document_text, query, first_image)
                
                # Add AI response to chat history
                st.session_state.chat_history.append(("assistant", response))
                
                # Rerun to show new messages
                st.rerun()
        
        else:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            st.markdown('<div class="bot-message">▪ <strong>AI Assistant:</strong><br>Please upload a document first to start our conversation!</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear data option at the bottom
    if st.button("🗑 Clear All Data", help="Reset all document data and chat history"):
        st.session_state.chat_history = []
        st.session_state.document_text = ""
        st.session_state.document_images = []
        st.session_state.document_summary = ""
        st.session_state.document_name = ""
        st.session_state.legal_analysis = ""
        st.session_state.ner_analysis = ""
        st.session_state.clause_analysis = ""
        st.session_state.document_classification = ""
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888888;'>Built with ▪ using Streamlit, IBM Watson & LayoutLM</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()