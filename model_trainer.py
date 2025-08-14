"""
Model Training and Fine-tuning Module
Provides capabilities to train and improve the document analysis models
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

class ModelTrainer:
    """Class for training and fine-tuning document analysis models"""
    
    def __init__(self):
        self.training_data = []
        self.models = {}
        self.vectorizers = {}
        self.training_history = []
        
    def collect_training_data(self, document_text: str, question: str, 
                            correct_answer: str, model_answer: str, 
                            feedback: str = "correct"):
        """Collect training data from user interactions"""
        training_sample = {
            'timestamp': datetime.now().isoformat(),
            'document_text': document_text[:1000],  # Truncate for storage
            'question': question,
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'feedback': feedback,  # 'correct', 'incorrect', 'partial'
            'document_length': len(document_text),
            'question_type': self._classify_question_type(question)
        }
        
        self.training_data.append(training_sample)
        return training_sample
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question for training purposes"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['who', 'party', 'parties', 'entity']):
            return 'parties'
        elif any(word in question_lower for word in ['when', 'date', 'deadline', 'time']):
            return 'dates'
        elif any(word in question_lower for word in ['how much', 'amount', 'payment', 'cost', 'fee']):
            return 'financial'
        elif any(word in question_lower for word in ['what', 'obligation', 'responsibility', 'duty']):
            return 'obligations'
        elif any(word in question_lower for word in ['where', 'location', 'jurisdiction']):
            return 'location'
        else:
            return 'general'
    
    def train_question_classifier(self) -> Dict[str, Any]:
        """Train a model to classify question types"""
        if len(self.training_data) < 10:
            return {"error": "Need at least 10 training samples"}
        
        # Prepare training data
        questions = [sample['question'] for sample in self.training_data]
        question_types = [sample['question_type'] for sample in self.training_data]
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            questions, question_types, test_size=0.2, random_state=42
        )
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store model
        self.models['question_classifier'] = pipeline
        
        training_result = {
            'model_type': 'question_classifier',
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.training_history.append(training_result)
        return training_result
    
    def train_answer_quality_predictor(self) -> Dict[str, Any]:
        """Train a model to predict answer quality"""
        if len(self.training_data) < 10:
            return {"error": "Need at least 10 training samples"}
        
        # Prepare features
        features = []
        labels = []
        
        for sample in self.training_data:
            # Extract features
            feature_vector = [
                len(sample['model_answer']),  # Answer length
                len(sample['question']),      # Question length
                sample['document_length'],    # Document length
                len(sample['model_answer'].split()),  # Answer word count
                1 if sample['question_type'] == 'parties' else 0,
                1 if sample['question_type'] == 'dates' else 0,
                1 if sample['question_type'] == 'financial' else 0,
                1 if sample['question_type'] == 'obligations' else 0,
            ]
            
            features.append(feature_vector)
            
            # Convert feedback to numeric
            feedback_score = {
                'correct': 1,
                'partial': 0.5,
                'incorrect': 0
            }.get(sample['feedback'], 0)
            
            labels.append(feedback_score)
        
        # Train simple model
        from sklearn.ensemble import RandomForestRegressor
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = ((y_test - y_pred) ** 2).mean()
        
        self.models['answer_quality'] = model
        
        training_result = {
            'model_type': 'answer_quality_predictor',
            'mse': mse,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(training_result)
        return training_result
    
    def improve_entity_extraction(self) -> Dict[str, Any]:
        """Improve entity extraction based on user feedback"""
        # Collect patterns from correct answers
        entity_patterns = {
            'parties': [],
            'dates': [],
            'amounts': []
        }
        
        for sample in self.training_data:
            if sample['feedback'] == 'correct':
                question_type = sample['question_type']
                answer = sample['correct_answer']
                
                if question_type == 'parties':
                    # Extract party patterns
                    parties = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', answer)
                    entity_patterns['parties'].extend(parties)
                
                elif question_type == 'dates':
                    # Extract date patterns
                    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\w+ \d{1,2}, \d{4}\b', answer)
                    entity_patterns['dates'].extend(dates)
                
                elif question_type == 'financial':
                    # Extract amount patterns
                    amounts = re.findall(r'\$[\d,]+\.?\d*|\b\d+\s*dollars?\b', answer)
                    entity_patterns['amounts'].extend(amounts)
        
        # Create improved patterns
        improved_patterns = {}
        for entity_type, patterns in entity_patterns.items():
            if patterns:
                # Get unique patterns
                unique_patterns = list(set(patterns))
                improved_patterns[entity_type] = unique_patterns[:10]  # Top 10 patterns
        
        training_result = {
            'model_type': 'entity_extraction_improvement',
            'improved_patterns': improved_patterns,
            'training_samples': len(self.training_data),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(training_result)
        return training_result
    
    def save_training_data(self, filename: str = "training_data.json"):
        """Save training data to file"""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'training_data': self.training_data,
                    'training_history': self.training_history
                }, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving training data: {str(e)}")
            return False
    
    def load_training_data(self, filename: str = "training_data.json"):
        """Load training data from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.training_data = data.get('training_data', [])
                    self.training_history = data.get('training_history', [])
                return True
        except Exception as e:
            st.error(f"Error loading training data: {str(e)}")
        return False
    
    def save_models(self, directory: str = "trained_models"):
        """Save trained models"""
        try:
            os.makedirs(directory, exist_ok=True)
            
            for model_name, model in self.models.items():
                filename = os.path.join(directory, f"{model_name}.pkl")
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
            
            return True
        except Exception as e:
            st.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, directory: str = "trained_models"):
        """Load trained models"""
        try:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if filename.endswith('.pkl'):
                        model_name = filename[:-4]  # Remove .pkl extension
                        filepath = os.path.join(directory, filename)
                        with open(filepath, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
        return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.training_data:
            return {"message": "No training data available"}
        
        stats = {
            'total_samples': len(self.training_data),
            'question_types': {},
            'feedback_distribution': {},
            'recent_training': len([s for s in self.training_data 
                                  if (datetime.now() - datetime.fromisoformat(s['timestamp'])).days < 7])
        }
        
        # Count question types
        for sample in self.training_data:
            q_type = sample['question_type']
            stats['question_types'][q_type] = stats['question_types'].get(q_type, 0) + 1
        
        # Count feedback
        for sample in self.training_data:
            feedback = sample['feedback']
            stats['feedback_distribution'][feedback] = stats['feedback_distribution'].get(feedback, 0) + 1
        
        return stats
