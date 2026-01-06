"""
Main Chatbot Module
CNN-based conversational chatbot for placement queries
"""

from models.intent_classifier import IntentClassifier
from chatbot.responses import ResponseGenerator
from nlp.preprocessing import NLPPreprocessor


class PlacementChatbot:
    """Main chatbot class for handling placement queries"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
        self.preprocessor = NLPPreprocessor()
        self.initialized = False
    
    def initialize(self):
        """Initialize the chatbot by loading models"""
        try:
            self.intent_classifier.load_model()
            self.initialized = True
            print("Chatbot initialized successfully")
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            print("Please train the models first using train_models.py")
            self.initialized = False
    
    def process_query(self, user_query):
        """
        Process user query and generate response
        
        Args:
            user_query: User's text query
        
        Returns:
            Dictionary with response and metadata
        """
        if not self.initialized:
            return {
                'response': "Chatbot is not initialized. Please train the models first.",
                'intent': None,
                'confidence': 0.0
            }
        
        if not user_query or not user_query.strip():
            return {
                'response': "Please enter a valid query.",
                'intent': None,
                'confidence': 0.0
            }
        
        # Preprocess query
        processed_query = self.preprocessor.clean_text(user_query)
        
        # Classify intent
        try:
            intent, confidence = self.intent_classifier.predict(processed_query)
        except Exception as e:
            print(f"Error in intent classification: {e}")
            intent = None
            confidence = 0.0
        
        # If confidence is low or intent missing, fall back to open-ended chat
        if not intent or confidence < 0.35:
            intent = 'fallback'
        
        # Always generate a friendly response
        response = self.response_generator.generate_response(intent, user_query)
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'original_query': user_query
        }
    
    def chat(self, user_query):
        """
        Simple chat interface method
        
        Args:
            user_query: User's text query
        
        Returns:
            Response string
        """
        result = self.process_query(user_query)
        return result['response']
