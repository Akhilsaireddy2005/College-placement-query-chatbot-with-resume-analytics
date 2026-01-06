"""
NLP Preprocessing Module
Handles tokenization, stop-word removal, lemmatization, and POS tagging
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class NLPPreprocessor:
    """NLP preprocessing class for text cleaning and processing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # Add custom stop words for placement context
        self.stop_words.update(['placement', 'company', 'job', 'role', 'position'])
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        """Tokenize text into words"""
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stop words from tokens"""
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens
    
    def lemmatize(self, tokens):
        """Lemmatize tokens"""
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized
    
    def pos_tagging(self, tokens):
        """Perform POS tagging"""
        tagged = pos_tag(tokens)
        return tagged
    
    def preprocess(self, text, remove_stopwords=True, lemmatize=True, pos_tag=False):
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text string
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize tokens
            pos_tag: Whether to include POS tagging
        
        Returns:
            Processed tokens or tagged tokens
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stop words
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # POS tagging
        if pos_tag:
            tokens = self.pos_tagging(tokens)
        
        return tokens
    
    def preprocess_for_embedding(self, text):
        """Preprocess text for word embedding input"""
        tokens = self.preprocess(text, remove_stopwords=True, lemmatize=True, pos_tag=False)
        return ' '.join(tokens)
    
    def extract_nouns(self, text):
        """Extract nouns from text using POS tagging"""
        tokens = self.tokenize(text)
        tagged = self.pos_tagging(tokens)
        nouns = [word for word, pos in tagged if pos.startswith('NN')]
        return nouns
    
    def extract_verbs(self, text):
        """Extract verbs from text using POS tagging"""
        tokens = self.tokenize(text)
        tagged = self.pos_tagging(tokens)
        verbs = [word for word, pos in tagged if pos.startswith('VB')]
        return verbs

