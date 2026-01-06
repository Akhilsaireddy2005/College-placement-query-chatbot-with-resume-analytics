"""
Word Embeddings Module
Handles Word2Vec, GloVe, and trainable embeddings
"""

import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nlp.preprocessing import NLPPreprocessor
from config import WORD2VEC_MODEL_PATH, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH


class WordEmbeddings:
    """Word embedding utilities for text vectorization"""
    
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.preprocessor = NLPPreprocessor()
        self.word2vec_model = None
        self.vocab_size = 0
        self.word_index = {}
        self.index_word = {}
    
    def train_word2vec(self, texts, min_count=2, window=5, workers=4):
        """
        Train Word2Vec model on given texts
        
        Args:
            texts: List of text strings
            min_count: Minimum word frequency
            window: Context window size
            workers: Number of worker threads
        """
        # Preprocess texts
        processed_texts = []
        for text in texts:
            tokens = self.preprocessor.preprocess(text, remove_stopwords=True, lemmatize=True)
            processed_texts.append(tokens)
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            processed_texts,
            vector_size=self.embedding_dim,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1  # Skip-gram
        )
        
        # Build vocabulary index
        self._build_vocab_index()
        
        # Save model
        os.makedirs(os.path.dirname(WORD2VEC_MODEL_PATH), exist_ok=True)
        self.word2vec_model.save(WORD2VEC_MODEL_PATH)
        print(f"Word2Vec model saved to {WORD2VEC_MODEL_PATH}")
    
    def load_word2vec(self, model_path=None):
        """Load pre-trained Word2Vec model"""
        if model_path is None:
            model_path = WORD2VEC_MODEL_PATH
        
        if os.path.exists(model_path):
            self.word2vec_model = Word2Vec.load(model_path)
            self._build_vocab_index()
            print(f"Word2Vec model loaded from {model_path}")
        else:
            print(f"Model not found at {model_path}. Train a new model first.")
    
    def _build_vocab_index(self):
        """Build word index from Word2Vec vocabulary"""
        if self.word2vec_model:
            self.word_index = {word: idx + 1 for idx, word in enumerate(self.word2vec_model.wv.index_to_key)}
            self.index_word = {idx + 1: word for word, idx in self.word_index.items()}
            self.vocab_size = len(self.word_index) + 1  # +1 for padding token (0)
    
    def get_embedding_matrix(self):
        """
        Create embedding matrix for use in neural networks
        
        Returns:
            numpy array of shape (vocab_size, embedding_dim)
        """
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not loaded or trained")
        
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        
        for word, idx in self.word_index.items():
            if word in self.word2vec_model.wv:
                embedding_matrix[idx] = self.word2vec_model.wv[word]
        
        return embedding_matrix
    
    def text_to_sequence(self, text, max_length=MAX_SEQUENCE_LENGTH):
        """
        Convert text to sequence of word indices
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
        
        Returns:
            List of word indices
        """
        if not self.word_index:
            raise ValueError("Vocabulary not built. Train or load a model first.")
        
        tokens = self.preprocessor.preprocess(text, remove_stopwords=True, lemmatize=True)
        sequence = [self.word_index.get(token, 0) for token in tokens[:max_length]]
        
        # Pad sequence
        while len(sequence) < max_length:
            sequence.append(0)
        
        return sequence[:max_length]
    
    def get_word_vector(self, word):
        """Get embedding vector for a word"""
        if self.word2vec_model and word in self.word2vec_model.wv:
            return self.word2vec_model.wv[word]
        return np.zeros(self.embedding_dim)
    
    def get_similar_words(self, word, topn=10):
        """Get most similar words"""
        if self.word2vec_model and word in self.word2vec_model.wv:
            return self.word2vec_model.wv.most_similar(word, topn=topn)
        return []

