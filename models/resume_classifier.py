"""
CNN-based Resume Domain Classifier
Classifies resumes into job domains
"""

import os
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, concatenate
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
except ImportError:
    import keras
    from keras.models import Sequential, load_model
    from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, concatenate
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from config import (
    RESUME_MODEL_PATH, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,
    NUM_FILTERS, FILTER_SIZES, DROPOUT_RATE, NUM_EPOCHS, BATCH_SIZE,
    VALIDATION_SPLIT, JOB_DOMAINS, WORD2VEC_MODEL_PATH
)
from models.word_embeddings import WordEmbeddings


class ResumeClassifier:
    """CNN-based resume domain classification model"""
    
    def __init__(self):
        self.model = None
        self.word_embeddings = WordEmbeddings()
        self.num_classes = len(JOB_DOMAINS)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(JOB_DOMAINS)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(JOB_DOMAINS)}
    
    def build_model(self, vocab_size, embedding_matrix):
        """
        Build CNN model for resume classification
        
        Args:
            vocab_size: Size of vocabulary
            embedding_matrix: Pre-trained embedding matrix
        """
        # Input layer
        input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
        
        # Embedding layer
        embedding = Embedding(
            vocab_size,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False
        )(input_layer)
        
        # Multiple convolution layers with different filter sizes
        conv_layers = []
        for filter_size in FILTER_SIZES:
            conv = Conv1D(
                NUM_FILTERS,
                filter_size,
                activation='relu',
                padding='same'
            )(embedding)
            pooled = GlobalMaxPooling1D()(conv)
            conv_layers.append(pooled)
        
        # Concatenate all convolution outputs
        if len(conv_layers) > 1:
            merged = concatenate(conv_layers)
        else:
            merged = conv_layers[0]
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(merged)
        dropout1 = Dropout(DROPOUT_RATE)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(DROPOUT_RATE)(dense2)
        output = Dense(self.num_classes, activation='softmax')(dropout2)
        
        # Build model
        try:
            model = tf.keras.Model(inputs=input_layer, outputs=output)
        except:
            from keras.models import Model
            model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, texts, labels):
        """
        Prepare training data
        
        Args:
            texts: List of resume text strings
            labels: List of domain labels
        """
        # Convert texts to sequences
        sequences = []
        for text in texts:
            seq = self.word_embeddings.text_to_sequence(text, MAX_SEQUENCE_LENGTH)
            sequences.append(seq)
        
        X = np.array(sequences)
        
        # Convert labels to categorical
        y = [self.class_to_idx[label] for label in labels]
        y = to_categorical(y, num_classes=self.num_classes)
        
        return X, y
    
    def train(self, texts, labels, validation_split=VALIDATION_SPLIT):
        """
        Train the resume classifier
        
        Args:
            texts: List of training resume texts
            labels: List of training domain labels
            validation_split: Fraction of data to use for validation
        """
        # Load or train word embeddings
        if os.path.exists(WORD2VEC_MODEL_PATH):
            try:
                self.word_embeddings.load_word2vec()
            except:
                print("Training Word2Vec model...")
                self.word_embeddings.train_word2vec(texts)
        else:
            print("Training Word2Vec model...")
            self.word_embeddings.train_word2vec(texts)
        
        # Prepare data
        X, y = self.prepare_data(texts, labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Get embedding matrix
        embedding_matrix = self.word_embeddings.get_embedding_matrix()
        vocab_size = self.word_embeddings.vocab_size
        
        # Build model
        self.build_model(vocab_size, embedding_matrix)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Save model
        os.makedirs(os.path.dirname(RESUME_MODEL_PATH), exist_ok=True)
        self.model.save(RESUME_MODEL_PATH)
        print(f"Resume classifier saved to {RESUME_MODEL_PATH}")
        
        return history
    
    def load_model(self, model_path=None):
        """Load pre-trained model"""
        if model_path is None:
            model_path = RESUME_MODEL_PATH
        
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            # Load word embeddings
            self.word_embeddings.load_word2vec()
            print(f"Resume classifier loaded from {model_path}")
        else:
            print(f"Model not found at {model_path}")
    
    def predict(self, text):
        """
        Predict domain for a given resume text
        
        Args:
            text: Input resume text string
        
        Returns:
            Predicted domain class and confidence score
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Convert text to sequence
        sequence = self.word_embeddings.text_to_sequence(text, MAX_SEQUENCE_LENGTH)
        X = np.array([sequence])
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = self.idx_to_class[predicted_idx]
        
        return predicted_class, float(confidence)

