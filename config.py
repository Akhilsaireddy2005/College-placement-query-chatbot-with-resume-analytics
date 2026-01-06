"""
Configuration settings for the College Placement Chatbot system
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'training')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model paths
INTENT_MODEL_PATH = os.path.join(MODELS_DIR, 'intent_classifier.h5')
RESUME_MODEL_PATH = os.path.join(MODELS_DIR, 'resume_classifier.h5')
WORD2VEC_MODEL_PATH = os.path.join(MODELS_DIR, 'word2vec.model')

# Model parameters
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
NUM_FILTERS = 128
FILTER_SIZES = [3, 4, 5]
DROPOUT_RATE = 0.5
NUM_EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Intent classes
INTENT_CLASSES = [
    'eligibility',
    'company_info',
    'skill_requirements',
    'preparation',
    'statistics',
    'greeting',
    'goodbye'
]

# Job domains
JOB_DOMAINS = [
    'Web Development',
    'Data Science',
    'AI/ML',
    'Core Engineering'
]

# Technical skills keywords
TECHNICAL_SKILLS = [
    'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'html', 'css',
    'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
    'machine learning', 'deep learning', 'neural networks', 'nlp',
    'computer vision', 'data analysis', 'data visualization',
    'mongodb', 'mysql', 'postgresql', 'redis', 'aws', 'docker', 'kubernetes',
    'git', 'github', 'agile', 'scrum', 'rest api', 'graphql', 'microservices'
]

# Soft skills keywords
SOFT_SKILLS = [
    'communication', 'leadership', 'teamwork', 'problem solving',
    'time management', 'adaptability', 'creativity', 'critical thinking',
    'collaboration', 'presentation', 'negotiation', 'project management'
]

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
UPLOAD_FOLDER = UPLOADS_DIR
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

