# System Architecture Documentation

## Overview

The College Placement Query Chatbot with Resume Text Analytics is an end-to-end AI system that integrates Natural Language Processing (NLP), Deep Learning (CNN models), and Text Analytics to assist students with placement-related queries and resume analysis.

## System Architecture

### 1. Core Components

#### 1.1 NLP Preprocessing Module (`nlp/preprocessing.py`)
- **Purpose**: Text cleaning and preprocessing
- **Features**:
  - Tokenization (word and sentence)
  - Stop-word removal
  - Lemmatization
  - POS (Part-of-Speech) tagging
  - Text cleaning and normalization

#### 1.2 Word Embeddings Module (`models/word_embeddings.py`)
- **Purpose**: Convert text to numerical representations
- **Features**:
  - Word2Vec model training and loading
  - Text-to-sequence conversion
  - Embedding matrix generation for neural networks
  - Vocabulary management

#### 1.3 Intent Classifier (`models/intent_classifier.py`)
- **Purpose**: Classify user queries into intent categories
- **Architecture**: CNN (Convolutional Neural Network)
- **Model Details**:
  - Input: Word embeddings (sequence of word indices)
  - Architecture: Embedding → Multiple Conv1D layers → GlobalMaxPooling → Dense layers → Softmax
  - Output: Intent classes (eligibility, company_info, skill_requirements, preparation, statistics, greeting, goodbye)
- **Training**: Uses labeled query data with validation split

#### 1.4 Resume Domain Classifier (`models/resume_classifier.py`)
- **Purpose**: Classify resumes into job domains
- **Architecture**: CNN (Convolutional Neural Network)
- **Model Details**:
  - Input: Resume text embeddings
  - Architecture: Similar to intent classifier with additional dense layer
  - Output: Job domains (Web Development, Data Science, AI/ML, Core Engineering)
- **Training**: Uses labeled resume text data

### 2. Chatbot Module

#### 2.1 Main Chatbot (`chatbot/chatbot.py`)
- **Purpose**: Process user queries and generate responses
- **Workflow**:
  1. Receive user query
  2. Preprocess text
  3. Classify intent using CNN model
  4. Generate response using rule-based system
  5. Return response with metadata

#### 2.2 Response Generator (`chatbot/responses.py`)
- **Purpose**: Generate contextual responses based on intent
- **Features**:
  - Rule-based response templates
  - Context-aware responses (company-specific, domain-specific)
  - Multiple response variations
  - Intent-specific response logic

### 3. Resume Analytics Module

#### 3.1 Resume Parser (`nlp/resume_parser.py`)
- **Purpose**: Extract text from various file formats
- **Supported Formats**: PDF, DOC, DOCX, TXT
- **Libraries**: PyPDF2, python-docx

#### 3.2 Resume Analytics Engine (`analytics/resume_analytics.py`)
- **Purpose**: Comprehensive resume analysis
- **Features**:
  - **Skill Extraction**: Extract technical and soft skills using keyword matching
  - **Domain Classification**: Classify resume into job domain using CNN
  - **Skill Gap Analysis**: Identify missing skills for target domain
  - **Job Role Recommendations**: Suggest suitable roles based on resume content
  - **Readiness Scoring**: Calculate resume readiness score (0-100)

### 4. Visualization Module (`analytics/visualizations.py`)

- **Purpose**: Generate visualizations for analytics
- **Visualizations**:
  - Skill frequency bar charts
  - Domain distribution pie charts
  - Readiness score histograms
  - Skill gap analysis charts
  - Word clouds
  - Comparison charts

### 5. Web Application (`app.py`)

- **Framework**: Flask
- **Features**:
  - Chat interface for placement queries
  - Resume upload (file or text)
  - Real-time analysis results
  - Visualization display
  - RESTful API endpoints

#### API Endpoints:
- `GET /`: Main web interface
- `POST /chat`: Process chat queries
- `POST /upload_resume`: Upload and analyze resume file
- `POST /analyze_text`: Analyze plain text resume
- `GET /static/images/<filename>`: Serve generated visualizations

## Data Flow

### Chat Query Flow:
```
User Query → Preprocessing → Intent Classification (CNN) → Response Generation → User
```

### Resume Analysis Flow:
```
Resume File/Text → Resume Parser → Text Extraction → 
  ├→ Skill Extraction (Keyword Matching)
  ├→ Domain Classification (CNN)
  ├→ Gap Analysis
  ├→ Role Recommendations
  └→ Readiness Scoring
→ Visualization Generation → Results Display
```

## Model Training

### Training Process (`train_models.py`):

1. **Data Preparation**:
   - Generate/load training data
   - Preprocess texts
   - Convert to sequences

2. **Word Embeddings**:
   - Train Word2Vec on all training texts
   - Build vocabulary index
   - Create embedding matrix

3. **Model Training**:
   - Split data (train/validation)
   - Build CNN architecture
   - Train with early stopping
   - Save trained models

### Training Data:
- **Intent Classification**: ~100+ labeled query examples
- **Resume Classification**: ~30+ labeled resume text examples
- Both datasets include variations and expansions

## Configuration

All configuration is centralized in `config.py`:
- Model parameters (embedding dimensions, filter sizes, etc.)
- File paths
- Intent classes and job domains
- Skill keywords
- Flask settings

## File Structure

```
project/
├── app.py                      # Flask application
├── config.py                   # Configuration
├── train_models.py             # Model training script
├── setup.py                    # Setup script
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── ARCHITECTURE.md             # This file
├── models/                     # ML models
│   ├── intent_classifier.py
│   ├── resume_classifier.py
│   └── word_embeddings.py
├── nlp/                        # NLP modules
│   ├── preprocessing.py
│   └── resume_parser.py
├── chatbot/                    # Chatbot modules
│   ├── chatbot.py
│   └── responses.py
├── analytics/                  # Analytics modules
│   ├── resume_analytics.py
│   └── visualizations.py
├── utils/                      # Utilities
│   └── helpers.py
├── templates/                  # HTML templates
│   └── index.html
├── static/                     # Static files
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── images/                 # Generated visualizations
└── data/                       # Data directories
    ├── training/
    └── uploads/
```

## Technologies Used

- **NLP**: NLTK, spaCy
- **Deep Learning**: TensorFlow/Keras
- **ML**: scikit-learn
- **Web Framework**: Flask
- **Visualization**: matplotlib, seaborn, wordcloud
- **Text Processing**: gensim (Word2Vec), PyPDF2, python-docx

## Model Performance

### Intent Classifier:
- Architecture: Multi-filter CNN (3, 4, 5 filter sizes)
- Embedding Dimension: 100
- Filters per size: 128
- Dense layers: 128 → num_classes
- Expected accuracy: 80-90% (depends on training data)

### Resume Classifier:
- Architecture: Similar to intent classifier
- Additional dense layer: 64 units
- Expected accuracy: 75-85% (depends on training data)

## Future Enhancements

1. **Enhanced NLP**:
   - Named Entity Recognition (NER) for better skill extraction
   - Advanced resume parsing with structured extraction

2. **Model Improvements**:
   - Fine-tuning with more training data
   - Transfer learning (BERT, GPT embeddings)
   - Attention mechanisms

3. **Features**:
   - Multi-language support
   - Resume comparison
   - Interview question generation
   - Company-specific insights

4. **UI/UX**:
   - Real-time chat with typing indicators
   - Advanced filtering and search
   - Export analytics reports

## Usage Instructions

1. **Installation**:
   ```bash
   pip install -r requirements.txt
   python setup.py
   ```

2. **Training Models**:
   ```bash
   python train_models.py
   ```

3. **Running Application**:
   ```bash
   python app.py
   ```

4. **Access**: Open browser to `http://localhost:5000`

## Notes

- Models need to be trained before first use
- Training may take several minutes depending on hardware
- For production, increase training data and epochs
- Consider using GPU for faster training
- Update skill keywords in `config.py` as needed

