# College Placement Query Chatbot with Resume Text Analytics

An end-to-end AI system that integrates Natural Language Processing (NLP), Deep Learning, CNN-based chatbot, and Text Analytics for college placement assistance.

## Features

- **Intelligent Chatbot**: CNN-based conversational chatbot for placement-related queries
- **Resume Analytics**: Automated text analysis of student resumes (PDF/DOC/text)
- **Intent Classification**: Deep learning model for understanding student queries
- **Domain Classification**: Automatic classification of resumes into job domains
- **Skill Extraction**: Extract technical and soft skills from resumes
- **Gap Analysis**: Identify skill gaps and recommend suitable job roles
- **Visualizations**: Interactive charts and graphs for analytics

## Project Structure

```
project/
├── app.py                      # Flask web application
├── config.py                   # Configuration settings
├── models/
│   ├── intent_classifier.py   # CNN model for intent classification
│   ├── resume_classifier.py   # CNN model for resume domain classification
│   └── word_embeddings.py     # Word embedding utilities
├── nlp/
│   ├── preprocessing.py       # NLP preprocessing functions
│   └── resume_parser.py       # Resume text extraction
├── chatbot/
│   ├── chatbot.py             # Main chatbot logic
│   └── responses.py           # Rule-based response generation
├── analytics/
│   ├── resume_analytics.py    # Resume analysis engine
│   └── visualizations.py      # Data visualization functions
├── data/
│   ├── training/              # Training datasets
│   └── uploads/               # Uploaded resumes
├── static/
│   ├── css/                   # CSS styles
│   └── js/                    # JavaScript files
├── templates/
│   ├── index.html             # Main web interface
│   └── analytics.html         # Analytics dashboard
└── utils/
    └── helpers.py             # Utility functions
```

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Train the models (first time only):
```bash
python train_models.py
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Usage Examples

### Chatbot Queries
- "What are the eligibility criteria for Google?"
- "What skills are required for data science roles?"
- "How should I prepare for placement interviews?"
- "Show me placement statistics"

### Resume Analysis
- Upload a resume (PDF/DOC/text format)
- Get automated analysis including:
  - Extracted skills
  - Domain classification
  - Skill gaps
  - Job role recommendations

## Technologies Used

- **NLP**: NLTK, spaCy
- **Deep Learning**: TensorFlow, PyTorch
- **ML**: scikit-learn
- **Web Framework**: Flask
- **Visualization**: matplotlib, seaborn
- **Text Processing**: gensim, PyPDF2, python-docx

## Model Architecture

### Intent Classification CNN
- Input: Word embeddings (Word2Vec/GloVe)
- Architecture: Convolutional layers + Dense layers
- Output: Intent classes (eligibility, skills, preparation, statistics)

### Resume Domain Classification CNN
- Input: Resume text embeddings
- Architecture: Convolutional layers + Dense layers
- Output: Domain classes (Web Development, Data Science, AI/ML, Core Engineering)

## License

This project is for educational purposes.

