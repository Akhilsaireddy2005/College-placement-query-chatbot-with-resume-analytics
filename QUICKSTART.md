# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for downloading NLTK data and spaCy models)

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Required Data

Run the setup script to download NLTK data and spaCy models:

```bash
python setup.py
```

**Manual Alternative** (if setup.py doesn't work):

```python
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Train the Models

**Important**: You must train the models before running the application!

```bash
python train_models.py
```

This will:
- Train Word2Vec embeddings
- Train Intent Classification CNN model
- Train Resume Domain Classification CNN model
- Save all models to the `models/` directory

**Note**: Training may take 5-15 minutes depending on your hardware.

### 4. Run the Application

```bash
python app.py
```

You should see:
```
==================================================
College Placement Chatbot - Starting Flask App
==================================================

Access the application at: http://localhost:5000

Press Ctrl+C to stop the server
==================================================
```

### 5. Access the Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

## Using the Application

### Chat Interface

1. Type your query in the chat input box
2. Examples:
   - "What are the eligibility criteria for Google?"
   - "What skills are required for data science?"
   - "How should I prepare for interviews?"
   - "Show me placement statistics"

### Resume Analysis

**Option 1: Upload File**
1. Click on "Upload File" tab
2. Click "Choose Resume File"
3. Select a PDF, DOC, DOCX, or TXT file
4. Click "Analyze Resume"

**Option 2: Paste Text**
1. Click on "Paste Text" tab
2. Paste your resume text
3. Click "Analyze Resume"

The system will analyze:
- Technical and soft skills
- Job domain classification
- Skill gaps
- Recommended job roles
- Readiness score
- Visualizations

## Troubleshooting

### Issue: "Model not found" error

**Solution**: Train the models first:
```bash
python train_models.py
```

### Issue: Import errors

**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: NLTK data not found

**Solution**: Run setup script:
```bash
python setup.py
```

### Issue: spaCy model not found

**Solution**: Download manually:
```bash
python -m spacy download en_core_web_sm
```

### Issue: Training takes too long

**Solution**: 
- Reduce `NUM_EPOCHS` in `config.py`
- Use GPU if available
- Reduce training data size

### Issue: Low accuracy

**Solution**:
- Add more training data
- Increase `NUM_EPOCHS` in `config.py`
- Fine-tune hyperparameters

## Project Structure Overview

- `app.py` - Main Flask application
- `train_models.py` - Model training script
- `config.py` - Configuration settings
- `models/` - ML models (trained models saved here)
- `nlp/` - NLP preprocessing modules
- `chatbot/` - Chatbot logic
- `analytics/` - Resume analytics and visualizations
- `templates/` - HTML templates
- `static/` - CSS, JS, and generated images

## Next Steps

1. **Customize Responses**: Edit `chatbot/responses.py` to add more response templates
2. **Add Training Data**: Expand training data in `train_models.py` for better accuracy
3. **Update Skills**: Modify skill keywords in `config.py`
4. **Enhance UI**: Customize `templates/index.html` and `static/css/style.css`

## Support

For detailed architecture information, see `ARCHITECTURE.md`
For project documentation, see `README.md`

