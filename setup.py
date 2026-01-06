"""
Setup script for downloading required NLTK data and spaCy models
"""

import nltk
import subprocess
import sys

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

def download_spacy_model():
    """Download spaCy English model"""
    print("\nDownloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy model downloaded successfully")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        print("You can manually download it using: python -m spacy download en_core_web_sm")

def main():
    print("="*50)
    print("Setting up College Placement Chatbot")
    print("="*50)
    
    download_nltk_data()
    download_spacy_model()
    
    print("\n" + "="*50)
    print("Setup complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Train the models: python train_models.py")
    print("2. Run the application: python app.py")

if __name__ == "__main__":
    main()

