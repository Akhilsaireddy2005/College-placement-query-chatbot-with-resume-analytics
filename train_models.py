"""
Model Training Script
Trains CNN models for intent classification and resume domain classification
"""

import os
import json
import numpy as np
from models.intent_classifier import IntentClassifier
from models.resume_classifier import ResumeClassifier
from config import DATA_DIR, TRAINING_DATA_DIR


def load_training_data():
    """Load or generate sample training data"""
    
    # Sample intent classification training data
    intent_data = {
        'texts': [
            # Greeting
            "hello", "hi", "hey", "good morning", "good afternoon",
            # Eligibility
            "what are the eligibility criteria", "eligibility requirements", "minimum cgpa required",
            "what cgpa is needed", "eligibility for google", "amazon eligibility",
            "microsoft requirements", "what are the requirements",
            # Company info
            "tell me about google", "information about microsoft", "amazon details",
            "company information", "what about the company", "company profile",
            # Skills
            "what skills are required", "required skills", "what skills do i need",
            "skills for data science", "web development skills", "ai ml skills",
            "technical skills needed", "what are the skill requirements",
            # Preparation
            "how to prepare", "preparation tips", "interview preparation",
            "how should i prepare", "coding interview tips", "resume tips",
            "behavioral interview", "preparation guide",
            # Statistics
            "placement statistics", "placement data", "average package",
            "placement rate", "statistics", "placement numbers",
            # Goodbye
            "bye", "goodbye", "see you", "thanks", "thank you"
        ],
        'labels': [
            # Greeting
            'greeting', 'greeting', 'greeting', 'greeting', 'greeting',
            # Eligibility
            'eligibility', 'eligibility', 'eligibility', 'eligibility', 'eligibility',
            'eligibility', 'eligibility', 'eligibility',
            # Company info
            'company_info', 'company_info', 'company_info', 'company_info',
            'company_info', 'company_info',
            # Skills
            'skill_requirements', 'skill_requirements', 'skill_requirements',
            'skill_requirements', 'skill_requirements', 'skill_requirements',
            'skill_requirements', 'skill_requirements',
            # Preparation
            'preparation', 'preparation', 'preparation', 'preparation',
            'preparation', 'preparation', 'preparation', 'preparation',
            # Statistics
            'statistics', 'statistics', 'statistics', 'statistics',
            'statistics', 'statistics',
            # Goodbye
            'goodbye', 'goodbye', 'goodbye', 'goodbye', 'goodbye'
        ]
    }
    
    # Expand dataset with variations
    expanded_texts = []
    expanded_labels = []
    
    variations = {
        'eligibility': [
            "what is the eligibility", "eligibility criteria for placements",
            "minimum requirements", "what are the minimum qualifications",
            "cgpa requirement", "backlog criteria"
        ],
        'company_info': [
            "company details", "about the company", "company overview",
            "tell me more about", "company culture"
        ],
        'skill_requirements': [
            "what technical skills", "programming skills needed",
            "required technical knowledge", "skills for placement"
        ],
        'preparation': [
            "how can i prepare", "preparation strategy", "interview tips",
            "how to crack interviews", "placement preparation"
        ],
        'statistics': [
            "placement info", "placement details", "how many got placed",
            "placement percentage", "salary statistics"
        ]
    }
    
    for label, texts in variations.items():
        expanded_texts.extend(texts)
        expanded_labels.extend([label] * len(texts))
    
    intent_data['texts'].extend(expanded_texts)
    intent_data['labels'].extend(expanded_labels)
    
    # Sample resume domain classification training data
    resume_data = {
        'texts': [
            # Web Development
            "python javascript react node.js html css web development frontend backend full stack",
            "html css javascript react angular vue frontend developer web applications",
            "node.js express mongodb rest api web development backend developer",
            "react redux javascript typescript web development frontend engineer",
            # Data Science
            "python pandas numpy matplotlib seaborn data analysis sql machine learning statistics",
            "data science python sql tableau power bi data visualization analytics",
            "pandas numpy scikit-learn data analysis python sql database",
            "data analyst python sql excel statistics data visualization",
            # AI/ML
            "python tensorflow pytorch keras machine learning deep learning neural networks",
            "machine learning deep learning nlp computer vision ai ml engineer",
            "tensorflow pytorch keras neural networks deep learning artificial intelligence",
            "nlp natural language processing machine learning python tensorflow",
            # Core Engineering
            "engineering problem solving technical knowledge project management",
            "mechanical engineering electrical engineering core engineering projects",
            "engineering fundamentals technical skills problem solving",
            "core engineering domain knowledge technical expertise"
        ],
        'labels': [
            'Web Development', 'Web Development', 'Web Development', 'Web Development',
            'Data Science', 'Data Science', 'Data Science', 'Data Science',
            'AI/ML', 'AI/ML', 'AI/ML', 'AI/ML',
            'Core Engineering', 'Core Engineering', 'Core Engineering', 'Core Engineering'
        ]
    }
    
    # Expand resume dataset
    resume_expansions = {
        'Web Development': [
            "javascript react node.js express mongodb web developer",
            "html css bootstrap javascript jquery web development",
            "react native mobile web development javascript",
            "django flask python web development backend"
        ],
        'Data Science': [
            "python data science pandas numpy jupyter notebook",
            "sql database data analysis python r statistics",
            "machine learning data science python scikit-learn",
            "data engineer etl python sql hadoop spark"
        ],
        'AI/ML': [
            "deep learning tensorflow keras neural networks",
            "computer vision opencv tensorflow python",
            "reinforcement learning pytorch machine learning",
            "nlp transformers bert gpt natural language processing"
        ],
        'Core Engineering': [
            "engineering design problem solving technical",
            "core engineering fundamentals technical knowledge",
            "engineering projects technical skills"
        ]
    }
    
    for domain, texts in resume_expansions.items():
        resume_data['texts'].extend(texts)
        resume_data['labels'].extend([domain] * len(texts))
    
    return intent_data, resume_data


def train_intent_classifier():
    """Train intent classification model"""
    print("\n" + "="*50)
    print("Training Intent Classifier")
    print("="*50)
    
    intent_classifier = IntentClassifier()
    intent_data, _ = load_training_data()
    
    print(f"Training samples: {len(intent_data['texts'])}")
    print(f"Intent classes: {set(intent_data['labels'])}")
    
    history = intent_classifier.train(
        intent_data['texts'],
        intent_data['labels']
    )
    
    print("\nIntent Classifier Training Complete!")
    return history


def train_resume_classifier():
    """Train resume domain classification model"""
    print("\n" + "="*50)
    print("Training Resume Domain Classifier")
    print("="*50)
    
    resume_classifier = ResumeClassifier()
    _, resume_data = load_training_data()
    
    print(f"Training samples: {len(resume_data['texts'])}")
    print(f"Domain classes: {set(resume_data['labels'])}")
    
    history = resume_classifier.train(
        resume_data['texts'],
        resume_data['labels']
    )
    
    print("\nResume Classifier Training Complete!")
    return history


def main():
    """Main training function"""
    print("="*50)
    print("College Placement Chatbot - Model Training")
    print("="*50)
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/training', exist_ok=True)
    
    # Train models
    try:
        train_intent_classifier()
        train_resume_classifier()
        
        print("\n" + "="*50)
        print("All Models Trained Successfully!")
        print("="*50)
        print("\nYou can now run the Flask application using: python app.py")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
