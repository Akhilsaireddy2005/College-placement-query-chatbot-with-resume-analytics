"""
Resume Text Analytics Module
Extracts skills, classifies domains, identifies gaps, and recommends roles
"""

import re
from collections import Counter
from models.resume_classifier import ResumeClassifier
from nlp.preprocessing import NLPPreprocessor
from config import TECHNICAL_SKILLS, SOFT_SKILLS, JOB_DOMAINS


class ResumeAnalytics:
    """Resume analytics engine for skill extraction and analysis"""
    
    def __init__(self):
        self.resume_classifier = ResumeClassifier()
        self.preprocessor = NLPPreprocessor()
        self.initialized = False
    
    def initialize(self):
        """Initialize the analytics engine by loading models"""
        try:
            self.resume_classifier.load_model()
            self.initialized = True
            print("Resume analytics initialized successfully")
        except Exception as e:
            print(f"Error initializing resume analytics: {e}")
            print("Please train the models first using train_models.py")
            self.initialized = False
    
    def extract_technical_skills(self, resume_text):
        """
        Extract technical skills from resume text
        """
        text_lower = resume_text.lower()
        found_skills = []
        
        for skill in TECHNICAL_SKILLS:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        # Return unique, sorted for consistency
        return sorted(list(set(found_skills)))
    
    def extract_soft_skills(self, resume_text):
        """
        Extract soft skills from resume text
        """
        text_lower = resume_text.lower()
        found_skills = []
        
        for skill in SOFT_SKILLS:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill.title())
        
        return sorted(list(set(found_skills)))
    
    def classify_domain(self, resume_text):
        """
        Classify resume into job domain with confidence fallback
        """
        # If model not initialized, use keyword scoring
        if not self.initialized:
            return self._keyword_based_classification(resume_text)
        
        try:
            domain, confidence = self.resume_classifier.predict(resume_text)
            # If model confidence is low, blend with keyword fallback
            if confidence < 0.45:
                keyword_result = self._keyword_based_classification(resume_text)
                # Pick the higher-confidence result
                if keyword_result['confidence'] > confidence:
                    return keyword_result
            return {
                'domain': domain,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error in domain classification: {e}")
            return self._keyword_based_classification(resume_text)
    
    def _keyword_based_classification(self, resume_text):
        """Fallback keyword-based domain classification with weighted scores"""
        text_lower = resume_text.lower()
        domain_scores = {}
        
        keyword_map = {
            'Web Development': {
                'keywords': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'node', 'web', 'frontend', 'backend', 'django', 'flask'],
                'weight': 1.0
            },
            'Data Science': {
                'keywords': ['data science', 'data analysis', 'pandas', 'numpy', 'statistics', 'sql', 'tableau', 'power bi', 'matplotlib', 'seaborn'],
                'weight': 1.05
            },
            'AI/ML': {
                'keywords': ['machine learning', 'deep learning', 'neural network', 'tensorflow', 'pytorch', 'nlp', 'ai', 'ml', 'transformer', 'bert', 'gpt'],
                'weight': 1.1
            },
            'Core Engineering': {
                'keywords': ['engineering', 'mechanical', 'electrical', 'civil', 'core', 'embedded', 'vlsi'],
                'weight': 0.95
            }
        }
        
        for domain, cfg in keyword_map.items():
            hits = sum(1 for keyword in cfg['keywords'] if keyword in text_lower)
            domain_scores[domain] = hits * cfg['weight']
        
        top_domain = max(domain_scores, key=domain_scores.get)
        top_score = domain_scores[top_domain]
        confidence = min(top_score / 6.0, 1.0) if top_score > 0 else 0.25
        
        return {
            'domain': top_domain,
            'confidence': confidence
        }
    
    def identify_skill_gaps(self, resume_text, target_domain):
        """
        Identify skill gaps for a target domain
        
        Args:
            resume_text: Resume text string
            target_domain: Target job domain
        
        Returns:
            Dictionary with required skills, found skills, and gaps
        """
        found_skills = self.extract_technical_skills(resume_text)
        found_skills_lower = [s.lower() for s in found_skills]
        
        domain_requirements = {
            'Web Development': ['html', 'css', 'javascript', 'react', 'node.js', 'sql', 'git'],
            'Data Science': ['python', 'sql', 'pandas', 'numpy', 'statistics', 'machine learning', 'data analysis'],
            'AI/ML': ['python', 'tensorflow', 'pytorch', 'machine learning', 'deep learning', 'nlp'],
            'Core Engineering': ['problem solving', 'technical knowledge', 'project experience']
        }
        
        required_skills = domain_requirements.get(target_domain, [])
        required_skills_lower = [s.lower() for s in required_skills]
        
        gaps = [skill for skill in required_skills_lower if skill not in found_skills_lower]
        coverage = (len(required_skills_lower) - len(gaps)) / len(required_skills_lower) if required_skills_lower else 0
        
        return {
            'required_skills': required_skills,
            'found_skills': found_skills,
            'gaps': gaps,
            'coverage': coverage
        }
    
    def recommend_job_roles(self, resume_text):
        """
        Recommend suitable job roles based on resume content
        
        Args:
            resume_text: Resume text string
        
        Returns:
            List of recommended roles with scores
        """
        domain_result = self.classify_domain(resume_text)
        primary_domain = domain_result['domain']
        
        # Role recommendations by domain
        role_mapping = {
            'Web Development': [
                'Frontend Developer',
                'Backend Developer',
                'Full Stack Developer',
                'React Developer',
                'Node.js Developer'
            ],
            'Data Science': [
                'Data Analyst',
                'Data Scientist',
                'Business Analyst',
                'Data Engineer',
                'ML Engineer'
            ],
            'AI/ML': [
                'Machine Learning Engineer',
                'Deep Learning Engineer',
                'NLP Engineer',
                'Computer Vision Engineer',
                'AI Researcher'
            ],
            'Core Engineering': [
                'Software Engineer',
                'Systems Engineer',
                'Product Engineer',
                'Research Engineer'
            ]
        }
        
        recommended_roles = role_mapping.get(primary_domain, ['Software Engineer'])
        
        return {
            'primary_domain': primary_domain,
            'recommended_roles': recommended_roles,
            'domain_confidence': domain_result['confidence']
        }
    
    def calculate_readiness_score(self, resume_text, target_domain=None):
        """
        Calculate resume readiness score
        
        Args:
            resume_text: Resume text string
            target_domain: Optional target domain for scoring
        
        Returns:
            Readiness score (0-100)
        """
        # Extract skills
        technical_skills = self.extract_technical_skills(resume_text)
        soft_skills = self.extract_soft_skills(resume_text)
        
        # Base score from skills
        skill_score = min(len(technical_skills) * 5, 40) + min(len(soft_skills) * 3, 20)
        
        # Domain classification confidence
        domain_result = self.classify_domain(resume_text)
        domain_score = domain_result['confidence'] * 20
        
        # Text quality (length and structure)
        text_length = len(resume_text.split())
        length_score = min(text_length / 10, 20)  # Max 20 points
        
        total_score = skill_score + domain_score + length_score
        
        # If target domain specified, check gaps
        if target_domain:
            gaps_result = self.identify_skill_gaps(resume_text, target_domain)
            gap_penalty = len(gaps_result['gaps']) * 5
            total_score = max(0, total_score - gap_penalty)
        
        return min(100, max(0, total_score))
    
    def analyze_resume(self, resume_text):
        """
        Complete resume analysis
        
        Args:
            resume_text: Resume text string
        
        Returns:
            Complete analysis dictionary
        """
        # Extract skills
        technical_skills = self.extract_technical_skills(resume_text)
        soft_skills = self.extract_soft_skills(resume_text)
        
        # Classify domain with fallback
        domain_result = self.classify_domain(resume_text)
        
        # Recommend roles
        role_recommendations = self.recommend_job_roles(resume_text)
        
        # Identify gaps
        skill_gaps = self.identify_skill_gaps(resume_text, domain_result['domain'])
        
        # Calculate readiness
        readiness_score = self.calculate_readiness_score(resume_text, domain_result['domain'])
        
        return {
            'technical_skills': technical_skills,
            'soft_skills': soft_skills,
            'domain': domain_result['domain'],
            'domain_confidence': domain_result['confidence'],
            'recommended_roles': role_recommendations['recommended_roles'],
            'readiness_score': readiness_score,
            'skill_gaps': skill_gaps['gaps'],
            'skill_coverage': skill_gaps['coverage'],
            'total_skills_count': len(technical_skills) + len(soft_skills)
        }
