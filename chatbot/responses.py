"""
Rule-based Response Generation
Generates appropriate responses based on intent classification
"""

from config import INTENT_CLASSES, JOB_DOMAINS, TECHNICAL_SKILLS


class ResponseGenerator:
    """Generate rule-based responses for chatbot"""
    
    def __init__(self):
        self.responses = self._initialize_responses()
    
    def _initialize_responses(self):
        """Initialize response templates"""
        return {
            'greeting': [
                "Hello! I'm your placement assistant. How can I help you today?",
                "Hi! I'm here to help with placement queries. What would you like to know?",
                "Welcome! Ask me anything about placements, companies, or skills."
            ],
            'goodbye': [
                "Goodbye! Good luck with your placements!",
                "See you later! All the best!",
                "Take care! Hope I was helpful."
            ],
            'eligibility': {
                'default': "Eligibility criteria typically include:\n"
                          "- Minimum CGPA (usually 7.0+)\n"
                          "- No active backlogs\n"
                          "- Good communication skills\n"
                          "- Relevant technical skills\n\n"
                          "Specific criteria vary by company. Would you like details for a specific company?",
                'google': "Google typically requires:\n"
                         "- CGPA: 8.0+\n"
                         "- Strong problem-solving skills\n"
                         "- Proficiency in data structures and algorithms\n"
                         "- Good communication skills\n"
                         "- Relevant projects and internships",
                'microsoft': "Microsoft typically requires:\n"
                           "- CGPA: 7.5+\n"
                           "- Strong coding skills\n"
                           "- Knowledge of cloud technologies\n"
                           "- Problem-solving abilities\n"
                           "- Team collaboration skills",
                'amazon': "Amazon typically requires:\n"
                         "- CGPA: 7.0+\n"
                         "- Strong algorithmic thinking\n"
                         "- System design knowledge\n"
                         "- Leadership principles\n"
                         "- Technical expertise"
            },
            'company_info': {
                'default': "I can provide information about various companies including:\n"
                          "- Eligibility criteria\n"
                          "- Interview process\n"
                          "- Required skills\n"
                          "- Package details\n\n"
                          "Which company would you like to know about?",
                'google': "Google:\n"
                         "- Focus: Search, Cloud, AI/ML\n"
                         "- Process: Online test, Technical interviews, HR round\n"
                         "- Package: 15-25 LPA (varies by role)\n"
                         "- Culture: Innovation-driven, collaborative",
                'microsoft': "Microsoft:\n"
                            "- Focus: Cloud, Software, AI\n"
                            "- Process: Online test, Technical rounds, Final interview\n"
                            "- Package: 12-20 LPA (varies by role)\n"
                            "- Culture: Growth mindset, inclusive",
                'amazon': "Amazon:\n"
                         "- Focus: E-commerce, Cloud, AI\n"
                         "- Process: Online assessment, Technical interviews, Bar raiser\n"
                         "- Package: 10-18 LPA (varies by role)\n"
                         "- Culture: Customer-obsessed, ownership"
            },
            'skill_requirements': {
                'default': "Common skills required for placements:\n\n"
                          "**Technical Skills:**\n"
                          "- Programming languages (Python, Java, C++)\n"
                          "- Data structures and algorithms\n"
                          "- Database management\n"
                          "- Web development frameworks\n"
                          "- Machine learning (for AI/ML roles)\n\n"
                          "**Soft Skills:**\n"
                          "- Communication\n"
                          "- Problem-solving\n"
                          "- Teamwork\n"
                          "- Leadership\n\n"
                          "Which domain are you interested in?",
                'data_science': "Data Science roles require:\n"
                               "- Python/R programming\n"
                               "- Statistics and probability\n"
                               "- Machine learning algorithms\n"
                               "- Data visualization\n"
                               "- SQL and databases\n"
                               "- Big data tools (Hadoop, Spark)",
                'web_development': "Web Development roles require:\n"
                                 "- HTML, CSS, JavaScript\n"
                                 "- Frontend frameworks (React, Angular, Vue)\n"
                                 "- Backend frameworks (Node.js, Django, Flask)\n"
                                 "- Database design\n"
                                 "- RESTful APIs\n"
                                 "- Version control (Git)",
                'ai_ml': "AI/ML roles require:\n"
                        "- Deep learning frameworks (TensorFlow, PyTorch)\n"
                        "- Neural networks\n"
                        "- Natural language processing\n"
                        "- Computer vision\n"
                        "- Reinforcement learning\n"
                        "- Strong mathematical foundation",
                'core_engineering': "Core Engineering roles require:\n"
                                   "- Strong fundamentals in your domain\n"
                                   "- Problem-solving skills\n"
                                   "- Technical knowledge\n"
                                   "- Project experience\n"
                                   "- Industry-specific tools"
            },
            'preparation': {
                'default': "Here's a comprehensive preparation guide:\n\n"
                          "1. **Technical Preparation:**\n"
                          "   - Practice coding problems (LeetCode, HackerRank)\n"
                          "   - Review data structures and algorithms\n"
                          "   - Study system design basics\n\n"
                          "2. **Resume:**\n"
                          "   - Highlight relevant projects\n"
                          "   - Quantify achievements\n"
                          "   - Keep it concise (1-2 pages)\n\n"
                          "3. **Interview Skills:**\n"
                          "   - Practice mock interviews\n"
                          "   - Prepare STAR method for behavioral questions\n"
                          "   - Research the company\n\n"
                          "4. **Communication:**\n"
                          "   - Practice explaining your projects\n"
                          "   - Work on clarity and confidence\n\n"
                          "Would you like tips for a specific area?",
                'coding': "Coding Interview Tips:\n"
                         "- Practice daily on platforms like LeetCode\n"
                         "- Focus on arrays, strings, trees, graphs\n"
                         "- Understand time/space complexity\n"
                         "- Practice explaining your approach\n"
                         "- Start with brute force, then optimize",
                'resume': "Resume Tips:\n"
                         "- Use action verbs (developed, implemented, optimized)\n"
                         "- Include metrics (improved performance by 30%)\n"
                         "- Tailor resume for each role\n"
                         "- Keep formatting clean and consistent\n"
                         "- Proofread multiple times",
                'behavioral': "Behavioral Interview Tips:\n"
                             "- Prepare STAR stories (Situation, Task, Action, Result)\n"
                             "- Research company values\n"
                             "- Prepare questions to ask interviewer\n"
                             "- Show enthusiasm and cultural fit\n"
                             "- Be authentic and honest"
            },
            'statistics': {
                'default': "Placement Statistics (Sample Data):\n\n"
                          "**Overall:**\n"
                          "- Average Package: 8-12 LPA\n"
                          "- Highest Package: 25+ LPA\n\n"
                          "**By Domain:**\n"
                          "- Data Science: 10-15 LPA average\n"
                          "- Web Development: 8-12 LPA average\n"
                          "- AI/ML: 12-18 LPA average\n"
                          "- Core Engineering: 6-10 LPA average\n\n"
                          "Note: These are illustrative statistics for demo purposes."
            },
            'fallback': [
                "I’m mainly a placement assistant, but here’s how I’d think about that: try to relate your question to your skills, projects, and the kind of roles you want.",
                "I may not have a perfect answer for this, but I can still help you frame it in terms of placements, skills, and preparation.",
                "That’s an interesting question. I’m focused on placements, so tell me what role or company you’re targeting and I’ll connect it to that.",
                "I might not know everything about that topic, but we can still discuss how it affects your resume, skills, or placement strategy.",
                "I don’t have an exact answer, but I can help you turn this into a concrete plan for improving your profile or preparation."
            ]
        }
    
    def generate_response(self, intent, query_text=""):
        """
        Generate response based on intent
        
        Args:
            intent: Predicted intent class
            query_text: Original query text for context
        
        Returns:
            Response string
        """
        import random
        
        if intent == 'greeting':
            return random.choice(self.responses['greeting'])
        
        if intent == 'goodbye':
            return random.choice(self.responses['goodbye'])
        
        if intent == 'eligibility':
            # Try to extract company name from query
            query_lower = query_text.lower()
            if 'google' in query_lower:
                return self.responses['eligibility']['google']
            elif 'microsoft' in query_lower or 'ms' in query_lower:
                return self.responses['eligibility']['microsoft']
            elif 'amazon' in query_lower:
                return self.responses['eligibility']['amazon']
            else:
                return self.responses['eligibility']['default']
        
        if intent == 'company_info':
            query_lower = query_text.lower()
            if 'google' in query_lower:
                return self.responses['company_info']['google']
            elif 'microsoft' in query_lower or 'ms' in query_lower:
                return self.responses['company_info']['microsoft']
            elif 'amazon' in query_lower:
                return self.responses['company_info']['amazon']
            else:
                return self.responses['company_info']['default']
        
        if intent == 'skill_requirements':
            query_lower = query_text.lower()
            if 'data' in query_lower and 'science' in query_lower:
                return self.responses['skill_requirements']['data_science']
            elif 'web' in query_lower or 'frontend' in query_lower or 'backend' in query_lower:
                return self.responses['skill_requirements']['web_development']
            elif 'ai' in query_lower or 'ml' in query_lower or 'machine learning' in query_lower:
                return self.responses['skill_requirements']['ai_ml']
            elif 'core' in query_lower or 'engineering' in query_lower:
                return self.responses['skill_requirements']['core_engineering']
            else:
                return self.responses['skill_requirements']['default']
        
        if intent == 'preparation':
            query_lower = query_text.lower()
            if 'coding' in query_lower or 'programming' in query_lower:
                return self.responses['preparation']['coding']
            elif 'resume' in query_lower or 'cv' in query_lower:
                return self.responses['preparation']['resume']
            elif 'behavioral' in query_lower or 'hr' in query_lower:
                return self.responses['preparation']['behavioral']
            else:
                return self.responses['preparation']['default']
        
        if intent == 'statistics':
            return self.responses['statistics']['default']
        
        # Default response
        return "I understand you're asking about placements. Could you rephrase your question? I can help with eligibility, company info, skills, preparation tips, or statistics."

