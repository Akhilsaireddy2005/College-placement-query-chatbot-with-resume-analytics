"""
Text Analytics Visualization Module
Generates charts and graphs for resume analytics
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from wordcloud import WordCloud

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class AnalyticsVisualizer:
    """Generate visualizations for text analytics"""
    
    def __init__(self, output_dir='static/images'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_skill_frequency(self, skills_list, title="Skill Frequency Analysis", filename="skill_frequency.png"):
        """
        Plot skill frequency bar chart
        
        Args:
            skills_list: List of skills (can have duplicates)
            title: Chart title
            filename: Output filename
        """
        if not skills_list:
            return None
        
        skill_counts = Counter(skills_list)
        top_skills = dict(skill_counts.most_common(15))
        
        plt.figure(figsize=(12, 6))
        skills = list(top_skills.keys())
        counts = list(top_skills.values())
        
        bars = plt.barh(skills, counts, color='steelblue')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Skills', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (skill, count) in enumerate(zip(skills, counts)):
            plt.text(count + 0.1, i, str(count), va='center', fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_domain_distribution(self, domain_counts, title="Job Domain Distribution", filename="domain_distribution.png"):
        """
        Plot domain distribution pie chart
        
        Args:
            domain_counts: Dictionary of domain -> count
            title: Chart title
            filename: Output filename
        """
        if not domain_counts:
            return None
        
        plt.figure(figsize=(10, 8))
        domains = list(domain_counts.keys())
        counts = list(domain_counts.values())
        colors = sns.color_palette("Set3", len(domains))
        
        plt.pie(counts, labels=domains, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('equal')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_readiness_scores(self, scores, title="Resume Readiness Scores", filename="readiness_scores.png"):
        """
        Plot readiness scores histogram
        
        Args:
            scores: List of readiness scores
            title: Chart title
            filename: Output filename
        """
        if not scores:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Readiness Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_skill_gaps(self, gaps_data, title="Skill Gap Analysis", filename="skill_gaps.png"):
        """
        Plot skill gaps bar chart
        
        Args:
            gaps_data: Dictionary with 'required', 'found', 'gaps' keys
            title: Chart title
            filename: Output filename
        """
        required = gaps_data.get('required_skills', [])
        found = gaps_data.get('found_skills', [])
        gaps = gaps_data.get('gaps', [])
        
        if not required:
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Count skills
        categories = ['Required Skills', 'Found Skills', 'Missing Skills']
        counts = [len(required), len(found), len(gaps)]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        plt.ylabel('Number of Skills', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_wordcloud(self, text, title="Resume Word Cloud", filename="wordcloud.png"):
        """
        Generate word cloud from text
        
        Args:
            text: Input text string
            title: Chart title
            filename: Output filename
        """
        if not text:
            return None

        # Basic cleanup and light stopwording to improve readability
        custom_stopwords = {
            'system', 'using', 'use', 'project', 'management', 'built',
            'developed', 'based', 'application', 'design', 'work', 'role',
            'team', 'tasks', 'django', 'react', 'github', 'linkedin',
            'resume', 'experience'
        }

        plt.figure(figsize=(14, 8))
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            max_words=120,
            colormap='viridis',
            prefer_horizontal=0.9,
            collocations=False,
            margin=2,
            stopwords=custom_stopwords,
            max_font_size=140,
            min_font_size=12,
        ).generate(text.lower())

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold', pad=18)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath
    
    def plot_comparison_chart(self, data_dict, title="Comparison Chart", filename="comparison.png"):
        """
        Plot comparison chart for multiple metrics
        
        Args:
            data_dict: Dictionary of metric -> value
            title: Chart title
            filename: Output filename
        """
        if not data_dict:
            return None
        
        plt.figure(figsize=(10, 6))
        metrics = list(data_dict.keys())
        values = list(data_dict.values())
        
        bars = plt.bar(metrics, values, color='coral', alpha=0.7, edgecolor='black')
        plt.ylabel('Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
