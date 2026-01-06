"""
Flask Web Application
Main application file for the College Placement Chatbot
"""

import os
import traceback
from flask import Flask, render_template, request, jsonify, send_from_directory, abort  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore
from chatbot.chatbot import PlacementChatbot
from analytics.resume_analytics import ResumeAnalytics
from analytics.visualizations import AnalyticsVisualizer
from nlp.resume_parser import ResumeParser
from utils.helpers import allowed_file, ensure_directories
from config import UPLOAD_FOLDER, MAX_CONTENT_LENGTH

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'

# Ensure directories exist
ensure_directories()
os.makedirs('static/images', exist_ok=True)

# Initialize components
chatbot = PlacementChatbot()
resume_analytics = ResumeAnalytics()
resume_parser = ResumeParser()
visualizer = AnalyticsVisualizer()

# Initialize models (with error handling)
models_initialized = False
try:
    chatbot.initialize()
    resume_analytics.initialize()
    models_initialized = True
    print("✓ Models initialized successfully")
except Exception as e:
    print(f"⚠ Warning: Models not initialized. Please train models first: {e}")
    print("Run: python train_models.py")
    models_initialized = False


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    try:
        if not models_initialized:
            return jsonify({
                'response': 'Chatbot is not ready. Please train the models first by running: python train_models.py',
                'intent': None,
                'confidence': 0.0
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'response': 'Invalid request. Please send JSON data.',
                'intent': None,
                'confidence': 0.0
            }), 400
        
        user_query = data.get('message', '').strip()
        
        if not user_query:
            return jsonify({
                'response': 'Please enter a valid query.',
                'intent': None,
                'confidence': 0.0
            }), 400
        
        # Process query
        result = chatbot.process_query(user_query)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Chat error: {traceback.format_exc()}")
        return jsonify({
            'response': f'Sorry, an error occurred: {str(e)}',
            'intent': None,
            'confidence': 0.0
        }), 500


@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    """Handle resume upload and analysis"""
    try:
        if not models_initialized:
            return jsonify({'error': 'Resume analytics is not ready. Please train the models first by running: python train_models.py'}), 503
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not filename:
                return jsonify({'error': 'Invalid filename'}), 400
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except Exception as save_error:
                return jsonify({'error': f'Error saving file: {str(save_error)}'}), 500
            
            # Parse resume
            try:
                resume_text = resume_parser.parse(filepath)
            except Exception as parse_error:
                return jsonify({'error': f'Error parsing resume: {str(parse_error)}'}), 500
            
            if not resume_text or len(resume_text.strip()) < 50:
                return jsonify({'error': 'Could not extract sufficient text from resume. Please ensure the file contains readable text.'}), 400
            
            # Analyze resume
            try:
                analysis = resume_analytics.analyze_resume(resume_text)
            except Exception as analysis_error:
                return jsonify({'error': f'Error analyzing resume: {str(analysis_error)}'}), 500
            
            # Generate visualizations
            viz_files = {}
            try:
                # Skill frequency
                all_skills = analysis.get('technical_skills', []) + analysis.get('soft_skills', [])
                if all_skills:
                    skill_freq_path = visualizer.plot_skill_frequency(
                        all_skills,
                        filename=f"skill_freq_{filename.rsplit('.', 1)[0]}.png"
                    )
                    if skill_freq_path:
                        viz_files['skill_frequency'] = '/static/images/' + os.path.basename(skill_freq_path)
                
                # Skill gaps
                skill_gaps_data = {
                    'required_skills': analysis.get('skill_gaps', []),
                    'found_skills': analysis.get('technical_skills', []),
                    'gaps': analysis.get('skill_gaps', [])
                }
                if skill_gaps_data['required_skills'] or skill_gaps_data['found_skills']:
                    skill_gaps_path = visualizer.plot_skill_gaps(
                        skill_gaps_data,
                        filename=f"skill_gaps_{filename.rsplit('.', 1)[0]}.png"
                    )
                    if skill_gaps_path:
                        viz_files['skill_gaps'] = '/static/images/' + os.path.basename(skill_gaps_path)
                
                # Word cloud
                wordcloud_path = visualizer.generate_wordcloud(
                    resume_text,
                    filename=f"wordcloud_{filename.rsplit('.', 1)[0]}.png"
                )
                if wordcloud_path:
                    viz_files['wordcloud'] = '/static/images/' + os.path.basename(wordcloud_path)
            except Exception as viz_error:
                print(f"Visualization error: {viz_error}")
                print(traceback.format_exc())
                # Continue without visualizations
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'visualizations': viz_files,
                'resume_text_length': len(resume_text)
            })
        
        else:
            return jsonify({'error': 'Invalid file type. Allowed: PDF, DOC, DOCX, TXT'}), 400
    
    except Exception as e:
        print(f"Upload resume error: {traceback.format_exc()}")
        return jsonify({'error': f'Error processing resume: {str(e)}'}), 500


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze plain text resume"""
    try:
        if not models_initialized:
            return jsonify({'error': 'Resume analytics is not ready. Please train the models first by running: python train_models.py'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request. Please send JSON data.'}), 400
        
        resume_text = data.get('text', '').strip()
        
        if not resume_text or len(resume_text) < 50:
            return jsonify({'error': 'Please provide sufficient resume text (at least 50 characters)'}), 400
        
        # Analyze resume
        analysis = resume_analytics.analyze_resume(resume_text)
        
        # Generate visualizations
        viz_files = {}
        try:
            # Skill frequency
            all_skills = analysis.get('technical_skills', []) + analysis.get('soft_skills', [])
            if all_skills:
                skill_freq_path = visualizer.plot_skill_frequency(
                    all_skills,
                    filename="skill_freq_text.png"
                )
                if skill_freq_path:
                    viz_files['skill_frequency'] = '/static/images/' + os.path.basename(skill_freq_path)
            
            # Skill gaps
            skill_gaps_data = {
                'required_skills': analysis.get('skill_gaps', []),
                'found_skills': analysis.get('technical_skills', []),
                'gaps': analysis.get('skill_gaps', [])
            }
            if skill_gaps_data['required_skills'] or skill_gaps_data['found_skills']:
                skill_gaps_path = visualizer.plot_skill_gaps(
                    skill_gaps_data,
                    filename="skill_gaps_text.png"
                )
                if skill_gaps_path:
                    viz_files['skill_gaps'] = '/static/images/' + os.path.basename(skill_gaps_path)
            
            # Word cloud
            wordcloud_path = visualizer.generate_wordcloud(
                resume_text,
                filename="wordcloud_text.png"
            )
            if wordcloud_path:
                viz_files['wordcloud'] = '/static/images/' + os.path.basename(wordcloud_path)
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
            print(traceback.format_exc())
            # Continue without visualizations
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'visualizations': viz_files
        })
    
    except Exception as e:
        print(f"Analyze text error: {traceback.format_exc()}")
        return jsonify({'error': f'Error analyzing text: {str(e)}'}), 500


@app.route('/static/images/<filename>')
def serve_image(filename):
    """Serve generated images"""
    try:
        return send_from_directory(visualizer.output_dir, filename)
    except FileNotFoundError:
        abort(404)
    except Exception as e:
        print(f"Error serving image: {e}")
        abort(500)


if __name__ == '__main__':
    print("="*50)
    print("College Placement Chatbot - Starting Flask App")
    print("="*50)
    print("\nAccess the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
