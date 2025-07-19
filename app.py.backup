from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import PyPDF2
import docx
import re
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime
import logging
from io import BytesIO
from urllib.parse import urlparse
import requests
from collections import Counter

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error extracting TXT text: {str(e)}")
        return ""

def extract_text_from_file(file_path, file_extension):
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == 'docx':
        return extract_text_from_docx(file_path)
    elif file_extension == 'txt':
        return extract_text_from_txt(file_path)
    else:
        return ""

def analyze_resume_content(text):
    text_lower = text.lower()
    score = 0
    suggestions = []

    criteria = {
        'contact_info': {
            'patterns': [r'email|@', r'phone|tel|\d{3}[-.]?\d{3}[-.]?\d{4}', r'linkedin|github'],
            'weight': 15,
            'suggestion': 'Include complete contact information (email, phone, LinkedIn)'
        },
        'experience': {
            'patterns': [r'experience|work|job|position|role', r'company|organization|corp'],
            'weight': 25,
            'suggestion': 'Add more detailed work experience with specific roles and companies'
        },
        'education': {
            'patterns': [r'education|degree|university|college|school', r'bachelor|master|phd|diploma'],
            'weight': 20,
            'suggestion': 'Include educational background with degrees and institutions'
        },
        'skills': {
            'patterns': [r'skills|technical|programming|software', r'python|java|javascript|html|css'],
            'weight': 20,
            'suggestion': 'List relevant technical and soft skills'
        },
        'achievements': {
            'patterns': [r'achievement|award|project|accomplishment', r'led|managed|developed|created'],
            'weight': 10,
            'suggestion': 'Highlight key achievements and projects'
        },
        'keywords': {
            'patterns': [r'responsible|managed|developed|implemented|designed|created'],
            'weight': 10,
            'suggestion': 'Use more action verbs and industry-specific keywords'
        }
    }

    for criterion, data in criteria.items():
        criterion_score = 0
        for pattern in data['patterns']:
            if re.search(pattern, text_lower):
                criterion_score += data['weight'] / len(data['patterns'])

        score += min(criterion_score, data['weight'])

        if criterion_score < data['weight'] * 0.6:
            suggestions.append(data['suggestion'])

    if len(text) > 500:
        score += 5
    if len(text) > 1000:
        score += 5

    score = min(score, 100)

    return int(score), suggestions

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Resume Analyzer API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Option 1: Analyze from URL
        if 'resume_url' in request.form:
            resume_url = request.form['resume_url'].strip()
            if not resume_url:
                return jsonify({'error': 'Resume URL is empty'}), 400

            scorer = ATSResumeScorer()
            result = scorer.score_resume(resume_url)

            if 'error' in result:
                return jsonify({'error': result['error']}), 400
            return jsonify(result), 200

        # Option 2: Analyze uploaded file
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400

        file = request.files['resume']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF, DOC, DOCX, and TXT files are allowed'}), 400

        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            text = extract_text_from_file(temp_file_path, file_extension)

            if not text.strip():
                return jsonify({'error': 'Could not extract text from the file'}), 400

            score, suggestions = analyze_resume_content(text)

            response = {
                'score': score,
                'suggestions': suggestions,
                'analysis_date': datetime.now().isoformat(),
                'file_processed': filename
            }

            logger.info(f"Resume analyzed successfully: {filename}, Score: {score}")
            return jsonify(response)

        finally:
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {str(e)}")

    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        return jsonify({'error': 'An error occurred while analyzing the resume'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 5MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ==========================================
# ⚙️ Paste the full ATSResumeScorer class here
# ==========================================

# Paste your full ATSResumeScorer class definition here

# ==========================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
