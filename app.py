from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import PyPDF2
import docx
from datetime import datetime
import uuid
import logging
from pymilvus import MilvusClient  # Fixed: removed 'model' from import
from werkzeug.utils import secure_filename
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost", "https://your-flutter-web-domain.com"])

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE_FILE = 'resume_history.db'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Milvus client
try:
    client = MilvusClient(DATABASE_FILE)
    logger.info(f"Milvus client initialized with database: {DATABASE_FILE}")
except Exception as e:
    logger.error(f"Failed to initialize Milvus client: {e}")
    client = None

# Collection name
COLLECTION_NAME = "resume_history"

def initialize_milvus():
    """Initialize Milvus collection for storing resume history"""
    if client is None:
        logger.error("Milvus client not initialized")
        return False
    
    try:
        # Check if collection exists
        collections = client.list_collections()
        
        if COLLECTION_NAME not in collections:
            # Create collection
            client.create_collection(
                collection_name=COLLECTION_NAME,
                dimension=384,  # Default dimension for sentence transformers
                metric_type="IP",  # Inner Product
                consistency_level="Strong"
            )
            logger.info(f"Created collection: {COLLECTION_NAME}")
        else:
            logger.info(f"Collection {COLLECTION_NAME} already exists")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Milvus collection: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path, filename):
    """Extract text from uploaded file"""
    try:
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
                
        elif file_extension == 'docx':
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
            
        elif file_extension == 'doc':
            # For .doc files, you might need python-docx2txt or similar
            return "DOC file processing not fully implemented. Please use DOCX or PDF."
            
        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
                
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        return None

def calculate_ats_score(text):
    """Calculate ATS score based on resume content"""
    if not text:
        return 0, ["Resume text could not be extracted"]
    
    score = 50  # Base score
    suggestions = []
    
    # Check for contact information
    if any(keyword in text.lower() for keyword in ['email', '@', 'phone', 'linkedin']):
        score += 10
    else:
        suggestions.append("Add contact information (email, phone, LinkedIn)")
    
    # Check for professional summary/objective
    if any(keyword in text.lower() for keyword in ['summary', 'objective', 'profile']):
        score += 10
    else:
        suggestions.append("Add a professional summary or objective section")
    
    # Check for work experience keywords
    if any(keyword in text.lower() for keyword in ['experience', 'work', 'employment', 'job']):
        score += 15
    else:
        suggestions.append("Include work experience section")
    
    # Check for education
    if any(keyword in text.lower() for keyword in ['education', 'degree', 'university', 'college']):
        score += 10
    else:
        suggestions.append("Add education information")
    
    # Check for skills section
    if any(keyword in text.lower() for keyword in ['skills', 'technologies', 'programming']):
        score += 10
    else:
        suggestions.append("Include a skills section")
    
    # Check for action verbs
    action_verbs = ['managed', 'developed', 'created', 'improved', 'increased', 'decreased', 'achieved']
    if any(verb in text.lower() for verb in action_verbs):
        score += 5
    else:
        suggestions.append("Use more action verbs to describe achievements")
    
    return min(score, 100), suggestions

def store_resume_history(user_email, filename, file_path, score, suggestions, analysis_date):
    """Store resume analysis in Milvus database"""
    if client is None:
        logger.error("Milvus client not available")
        return None
    
    try:
        # Generate unique ID
        record_id = str(uuid.uuid4())
        
        # Create a simple vector (in real implementation, you might use text embeddings)
        vector = [0.1] * 384  # Dummy vector for demonstration
        
        # Prepare data for insertion
        data = {
            "id": record_id,
            "vector": vector,
            "user_email": user_email,
            "filename": filename,
            "file_path": file_path,
            "ats_score": score,
            "suggestions": json.dumps(suggestions),
            "analysis_date": analysis_date,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        # Insert data
        client.insert(
            collection_name=COLLECTION_NAME,
            data=[data]
        )
        
        logger.info(f"Stored resume history for user: {user_email}")
        return record_id
        
    except Exception as e:
        logger.error(f"Failed to store resume history: {e}")
        return None

def get_user_history(user_email, limit=10):
    """Retrieve user's resume analysis history"""
    if client is None:
        logger.error("Milvus client not available")
        return []
    
    try:
        # Search for user's records
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'user_email == "{user_email}"',
            output_fields=["id", "user_email", "filename", "file_path", "ats_score", "suggestions", "analysis_date", "upload_timestamp"],
            limit=limit
        )
        
        # Sort by analysis_date descending
        sorted_results = sorted(results, key=lambda x: x.get('analysis_date', ''), reverse=True)
        
        # Parse suggestions back from JSON
        for result in sorted_results:
            if 'suggestions' in result and result['suggestions']:
                try:
                    result['suggestions'] = json.loads(result['suggestions'])
                except:
                    result['suggestions'] = []
        
        return sorted_results
        
    except Exception as e:
        logger.error(f"Failed to retrieve user history: {e}")
        return []

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Check if file is uploaded
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        user_email = request.form.get('user_email', 'anonymous@example.com')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File size too large (max 5MB)'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        # Extract text and analyze
        text = extract_text_from_file(file_path, filename)
        if text is None:
            return jsonify({'error': 'Failed to extract text from file'}), 500
        
        score, suggestions = calculate_ats_score(text)
        analysis_date = datetime.now().isoformat()
        
        # Store in history
        record_id = store_resume_history(
            user_email=user_email,
            filename=filename,
            file_path=file_path,
            score=score,
            suggestions=suggestions,
            analysis_date=analysis_date
        )
        
        response_data = {
            'score': score,
            'suggestions': suggestions,
            'file_processed': filename,
            'analysis_date': analysis_date,
            'record_id': record_id
        }
        
        logger.info(f"Successfully analyzed resume: {filename} with score: {score}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in analyze_resume: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/history/<user_email>', methods=['GET'])
def get_history(user_email):
    """Get user's resume analysis history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        history = get_user_history(user_email, limit)
        
        return jsonify({
            'success': True,
            'history': history,
            'total_count': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving history for {user_email}: {e}")
        return jsonify({'error': 'Failed to retrieve history'}), 500

@app.route('/download/<record_id>', methods=['GET'])
def download_resume(record_id):
    """Download a resume file by record ID"""
    try:
        if client is None:
            return jsonify({'error': 'Database not available'}), 500
        
        # Get record details
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'id == "{record_id}"',
            output_fields=["file_path", "filename"],
            limit=1
        )
        
        if not results:
            return jsonify({'error': 'File not found'}), 404
        
        file_path = results[0]['file_path']
        original_filename = results[0]['filename']
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File no longer exists on server'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=original_filename
        )
        
    except Exception as e:
        logger.error(f"Error downloading file for record {record_id}: {e}")
        return jsonify({'error': 'Failed to download file'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'milvus_connected': client is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Initialize Milvus collection on startup
    initialize_milvus()
    
    # Get port from environment variable (Render uses PORT)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
