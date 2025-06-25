from flask import Flask, request, jsonify, render_template, session
import ollama
import numpy as np
import os
import uuid
import PyPDF2
import faiss
import joblib
from threading import Thread
from werkzeug.utils import secure_filename
import re
import json

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['VECTOR_DB_PATH'] = 'vector_db.faiss'
app.config['CHUNKS_PATH'] = 'chunks.joblib'

# Configuration
EMBEDDING_MODEL = 'nomic-embed-text'
LLM_MODEL = 'llama3'
vector_index = None
chunks = []
model_loaded = False
quiz_sessions = {}  # Stores active quiz sessions

# Pre-download and warm up models
print("Initializing models...")
try:
    ollama.pull(EMBEDDING_MODEL)
    ollama.pull(LLM_MODEL)
    print("Models loaded and warmed up")
    model_loaded = True
except Exception as e:
    print(f"Model initialization error: {str(e)}")
    exit(1)

# Load or create vector database
def init_vector_db():
    global vector_index, chunks
    if os.path.exists(app.config['VECTOR_DB_PATH']):
        vector_index = faiss.read_index(app.config['VECTOR_DB_PATH'])
        chunks = joblib.load(app.config['CHUNKS_PATH'])
        print(f"Loaded vector DB with {len(chunks)} chunks")
    else:
        vector_index = faiss.IndexFlatL2(768)
        chunks = []
        print("Created new vector DB")

# Initialize at startup
init_vector_db()

def extract_text_from_file(filepath):
    """Optimized text extraction with limits"""
    if filepath.endswith('.pdf'):
        text = ""
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages[:3]:  # Reduce to 3 pages for speed
                text += page.extract_text() + "\n"
        return text[:5000]  # Limit to 5K characters
    elif filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read(10000)  # Read only first 10KB
    return ""

def split_text(text, chunk_size=300):  # Smaller chunks for faster processing
    """Efficient text splitting"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_document_async(text):
    """Store document in background thread"""
    def process():
        global vector_index, chunks
        new_chunks = split_text(text)
        new_embeddings = []
        
        for chunk in new_chunks:
            try:
                response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
                embedding = np.array(response['embedding'], dtype='float32').reshape(1, -1)
                new_embeddings.append(embedding)
            except Exception as e:
                print(f"Embedding error: {str(e)}")
        
        if new_embeddings:
            embeddings_matrix = np.vstack(new_embeddings)
            vector_index.add(embeddings_matrix)
            chunks.extend(new_chunks)
            
            faiss.write_index(vector_index, app.config['VECTOR_DB_PATH'])
            joblib.dump(chunks, app.config['CHUNKS_PATH'])
    
    Thread(target=process).start()

def retrieve_relevant_chunks(query, top_n=2):  # Reduce top_n for speed
    """Efficient similarity search with FAISS"""
    if vector_index.ntotal == 0:
        return []

    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
        query_embedding = np.array(response['embedding'], dtype='float32').reshape(1, -1)
        distances, indices = vector_index.search(query_embedding, top_n)
        return [chunks[i] for i in indices[0] if i < len(chunks)]
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return []

def generate_content(prompt, system_prompt=None, max_tokens=256):  # Reduced max tokens
    """Faster generation with optimizations"""
    try:
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt or "You are a helpful and knowledgeable learning assistant."}
        <|eot_id|>{prompt}"""
        
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=full_prompt,
            stream=False,
            options={
                'num_predict': max_tokens,  # Reduced for faster responses
                'temperature': 0.7,
                'num_ctx': 1024  # Smaller context window
            }
        )
        return response['response']
    except Exception as e:
        return f"Error generating content: {str(e)}"

# Session management
@app.before_request
def setup_session():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    if 'current_topic' not in session:
        session['current_topic'] = None

# Routes
@app.route('/')
def home():
    return render_template('index.html', agent_name="I Will Teach You")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        return jsonify({"error": "File type not allowed"}), 400
    
    try:
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        text = extract_text_from_file(filepath)
        store_document_async(text)
        
        return jsonify({
            "success": True,
            "message": "File is being processed in background",
        })
    except Exception as e:
        return jsonify({
            "error": "File processing failed",
            "details": str(e)
        }), 500

@app.route('/learn', methods=['POST'])
def learn_topic():
    data = request.get_json()
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400
    
    try:
        explanation = generate_content(
            f"Explain the topic of {topic} in comprehensive detail with key concepts and examples",
            "You are an expert educator. Provide clear, structured explanations with examples.",
            max_tokens=384
        )
        
        session['current_topic'] = topic
        return jsonify({
            "topic": topic,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to learn topic",
            "details": str(e)
        }), 500

@app.route('/explain', methods=['POST'])
def explain_concept():
    data = request.get_json()
    concept = data.get('concept', '').strip()
    user_level = data.get('level', 'beginner')
    
    if not concept:
        return jsonify({"error": "Concept cannot be empty"}), 400
    
    try:
        explanation = generate_content(
            f"Explain the concept of {concept} to a {user_level} with clear examples",
            "You are a patient tutor who breaks down complex concepts into understandable parts.",
            max_tokens=384
        )
        
        return jsonify({
            "concept": concept,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to explain concept",
            "details": str(e)
        }), 500

# Quiz endpoints
@app.route('/start_quiz', methods=['POST'])
def start_quiz():
    data = request.get_json()
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400
    
    try:
        system_prompt = """Generate exactly 5 quiz questions in JSON format with the following structure:
        [
            {
                "question": "Question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "A",
                "explanation": "Brief explanation"
            }
        ]
        Only output the JSON array with no additional text. Ensure there are exactly 5 question objects."""
        
        quiz_response = generate_content(f"Generate a quiz about {topic}", system_prompt, max_tokens=1024)
        
        # Clean up the response to extract just the JSON
        try:
            # Remove any text outside of JSON brackets
            start_index = quiz_response.find('[')
            end_index = quiz_response.rfind(']') + 1
            json_str = quiz_response[start_index:end_index]
            
            questions = json.loads(json_str)
            
            if not questions or len(questions) < 5:
                return jsonify({"error": "Failed to generate valid quiz"}), 500
                
        except Exception as e:
            print(f"Quiz parsing error: {str(e)}")
            print(f"Original response: {quiz_response}")
            return jsonify({"error": "Failed to parse quiz response"}), 500
        
        session_id = str(uuid.uuid4())
        quiz_sessions[session_id] = {
            "topic": topic,
            "questions": questions[:5]  # Take first 5 questions
        }
        
        return jsonify({
            "session_id": session_id,
            "questions": questions[:5],
            "topic": topic
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to start quiz",
            "details": str(e)
        }), 500

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.get_json()
    session_id = data.get('session_id')
    answers = data.get('answers')
    
    if not session_id or session_id not in quiz_sessions:
        return jsonify({"error": "Invalid quiz session"}), 400
    
    session = quiz_sessions[session_id]
    questions = session['questions']
    results = []
    score = 0
    
    # Check each answer
    for i, question in enumerate(questions):
        user_answer = answers.get(str(i), "").upper()
        correct = user_answer == question['correct_answer']
        results.append({
            "question": question['question'],
            "user_answer": user_answer,
            "correct_answer": question['correct_answer'],
            "explanation": question['explanation'],
            "correct": correct
        })
        
        if correct:
            score += 1
    
    # Generate analysis
    topic = session['topic']
    analysis = generate_quiz_analysis(topic, score, len(questions), results)
    
    return jsonify({
        "score": score,
        "total": len(questions),
        "results": results,
        "analysis": analysis
    })

@app.route('/memory_map', methods=['POST'])
def create_memory_map():
    data = request.get_json()
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400
    
    try:
        mind_map = generate_content(
            f"Create a comprehensive mind map for: {topic}",
            "You are a visual learning expert. Create mind maps using ASCII art",
            max_tokens=256
        )
        return jsonify({
            "topic": topic,
            "memory_map": mind_map
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to create memory map",
            "details": str(e)
        }), 500

def generate_quiz_analysis(topic, score, total, results):
    """Generate personalized analysis based on quiz results"""
    # Generate analysis
    prompt = f"""The user scored {score}/{total} on a quiz about {topic}. 
    Here are the results:
    {str(results)}
    
    Provide a concise analysis of the user's performance, highlighting strengths and areas for improvement. 
    Offer specific learning suggestions to help them improve in the areas where they struggled."""
    
    return generate_content(
        prompt,
        "You are an expert learning advisor who provides constructive feedback",
        max_tokens=384
    )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)