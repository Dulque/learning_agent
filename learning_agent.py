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
import time
import re

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

# New interactive quiz endpoints
@app.route('/start_quiz', methods=['POST'])
def start_quiz():
    data = request.get_json()
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400
    
    try:
        system_prompt = """Generate exactly 5 quiz questions with:
        - Clear question text
        - 4 options (A-D)
        - Mark correct answer with [Correct]
        - Brief explanation
        Format each as:
        Q1: [Question]
        A) [Option A]
        B) [Option B] [Correct]
        C) [Option C]
        D) [Option D]
        Explanation: [Explanation]"""
        
        quiz_response = generate_content(f"Generate a quiz about {topic}", system_prompt)
        questions = parse_quiz_response(quiz_response)
        
        if not questions:
            return jsonify({"error": "Failed to generate valid quiz"}), 500
        
        session_id = str(uuid.uuid4())
        quiz_sessions[session_id] = {
            'topic': topic,
            'questions': questions,
            'current_question': 0,
            'score': 0,
            'user_answers': {}
        }
        
        return jsonify({
            "session_id": session_id,
            "question": questions[0]['text'],
            "options": questions[0]['options']
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to start quiz",
            "details": str(e)
        }), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    data = request.get_json()
    session_id = data.get('session_id')
    question_id = data.get('question_id')
    answer = data.get('answer')
    
    if not session_id or session_id not in quiz_sessions:
        return jsonify({"error": "Invalid quiz session"}), 400
    
    quiz = quiz_sessions[session_id]
    if question_id >= len(quiz['questions']):
        return jsonify({"error": "Invalid question"}), 400
    
    question = quiz['questions'][question_id]
    is_correct = answer.upper() == question['correct_answer']
    
    # Update quiz state
    quiz['user_answers'][question_id] = answer
    if is_correct:
        quiz['score'] += 1
    
    # Move to next question
    next_question_id = question_id + 1
    
    if next_question_id < len(quiz['questions']):
        next_question = quiz['questions'][next_question_id]
        return jsonify({
            "correct": is_correct,
            "explanation": question['explanation'],
            "next_question": next_question['text'],
            "next_options": next_question['options'],
            "question_id": next_question_id,
            "score": quiz['score']
        })
    else:
        # Quiz completed - generate analysis
        analysis = generate_quiz_analysis(quiz)
        return jsonify({
            "correct": is_correct,
            "explanation": question['explanation'],
            "completed": True,
            "final_score": quiz['score'],
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

# Helper functions for quiz
def parse_quiz_response(response):
    """Parse the quiz response into structured questions"""
    questions = []
    current_question = {}
    
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Q') or line.startswith('Question'):
            if current_question:
                questions.append(current_question)
            current_question = {
                'text': line,
                'options': [],
                'correct_answer': None,
                'explanation': None
            }
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            current_question['options'].append(line)
            if '[Correct]' in line:
                current_question['correct_answer'] = line[0]
                current_question['options'][-1] = line.replace('[Correct]', '').strip()
        elif line.startswith('Explanation:'):
            current_question['explanation'] = line.replace('Explanation:', '').strip()
    
    if current_question:
        questions.append(current_question)
    
    return questions[:5]  # Return max 5 questions

def generate_quiz_analysis(quiz):
    """Generate personalized analysis based on quiz results"""
    topic = quiz['topic']
    score = quiz['score']
    total = len(quiz['questions'])
    
    # Build results summary
    results = []
    for i, q in enumerate(quiz['questions']):
        user_answer = quiz['user_answers'].get(i, "No answer")
        results.append({
            "question": q['text'],
            "correct_answer": q['correct_answer'],
            "user_answer": user_answer,
            "is_correct": user_answer == q['correct_answer']
        })
    
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