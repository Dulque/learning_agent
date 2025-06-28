# I Will Teach You – AI Learning Assistant

**I Will Teach You** is an AI-powered interactive learning assistant built using Flask and Ollama. It supports file uploads, question answering, concept explanations, interactive quizzes, and memory map generation based on user input and uploaded documents.

## 🧠 Features

- 📚 **Ask Questions**: Get clear answers based on your uploaded materials.
- 💡 **Explain Concepts**: AI breaks down any concept by user-selected difficulty (beginner, intermediate, advanced).
- 📝 **Interactive Quizzes**: Auto-generated quizzes with scoring, explanations, and feedback.
- 🗺️ **Memory Maps**: ASCII-style mind maps to help visualize topics.
- 📤 **File Uploads**: Upload PDFs or text files for AI to learn and answer based on your material.
- ⚡ **Asynchronous Document Processing**: Fast background storage using FAISS and vector embeddings.

## 🚀 Technologies Used

- **Python**
- **Flask**
- **Ollama** (for LLM and embeddings, e.g., LLaMA3, nomic-embed-text)
- **FAISS** (for similarity search)
- **Joblib** (to persist chunked documents)
- **PyPDF2** (PDF parsing)
- **HTML/CSS/Bootstrap** (frontend UI)

## 🔧 Installation

### Requirements

- Python 3.8+
- Ollama installed and running locally
- Flask
- faiss-cpu
- joblib
- numpy
- PyPDF2

### Setup Instructions

1. **Clone the repository**
   git clone <your-repo-url>
   cd learning_agent-main
   
2.Install dependencies:
pip install flask ollama faiss-cpu joblib numpy PyPDF2

3.Run the app:
python learning_agent.py

4.Open your browser and go to http://localhost:5000.

👤 Author
Umer Murthala Thangal K K

