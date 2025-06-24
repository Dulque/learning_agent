document.addEventListener('DOMContentLoaded', function() {
    // Upload form
    document.getElementById('uploadForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const fileInput = document.getElementById('fileInput');
        const statusDiv = document.getElementById('uploadStatus');
        
        if (fileInput.files.length === 0) {
            statusDiv.innerHTML = '<div class="alert alert-warning">Please select a file</div>';
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            if (response.ok) {
                statusDiv.innerHTML = '<div class="alert alert-success">File uploaded successfully and is being processed</div>';
            } else {
                statusDiv.innerHTML = `<div class="alert alert-danger">${data.error || 'Upload failed'}</div>`;
            }
        } catch (error) {
            statusDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    });
    
    // Learn form
    document.getElementById('learnForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const topic = document.getElementById('topicInput').value.trim();
        const resultDiv = document.getElementById('learnResult');
        
        if (!topic) {
            resultDiv.innerHTML = '<div class="alert alert-warning">Please enter a topic</div>';
            return;
        }
        
        try {
            const response = await fetch('/learn', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `topic=${encodeURIComponent(topic)}`
            });
            
            const data = await response.json();
            if (response.ok) {
                resultDiv.innerHTML = `
                    <div class="card">
                        <div class="card-header">
                            <h6>${data.topic}</h6>
                        </div>
                        <div class="card-body">
                            <p>${data.explanation}</p>
                        </div>
                    </div>
                `;
            } else {
                resultDiv.innerHTML = `<div class="alert alert-danger">${data.error || 'Failed to learn topic'}</div>`;
            }
        } catch (error) {
            resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    });
    
    // Explain form
    document.getElementById('explainForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const concept = document.getElementById('conceptInput').value.trim();
        const level = document.getElementById('levelSelect').value;
        const resultDiv = document.getElementById('explainResult');
        
        if (!concept) {
            resultDiv.innerHTML = '<div class="alert alert-warning">Please enter a concept</div>';
            return;
        }
        
        try {
            const response = await fetch('/explain', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `concept=${encodeURIComponent(concept)}&level=${encodeURIComponent(level)}`
            });
            
            const data = await response.json();
            if (response.ok) {
                resultDiv.innerHTML = `
                    <div class="card">
                        <div class="card-header">
                            <h6>${data.concept} (${level} level)</h6>
                        </div>
                        <div class="card-body">
                            <p>${data.explanation}</p>
                        </div>
                    </div>
                `;
            } else {
                resultDiv.innerHTML = `<div class="alert alert-danger">${data.error || 'Failed to explain concept'}</div>`;
            }
        } catch (error) {
            resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    });
    
    // Quiz functionality
    let currentQuizSession = null;
    let selectedOption = null;
    
    document.getElementById('quizForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const topic = document.getElementById('quizTopicInput').value.trim();
        const quizArea = document.getElementById('quizArea');
        const quizQuestion = document.getElementById('quizQuestion');
        const quizOptions = document.getElementById('quizOptions');
        const submitBtn = document.getElementById('submitAnswerBtn');
        const quizFeedback = document.getElementById('quizFeedback');
        
        if (!topic) {
            quizFeedback.innerHTML = '<div class="alert alert-warning">Please enter a quiz topic</div>';
            return;
        }
        
        try {
            const response = await fetch('/start_quiz', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `topic=${encodeURIComponent(topic)}`
            });
            
            const data = await response.json();
            if (response.ok) {
                currentQuizSession = data.session_id;
                quizArea.classList.remove('d-none');
                quizFeedback.innerHTML = '';
                
                // Display first question
                quizQuestion.innerHTML = `<h6>Question 1 of ${data.total_questions}</h6><p>${data.question}</p>`;
                
                quizOptions.innerHTML = '';
                data.options.forEach((option, index) => {
                    const optionDiv = document.createElement('div');
                    optionDiv.className = 'quiz-option';
                    optionDiv.textContent = option;
                    optionDiv.dataset.option = option.charAt(0);
                    optionDiv.addEventListener('click', function() {
                        // Remove selected class from all options
                        document.querySelectorAll('.quiz-option').forEach(opt => {
                            opt.classList.remove('selected');
                        });
                        // Add selected class to clicked option
                        this.classList.add('selected');
                        selectedOption = this.dataset.option;
                        submitBtn.classList.remove('d-none');
                    });
                    quizOptions.appendChild(optionDiv);
                });
                
                submitBtn.classList.add('d-none');
                submitBtn.addEventListener('click', submitQuizAnswer);
                
                updateQuizProgress();
            } else {
                quizFeedback.innerHTML = `<div class="alert alert-danger">${data.error || 'Failed to start quiz'}</div>`;
            }
        } catch (error) {
            quizFeedback.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    });
    
    async function submitQuizAnswer() {
        if (!selectedOption || !currentQuizSession) return;
        
        const quizQuestion = document.getElementById('quizQuestion');
        const quizOptions = document.getElementById('quizOptions');
        const quizFeedback = document.getElementById('quizFeedback');
        const submitBtn = document.getElementById('submitAnswerBtn');
        
        try {
            const response = await fetch('/submit_answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `session_id=${encodeURIComponent(currentQuizSession)}&question_id=${encodeURIComponent(quizQuestion.dataset.questionId || '0')}&answer=${encodeURIComponent(selectedOption)}`
            });
            
            const data = await response.json();
            if (response.ok) {
                // Highlight correct and wrong answers
                document.querySelectorAll('.quiz-option').forEach(option => {
                    if (option.dataset.option === data.correct_answer) {
                        option.classList.add('correct-answer');
                    } else if (option.dataset.option === selectedOption && !data.is_correct) {
                        option.classList.add('wrong-answer');
                    }
                });
                
                // Display feedback
                quizFeedback.innerHTML = `
                    <div class="alert ${data.is_correct ? 'alert-success' : 'alert-danger'}">
                        ${data.is_correct ? 'Correct!' : 'Incorrect!'} ${data.explanation}
                        <br>Current score: ${data.current_score}
                    </div>
                `;
                
                submitBtn.classList.add('d-none');
                selectedOption = null;
                
                if (data.completed) {
                    quizFeedback.innerHTML += `
                        <div class="alert alert-info mt-2">
                            Quiz completed! Final score: ${data.final_score}
                        </div>
                    `;
                } else {
                    // Load next question after a delay
                    setTimeout(() => {
                        quizQuestion.innerHTML = `<h6>Question ${data.question_number} of ${document.getElementById('quizForm').dataset.totalQuestions}</h6><p>${data.next_question}</p>`;
                        
                        quizOptions.innerHTML = '';
                        data.next_options.forEach((option, index) => {
                            const optionDiv = document.createElement('div');
                            optionDiv.className = 'quiz-option';
                            optionDiv.textContent = option;
                            optionDiv.dataset.option = option.charAt(0);
                            optionDiv.addEventListener('click', function() {
                                document.querySelectorAll('.quiz-option').forEach(opt => {
                                    opt.classList.remove('selected');
                                });
                                this.classList.add('selected');
                                selectedOption = this.dataset.option;
                                submitBtn.classList.remove('d-none');
                            });
                            quizOptions.appendChild(optionDiv);
                        });
                        
                        quizFeedback.innerHTML = '';
                        updateQuizProgress();
                    }, 2000);
                }
            } else {
                quizFeedback.innerHTML = `<div class="alert alert-danger">${data.error || 'Failed to submit answer'}</div>`;
            }
        } catch (error) {
            quizFeedback.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    }
    
    function updateQuizProgress() {
        if (!currentQuizSession) return;
        
        fetch(`/quiz_progress/${currentQuizSession}`)
            .then(response => response.json())
            .then(data => {
                const progressDiv = document.getElementById('quizProgress');
                progressDiv.innerHTML = `
                    <div class="progress">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${(data.current_question / data.total_questions) * 100}%" 
                             aria-valuenow="${data.current_question}" 
                             aria-valuemin="0" 
                             aria-valuemax="${data.total_questions}">
                            ${data.current_question}/${data.total_questions}
                        </div>
                    </div>
                    <small class="text-muted">Score: ${data.score}</small>
                `;
            });
    }
    
    // Memory map form
    document.getElementById('mapForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const topic = document.getElementById('mapTopicInput').value.trim();
        const mapResult = document.getElementById('mapResult');
        
        if (!topic) {
            mapResult.innerHTML = '<div class="alert alert-warning">Please enter a topic</div>';
            return;
        }
        
        try {
            const response = await fetch('/memory_map', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `topic=${encodeURIComponent(topic)}`
            });
            
            const data = await response.json();
            if (response.ok) {
                mapResult.querySelector('pre').textContent = data.memory_map;
            } else {
                mapResult.innerHTML = `<div class="alert alert-danger">${data.error || 'Failed to generate memory map'}</div>`;
            }
        } catch (error) {
            mapResult.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    });
});