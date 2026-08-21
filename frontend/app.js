const API_URL = 'http://127.0.0.1:8000';

// Elements
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const uploadBtn = document.getElementById('upload-btn');
const uploadStatus = document.getElementById('upload-status');
const docList = document.getElementById('doc-list');
const refreshDocsBtn = document.getElementById('refresh-docs-btn');
const clearAllBtn = document.getElementById('clear-all-btn');

const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

const chatHistory = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

const generateNotesBtn = document.getElementById('generate-notes-btn');
const notesOutput = document.getElementById('notes-output');

let selectedFiles = [];

// Tabs
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(btn.dataset.target).classList.add('active');
    });
});

// File Preview
fileInput.addEventListener('change', (e) => {
    selectedFiles = Array.from(e.target.files);
    previewContainer.innerHTML = '';
    
    selectedFiles.forEach(file => {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'preview-img';
                previewContainer.appendChild(img);
            };
            reader.readAsDataURL(file);
        } else {
            const pdfBadge = document.createElement('div');
            pdfBadge.className = 'preview-file';
            pdfBadge.innerHTML = `📄 <span>${file.name}</span>`;
            previewContainer.appendChild(pdfBadge);
        }
    });
});

// Upload and Process
uploadBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
        uploadStatus.textContent = 'Please select images or PDFs first.';
        return;
    }
    
    uploadStatus.textContent = 'Uploading...';
    uploadBtn.disabled = true;
    
    try {
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });
        
        // 1. Upload
        const uploadRes = await fetch(`${API_URL}/documents/upload`, {
            method: 'POST',
            body: formData
        });
        const uploadData = await uploadRes.json();
        
        // 2. Process each
        uploadStatus.textContent = 'Processing documents (this may take a moment)...';
        
        for (const file of uploadData.files) {
            const processRes = await fetch(`${API_URL}/documents/process?filename=${encodeURIComponent(file.filename)}`, {
                method: 'POST'
            });
            const processData = await processRes.json();
            console.log('Processed:', processData);
        }
        
        uploadStatus.textContent = 'All files processed successfully!';
        selectedFiles = [];
        fileInput.value = '';
        previewContainer.innerHTML = '';
        fetchDocuments();
        fetchTopics();
    } catch (error) {
        console.error(error);
        uploadStatus.textContent = 'Error during upload/processing.';
    } finally {
        uploadBtn.disabled = false;
    }
});

// List Documents
async function fetchDocuments() {
    try {
        const res = await fetch(`${API_URL}/documents`);
        const docs = await res.json();
        
        docList.innerHTML = '';
        if (docs.length === 0) {
            docList.innerHTML = '<li style="color:var(--text-muted)">No documents uploaded yet.</li>';
            return;
        }
        
        docs.forEach(doc => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span>${doc.filename}</span>
                <button class="delete-btn" onclick="deleteDocument('${doc.document_id}')">✕</button>
            `;
            docList.appendChild(li);
        });
    } catch (error) {
        console.error('Failed to fetch documents', error);
    }
}

// Delete Document
window.deleteDocument = async (id) => {
    try {
        await fetch(`${API_URL}/documents/${id}`, { method: 'DELETE' });
        fetchDocuments();
    } catch (error) {
        console.error('Delete failed', error);
    }
};

refreshDocsBtn.addEventListener('click', fetchDocuments);

if (clearAllBtn) {
    clearAllBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to delete ALL documents from the knowledge base? This action cannot be undone.')) {
            return;
        }
        
        try {
            uploadStatus.textContent = 'Clearing all documents...';
            const res = await fetch(`${API_URL}/documents/clear/all`, { method: 'DELETE' });
            const data = await res.json();
            uploadStatus.textContent = data.message || 'All documents cleared!';
            fetchDocuments();
        } catch (error) {
            console.error('Clear all failed', error);
            uploadStatus.textContent = 'Failed to clear documents.';
        }
    });
}

// Chat
async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    // Add user message
    addMessage(text, 'user-msg');
    chatInput.value = '';
    
    // Create assistant message container with typing indicator
    const msgId = 'msg-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-msg';
    msgDiv.id = msgId;
    
    const p = document.createElement('p');
    p.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    msgDiv.appendChild(p);
    
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    try {
        const res = await fetch(`${API_URL}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text })
        });
        
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullText = '';
        let isFirstToken = true;
        let sources = [];
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const jsonStr = line.slice(6).trim();
                        if (!jsonStr) continue;
                        const data = JSON.parse(jsonStr);
                        
                        if (data.type === 'sources') {
                            sources = data.sources || [];
                        } else if (data.type === 'token') {
                            if (isFirstToken) {
                                p.textContent = '';
                                isFirstToken = false;
                            }
                            fullText += data.content;
                            p.textContent = fullText;
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        } else if (data.type === 'done') {
                            if (sources && sources.length > 0) {
                                const sourceBox = document.createElement('div');
                                sourceBox.className = 'sources-box';
                                sourceBox.textContent = 'Sources: ' + sources.map(s => s.original_filename).join(', ');
                                msgDiv.appendChild(sourceBox);
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            }
                        }
                    } catch (err) {
                        console.error('Error parsing SSE event:', err);
                    }
                }
            }
        }
    } catch (error) {
        console.error(error);
        p.textContent = 'Sorry, an error occurred while generating response.';
    }
}

function addMessage(text, className, sources = null, isHTML = false) {
    const id = 'msg-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${className}`;
    msgDiv.id = id;
    
    const p = document.createElement('p');
    if (isHTML) {
        p.innerHTML = text;
    } else {
        p.textContent = text;
    }
    msgDiv.appendChild(p);
    
    if (sources && sources.length > 0) {
        const sourceBox = document.createElement('div');
        sourceBox.className = 'sources-box';
        sourceBox.textContent = 'Sources: ' + sources.map(s => s.original_filename).join(', ');
        msgDiv.appendChild(sourceBox);
    }
    
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return id;
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Notes
generateNotesBtn.addEventListener('click', async () => {
    generateNotesBtn.disabled = true;
    generateNotesBtn.textContent = 'Generating...';
    notesOutput.textContent = 'Reading textbook and generating notes. Please wait...';
    
    try {
        const res = await fetch(`${API_URL}/notes`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}) // Empty body fetches general notes
        });
        const data = await res.json();
        notesOutput.textContent = data.notes;
    } catch (error) {
        notesOutput.textContent = 'Failed to generate notes.';
        console.error(error);
    } finally {
        generateNotesBtn.disabled = false;
        generateNotesBtn.textContent = 'Generate Notes';
    }
});

// Init
fetchDocuments();
fetchTopics();

// Quiz & Topics State
const topicChipsContainer = document.getElementById('topic-chips');
const generateQuizBtn = document.getElementById('generate-quiz-btn');
const quizPlayer = document.getElementById('quiz-player');

let selectedTopicTitle = "";
let currentQuizData = null;
let currentQuestionIndex = 0;
let userScore = 0;

// Fetch Classified Topics
async function fetchTopics() {
    if (!topicChipsContainer) return;
    try {
        const res = await fetch(`${API_URL}/quiz/topics`);
        const data = await res.json();
        
        topicChipsContainer.innerHTML = '<button class="chip active" data-topic="">Overall Quiz</button>';
        
        if (data.topics && data.topics.length > 0) {
            data.topics.forEach(t => {
                const btn = document.createElement('button');
                btn.className = 'chip';
                btn.dataset.topic = t.title;
                btn.textContent = t.title;
                topicChipsContainer.appendChild(btn);
            });
        }
        
        // Add click listener to chips
        const chips = topicChipsContainer.querySelectorAll('.chip');
        chips.forEach(chip => {
            chip.addEventListener('click', () => {
                chips.forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                selectedTopicTitle = chip.dataset.topic;
            });
        });
    } catch (err) {
        console.error('Failed to fetch topics', err);
    }
}

// Generate Quiz
if (generateQuizBtn) {
    generateQuizBtn.addEventListener('click', async () => {
        const countSelect = document.getElementById('quiz-count-select');
        const numQ = countSelect ? (parseInt(countSelect.value) || 5) : 5;

        generateQuizBtn.disabled = true;
        generateQuizBtn.textContent = 'Generating Quiz...';
        quizPlayer.innerHTML = `<div class="quiz-placeholder"><div class="typing-indicator"><span></span><span></span><span></span></div><p style="margin-top:12px">Generating ${numQ} questions from textbook context...</p></div>`;
        
        try {
            const res = await fetch(`${API_URL}/quiz/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic: selectedTopicTitle, num_questions: numQ })
            });
            currentQuizData = await res.json();
            currentQuestionIndex = 0;
            userScore = 0;
            
            renderQuestion();
        } catch (err) {
            console.error('Quiz generation failed', err);
            quizPlayer.innerHTML = '<div class="quiz-placeholder" style="color:#ef4444">Failed to generate quiz. Please try again.</div>';
        } finally {
            generateQuizBtn.disabled = false;
            generateQuizBtn.textContent = 'Start Interactive Quiz';
        }
    });
}

function renderQuestion() {
    if (!currentQuizData || !currentQuizData.questions || currentQuizData.questions.length === 0) {
        quizPlayer.innerHTML = '<div class="quiz-placeholder">No questions generated. Please upload textbook materials first!</div>';
        return;
    }
    
    if (currentQuestionIndex >= currentQuizData.questions.length) {
        renderScoreCard();
        return;
    }
    
    const q = currentQuizData.questions[currentQuestionIndex];
    const totalQ = currentQuizData.questions.length;
    
    let html = `
        <div class="quiz-card">
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.9rem; color:var(--text-muted)">
                <span>Topic: <strong>${currentQuizData.topic || 'Overall'}</strong></span>
                <span>Question ${currentQuestionIndex + 1} of ${totalQ}</span>
            </div>
            
            <div class="quiz-question">${q.question}</div>
            
            <div class="options-grid" id="options-grid">
    `;
    
    q.options.forEach(opt => {
        const optionKey = opt.trim().substring(0, 1).toUpperCase();
        html += `<button class="option-btn" data-key="${optionKey}">${opt}</button>`;
    });
    
    html += `
            </div>
            
            <div id="explanation-box" class="quiz-explanation" style="display:none;"></div>
            
            <div style="display:flex; justify-content:flex-end;">
                <button id="next-q-btn" class="btn primary-btn" style="width:auto; display:none;">Next Question ➔</button>
            </div>
        </div>
    `;
    
    quizPlayer.innerHTML = html;
    
    const optionBtns = quizPlayer.querySelectorAll('.option-btn');
    const explanationBox = document.getElementById('explanation-box');
    const nextBtn = document.getElementById('next-q-btn');
    
    optionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const selectedKey = btn.dataset.key;
            const correctKey = q.correct.trim().toUpperCase();
            
            optionBtns.forEach(b => b.disabled = true);
            
            if (selectedKey === correctKey) {
                btn.classList.add('correct');
                userScore++;
                explanationBox.innerHTML = `✅ <strong>Correct!</strong> ${q.explanation || ''}`;
            } else {
                btn.classList.add('incorrect');
                optionBtns.forEach(b => {
                    if (b.dataset.key === correctKey) b.classList.add('correct');
                });
                explanationBox.innerHTML = `❌ <strong>Incorrect.</strong> ${q.explanation || ''}`;
            }
            
            explanationBox.style.display = 'block';
            nextBtn.style.display = 'inline-block';
        });
    });
    
    nextBtn.addEventListener('click', () => {
        currentQuestionIndex++;
        renderQuestion();
    });
}

function renderScoreCard() {
    const totalQ = currentQuizData.questions.length;
    const percentage = Math.round((userScore / totalQ) * 100);
    
    quizPlayer.innerHTML = `
        <div class="score-card">
            <h2>🎉 Quiz Completed!</h2>
            <div class="score-badge">${userScore} / ${totalQ}</div>
            <p style="color:var(--text-muted)">Score: ${percentage}%</p>
            <button id="retake-quiz-btn" class="btn primary-btn" style="max-width:200px;">Take Another Quiz</button>
        </div>
    `;
    
    document.getElementById('retake-quiz-btn').addEventListener('click', () => {
        quizPlayer.innerHTML = '<div class="quiz-placeholder"><p>Select a topic above and click "Generate 5-Question Quiz" to start testing your knowledge!</p></div>';
    });
}
