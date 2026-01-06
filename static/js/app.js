// Main JavaScript for the Chatbot Application

// Main section tabs (Chat vs Resume)
const mainTabs = document.querySelectorAll('.main-tab');
const mainSections = document.querySelectorAll('.main-section');

mainTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const target = tab.getAttribute('data-section');

        // Update active tab
        mainTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Show only the selected main section
        mainSections.forEach(section => {
            if (section.getAttribute('data-section') === target) {
                section.style.display = '';
            } else {
                section.style.display = 'none';
            }
        });
    });
});

// Chat functionality
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');

// Resume upload functionality
const fileInput = document.getElementById('resume-file');
const analyzeFileBtn = document.getElementById('analyze-file-btn');
const analyzeTextBtn = document.getElementById('analyze-text-btn');
const resumeTextarea = document.getElementById('resume-text');
const resultsSection = document.getElementById('results-section');
const resultsContent = document.getElementById('results-content');

// Analytics tab elements
const analyticsEmpty = document.getElementById('analytics-empty');
const analyticsContent = document.getElementById('analytics-content');

let lastAnalysis = null;
let lastVisualizations = null;

// Tab switching
const uploadTabs = document.querySelectorAll('.upload-tab');
const uploadContents = document.querySelectorAll('.upload-content');

uploadTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.getAttribute('data-tab');
        
        // Update active tab
        uploadTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update active content
        uploadContents.forEach(content => {
            content.classList.remove('active');
            if (content.id === `${targetTab}-upload`) {
                content.classList.add('active');
            }
        });
    });
});

// Chat functionality
function addMessage(text, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `<strong>${isUser ? 'You' : 'Bot'}:</strong> ${text}`;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Add user message
    addMessage(message, true);
    chatInput.value = '';
    
    // Show loading
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message';
    loadingDiv.innerHTML = '<div class="message-content"><strong>Bot:</strong> Thinking...</div>';
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Send to backend
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading message
        chatMessages.removeChild(loadingDiv);
        
        // Add bot response
        addMessage(data.response || 'Sorry, I could not process your request.');
        
        // Show intent info if available
        if (data.intent) {
            const intentInfo = document.createElement('div');
            intentInfo.className = 'message bot-message';
            intentInfo.innerHTML = `<div class="message-content" style="font-size: 0.9em; opacity: 0.8;">
                <em>Intent: ${data.intent} (${(data.confidence * 100).toFixed(1)}% confidence)</em>
            </div>`;
            chatMessages.appendChild(intentInfo);
        }
    })
    .catch(error => {
        chatMessages.removeChild(loadingDiv);
        addMessage('Sorry, an error occurred. Please try again.');
        console.error('Error:', error);
    });
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// File upload handling
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        analyzeFileBtn.disabled = false;
        analyzeFileBtn.textContent = `Analyze: ${e.target.files[0].name}`;
    } else {
        analyzeFileBtn.disabled = true;
        analyzeFileBtn.textContent = 'Analyze Resume';
    }
});

analyzeFileBtn.addEventListener('click', () => {
    const file = fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    analyzeResume(formData, 'file');
});

analyzeTextBtn.addEventListener('click', () => {
    const text = resumeTextarea.value.trim();
    if (!text || text.length < 50) {
        alert('Please provide at least 50 characters of resume text.');
        return;
    }
    
    analyzeResume({ text: text }, 'text');
});

function analyzeResume(data, type) {
    if (analyticsEmpty) {
        analyticsEmpty.style.display = 'block';
        analyticsEmpty.innerHTML = '<div class="loading">Analyzing resume... Please wait.</div>';
    }
    
    const url = type === 'file' ? '/upload_resume' : '/analyze_text';
    const options = {
        method: 'POST',
        body: type === 'file' ? data : JSON.stringify(data),
        headers: type === 'text' ? { 'Content-Type': 'application/json' } : {}
    };
    
    fetch(url, options)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                if (analyticsEmpty) {
                    analyticsEmpty.style.display = 'block';
                    analyticsEmpty.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
                return;
            }

            lastAnalysis = data.analysis;
            lastVisualizations = data.visualizations || {};
            
            // Only use the Visual Analytics tab for displaying results
            updateAnalyticsTab(lastAnalysis, lastVisualizations);
        })
        .catch(error => {
            if (analyticsEmpty) {
                analyticsEmpty.style.display = 'block';
                analyticsEmpty.innerHTML = `<div class="error">Error analyzing resume: ${error.message}</div>`;
            }
            console.error('Error:', error);
        });
}

function displayResults(analysis, visualizations) {
    // Kept for compatibility; results are now shown only in the Visual Analytics tab.
    // This function intentionally does nothing so that the “Analysis Results” block stays hidden.
    return;
}

// Populate the Analytics tab with charts + explanations
function updateAnalyticsTab(analysis, visualizations) {
    if (!analyticsEmpty || !analyticsContent) return;

    if (!analysis || !visualizations || Object.keys(visualizations).length === 0) {
        analyticsEmpty.style.display = 'block';
        analyticsContent.style.display = 'none';
        analyticsContent.innerHTML = '';
        return;
    }

    analyticsEmpty.style.display = 'none';
    analyticsContent.style.display = 'grid';

    let html = '';

    // High-level summary cards (score, domain, roles)
    const score = analysis.readiness_score?.toFixed
        ? analysis.readiness_score.toFixed(1)
        : analysis.readiness_score;

    html += `
        <div class="analytics-card">
            <div class="analytics-card-header">Overall Readiness Score</div>
            <div class="analytics-card-body">
                <div class="score-display" style="margin: 0 0 12px 0;">${score}/100</div>
                <p class="analytics-explanation">
                    This score combines skills, domain match, and resume length/quality.
                    Aim for <strong>70+</strong> for strong placement readiness; anything below
                    that is a signal to add projects, strengthen skills, or refine your resume wording.
                </p>
            </div>
        </div>
    `;

    html += `
        <div class="analytics-card">
            <div class="analytics-card-header">Domain & Recommended Roles</div>
            <div class="analytics-card-body">
                <p class="analytics-explanation" style="margin-bottom: 8px;">
                    Detected primary domain: <strong>${analysis.domain}</strong><br/>
                    Confidence: <strong>${(analysis.domain_confidence * 100).toFixed(1)}%</strong>
                </p>
                <p class="analytics-explanation">
                    Suggested roles that best fit your current profile:
                </p>
                <ul style="list-style:none; padding-left:0; margin-top:6px;">
                    ${(analysis.recommended_roles || []).map(
                        role => `<li>• ${role}</li>`
                    ).join('')}
                </ul>
            </div>
        </div>
    `;

    // Skill frequency chart
    if (visualizations.skill_frequency) {
        const totalSkills = (analysis.technical_skills?.length || 0) + (analysis.soft_skills?.length || 0);
        html += `
            <div class="analytics-card">
                <div class="analytics-card-header">Skill Frequency Overview</div>
                <div class="analytics-card-body">
                    <img src="${visualizations.skill_frequency}" alt="Skill frequency chart" />
                    <p class="analytics-explanation">
                        What it shows: which skills you mention most often.<br/>
                        How to read it: taller bars = more emphasis in your resume.<br/>
                        Why it matters: recruiters skim for repeated, relevant skills.<br/>
                        Your count: <strong>${totalSkills}</strong> unique skills (technical + soft).
                    </p>
                </div>
            </div>
        `;
    }

    // Skill gaps chart
    if (visualizations.skill_gaps && analysis.skill_gaps) {
        const missing = analysis.skill_gaps.length;
        html += `
            <div class="analytics-card">
                <div class="analytics-card-header">Skill Gap Analysis for ${analysis.domain}</div>
                <div class="analytics-card-body">
                    <img src="${visualizations.skill_gaps}" alt="Skill gaps chart" />
                    <p class="analytics-explanation">
                        What it shows: required vs. found vs. missing skills for <strong>${analysis.domain}</strong>.<br/>
                        How to read it: the red bar highlights missing items—add them to projects or upskill quickly.<br/>
                        Why it matters: gaps reduce shortlist chances; closing them lifts your readiness score.<br/>
                        Missing now: <strong>${missing}</strong> key skill${missing === 1 ? '' : 's'}.
                    </p>
                </div>
            </div>
        `;
    }

    // Word cloud
    if (visualizations.wordcloud) {
        html += `
            <div class="analytics-card">
                <div class="analytics-card-header">Resume Word Cloud</div>
                <div class="analytics-card-body">
                    <img src="${visualizations.wordcloud}" alt="Resume word cloud" />
                    <p class="analytics-explanation">
                        What it shows: the most frequent words in your resume (bigger = more mentions).<br/>
                        How to read it: core tech/role keywords should stand out (e.g., React, Django, Data Science).<br/>
                        Why it matters: recruiters scan for obvious alignment; make sure your target role keywords are bold here.<br/>
                        Tip: if an important skill is tiny/missing, add a project bullet or measurable impact line featuring it.
                    </p>
                </div>
            </div>
        `;
    }

    analyticsContent.innerHTML = html;

    // Automatically switch to the Analytics main tab so the user sees results
    const analyticsTab = Array.from(mainTabs || []).find(
        t => t.getAttribute('data-section') === 'analytics'
    );
    const analyticsSection = Array.from(mainSections || []).find(
        s => s.getAttribute('data-section') === 'analytics'
    );
    if (analyticsTab && analyticsSection) {
        mainTabs.forEach(t => t.classList.remove('active'));
        analyticsTab.classList.add('active');

        mainSections.forEach(section => {
            if (section === analyticsSection) {
                section.style.display = '';
            } else {
                section.style.display = 'none';
            }
        });

        // Scroll analytics section into view
        analyticsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

