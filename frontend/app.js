/**
 * TuneKit Frontend
 * ================
 * New flow: Upload → Questions → Recommendation → Training
 */

const API_URL = 'http://localhost:8000';

// State
let sessionId = null;
let currentStep = 1;
let jobId = null;
let statusPollInterval = null;

// User selections
let selectedTask = null;
let selectedDeployment = null;

// Analysis data
let analysisData = null;
let recommendationData = null;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileRows = document.getElementById('fileRows');
const fileSize = document.getElementById('fileSize');
const removeFile = document.getElementById('removeFile');
const continueBtn = document.getElementById('continueBtn');
const validationStatus = document.getElementById('validationStatus');
const errorBanner = document.getElementById('errorBanner');
const errorText = document.getElementById('errorText');
const errorClose = document.getElementById('errorClose');

// ============================================================================
// STEP NAVIGATION
// ============================================================================

function setStep(step) {
    currentStep = step;
    
    document.querySelectorAll('.step-content').forEach((el, i) => {
        el.classList.remove('active');
        if (i + 1 === step) {
            el.classList.add('active');
        }
    });
    
    // Update sidebar navigation
    document.querySelectorAll('.nav-step').forEach((el, i) => {
        el.classList.remove('active', 'completed');
        if (i + 1 < step) {
            el.classList.add('completed');
        } else if (i + 1 === step) {
            el.classList.add('active');
        }
    });
    
    // Scroll to top smoothly
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// QUESTION FADE TRANSITIONS
// ============================================================================

function showQuestion(questionId) {
    const questionWrapper = document.getElementById(questionId);
    const allQuestions = document.querySelectorAll('.question-wrapper');
    
    // Hide all questions
    allQuestions.forEach(q => {
        q.classList.remove('active');
    });
    
    // Show the selected question with fade
    setTimeout(() => {
        questionWrapper.classList.add('active');
    }, 50);
}

function hideQuestion(questionId) {
    const questionWrapper = document.getElementById(questionId);
    questionWrapper.classList.remove('active');
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
    errorText.textContent = message;
    errorBanner.classList.remove('hidden');
}

function hideError() {
    errorBanner.classList.add('hidden');
}

errorClose?.addEventListener('click', hideError);

// ============================================================================
// FILE UPLOAD
// ============================================================================

dropZone?.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone?.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone?.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
});

dropZone?.addEventListener('click', () => {
    fileInput.click();
});

fileInput?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelect(file);
});

removeFile?.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFileSelect(file) {
    hideError();
    
    // Validate file type
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    
    if (ext !== '.jsonl') {
        showError('Please upload a JSONL file. Other formats are not supported.');
        return;
    }
    
    uploadFile(file);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state
    continueBtn.disabled = true;
    continueBtn.innerHTML = 'Uploading... <span class="btn-loader"></span>';
    
    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            let errorMsg = `Upload failed`;
            try {
                const error = await response.json();
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                const text = await response.text();
                errorMsg = text || errorMsg;
            }
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        sessionId = data.session_id;
        
        // Debug: log the response
        console.log('Upload response:', data);
        console.log('data.rows:', data.rows, 'type:', typeof data.rows);
        console.log('data.stats:', data.stats);
        
        // Get number of examples - try multiple sources
        let numExamples = 0;
        
        // First try data.rows
        if (data.rows !== undefined && data.rows !== null) {
            numExamples = Number(data.rows) || 0;
        }
        
        // Fallback to stats.total_examples
        if (numExamples === 0 && data.stats && data.stats.total_examples) {
            numExamples = Number(data.stats.total_examples) || 0;
        }
        
        console.log('Number of examples (final):', numExamples);
        
        // Show file info
        fileName.textContent = file.name;
        if (numExamples > 0) {
            fileRows.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> ${numExamples.toLocaleString()} examples`;
        } else {
            fileRows.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> — examples`;
            console.warn('No examples count found in response');
        }
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.classList.remove('hidden');
        dropZone.style.display = 'none';
        
        // Show validation status
        if (data.stats) {
            showValidationStatus(data.stats);
        }
        
        // Store for later
        analysisData = data;
        
        // Enable continue
        continueBtn.disabled = false;
        resetContinueBtn();
        
    } catch (error) {
        console.error('Upload error:', error);
        resetContinueBtn();
        
        if (error.message === 'Failed to fetch' || error.message.includes('NetworkError')) {
            showError('Cannot connect to server. Is the backend running on http://localhost:8000?');
        } else {
            showError(error.message);
        }
    }
}

function resetContinueBtn() {
    continueBtn.innerHTML = `Continue <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>`;
}

function showValidationStatus(stats) {
    const validationIcon = document.getElementById('validationIcon');
    const validationTitle = document.getElementById('validationTitle');
    const validationMeta = document.getElementById('validationMeta');
    
    // Simple validation status - detailed info goes to summary card
    validationIcon.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    validationTitle.textContent = 'Format Validated';
    validationMeta.textContent = 'Dataset ready for analysis';
    
    // Show warnings if any
    if (stats.warnings && stats.warnings.length > 0) {
        stats.warnings.forEach(warning => {
            console.warn('Dataset warning:', warning);
        });
    }
    
    validationStatus.classList.remove('hidden');
    
    // Show dataset summary with all the details
    showDatasetSummary(stats);
}

function showDatasetSummary(stats) {
    const summaryCard = document.getElementById('datasetSummaryCard');
    const statsGrid = document.getElementById('summaryStatsGrid');
    
    if (!stats || !summaryCard) return;
    
    // Build clean stats table
    let html = '';
    
    // Primary metrics row
    html += '<div class="stats-primary">';
    html += `<div class="primary-stat">
        <span class="primary-value">${(stats.total_tokens || 0).toLocaleString()}</span>
        <span class="primary-label">Total Tokens</span>
    </div>`;
    html += `<div class="primary-stat">
        <span class="primary-value">${stats.avg_tokens_per_example || 0}</span>
        <span class="primary-label">Avg Tokens/Example</span>
    </div>`;
    html += `<div class="primary-stat">
        <span class="primary-value">~${stats.est_training_time_min || 3} min</span>
        <span class="primary-label">Est. Training Time</span>
    </div>`;
    html += `<div class="primary-stat">
        <span class="primary-value">$${stats.est_cost_usd || '0.15'}</span>
        <span class="primary-label">Est. Cost</span>
    </div>`;
    html += '</div>';
    
    // Secondary details
    html += '<div class="stats-details">';
    
    // Conversation type
    const singlePct = stats.single_turn_pct || 0;
    const multiPct = stats.multi_turn_pct || 0;
    html += `<div class="detail-row">
        <span class="detail-label">Conversation Type</span>
        <span class="detail-value">${singlePct}% single-turn, ${multiPct}% multi-turn</span>
    </div>`;
    
    // System prompts
    const sysPct = stats.system_prompt_pct || 0;
    html += `<div class="detail-row">
        <span class="detail-label">System Prompts</span>
        <span class="detail-value">${sysPct > 0 ? sysPct + '% of examples' : 'None'}</span>
    </div>`;
    
    // Response length
    const avgOutput = stats.avg_output_chars || 0;
    const outputTokens = Math.round(avgOutput / 4);
    html += `<div class="detail-row">
        <span class="detail-label">Avg Response Length</span>
        <span class="detail-value">~${outputTokens} tokens</span>
    </div>`;
    
    // Quality indicator
    const quality = stats.quality || 'good';
    const qualityLabel = quality === 'excellent' ? 'Excellent' : quality === 'good' ? 'Good' : 'Minimal';
    const qualityClass = quality === 'excellent' ? 'success' : quality === 'good' ? 'accent' : 'warning';
    html += `<div class="detail-row">
        <span class="detail-label">Dataset Quality</span>
        <span class="detail-value quality-badge ${qualityClass}">${qualityLabel}</span>
    </div>`;
    
    html += '</div>';
    
    // Warnings if any
    if (stats.warnings && stats.warnings.length > 0) {
        html += '<div class="stats-warnings">';
        stats.warnings.forEach(w => {
            html += `<div class="warning-item">⚠️ ${w}</div>`;
        });
        html += '</div>';
    }
    
    statsGrid.innerHTML = html;
    summaryCard.classList.remove('hidden');
}

function clearFile() {
    sessionId = null;
    analysisData = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    validationStatus.classList.add('hidden');
    const summaryCard = document.getElementById('datasetSummaryCard');
    if (summaryCard) {
        summaryCard.classList.add('hidden');
        document.getElementById('summaryStatsGrid').innerHTML = '';
    }
    dropZone.style.display = '';
    continueBtn.disabled = true;
    hideError();
}

// ============================================================================
// STEP 1 → STEP 2: Continue to Questions
// ============================================================================

continueBtn?.addEventListener('click', async () => {
    if (!sessionId) return;
    
    // Move to questions step
    setStep(2);
    
    // Reset selections
    selectedTask = null;
    selectedDeployment = null;
    
    // Show task question with fade
    setTimeout(() => {
        showQuestion('questionTask');
        hideQuestion('questionDeployment');
    }, 100);
    
    // Remove any previous selections
    document.querySelectorAll('.option-card').forEach(card => {
        card.classList.remove('selected');
    });
});

// ============================================================================
// STEP 2: Task & Deployment Selection
// ============================================================================

// Task selection
document.querySelectorAll('.option-card[data-task]').forEach(card => {
    card.addEventListener('click', () => {
        // Remove previous selection
        document.querySelectorAll('.option-card[data-task]').forEach(c => c.classList.remove('selected'));
        
        // Select this one
        card.classList.add('selected');
        selectedTask = card.dataset.task;
        
        // Fade out task question, fade in deployment question
        hideQuestion('questionTask');
        setTimeout(() => {
            showQuestion('questionDeployment');
        }, 300);
    });
});

// Deployment selection
document.querySelectorAll('.option-card[data-deployment]').forEach(card => {
    card.addEventListener('click', async () => {
        // Remove previous selection
        document.querySelectorAll('.option-card[data-deployment]').forEach(c => c.classList.remove('selected'));
        
        // Select this one
        card.classList.add('selected');
        selectedDeployment = card.dataset.deployment;
        
        // Fade out deployment question, move to recommendation
        hideQuestion('questionDeployment');
        setTimeout(() => {
            setStep(3);
            getRecommendation();
        }, 300);
    });
});

// Back buttons
document.getElementById('backToUpload')?.addEventListener('click', () => {
    setStep(1);
});

document.getElementById('backToQuestions')?.addEventListener('click', () => {
    setStep(2);
    // Show deployment question (user was already past task)
    setTimeout(() => {
        hideQuestion('questionTask');
        showQuestion('questionDeployment');
    }, 100);
});

// ============================================================================
// STEP 3: Get Recommendation
// ============================================================================

async function getRecommendation() {
    const recLoading = document.getElementById('recLoading');
    const recContent = document.getElementById('recContent');
    
    recLoading.classList.remove('hidden');
    recContent.classList.add('hidden');
    
    try {
        const response = await fetch(`${API_URL}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_task: selectedTask,
                deployment_target: selectedDeployment
            })
        });
        
        if (!response.ok) {
            let errorMsg = 'Failed to get recommendation';
            try {
                const error = await response.json();
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = await response.text() || errorMsg;
            }
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        recommendationData = data;
        
        displayRecommendation(data);
        
        recLoading.classList.add('hidden');
        recContent.classList.remove('hidden');
        
    } catch (error) {
        console.error('Recommendation error:', error);
        recLoading.classList.add('hidden');
        showError(error.message);
    }
}

function displayRecommendation(data) {
    const rec = data.recommendation?.primary_recommendation;
    
    if (!rec) {
        showError('No recommendation available');
        return;
    }
    
    // Model name and size
    document.getElementById('recModelName').textContent = rec.model_name;
    document.getElementById('recModelSize').textContent = rec.size;
    
    // Score with animation
    const scorePercent = Math.round((rec.score || 0) * 100);
    document.getElementById('recScore').textContent = scorePercent + '%';
    
    // Animate score ring
    const scoreCircle = document.getElementById('scoreCircle');
    setTimeout(() => {
        scoreCircle.style.strokeDasharray = `${scorePercent}, 100`;
    }, 100);
    
    // Reasons
    const reasonsContainer = document.getElementById('recReasons');
    reasonsContainer.innerHTML = '';
    (rec.reasons || []).forEach(reason => {
        const div = document.createElement('div');
        div.className = 'rec-reason';
        div.innerHTML = `<span class="reason-icon">✓</span><span>${reason}</span>`;
        reasonsContainer.appendChild(div);
    });
    
    // Stats
    document.getElementById('recTime').textContent = `~${rec.training_time_min || 3} min`;
    document.getElementById('recCost').textContent = `$${(rec.cost_usd || 0.18).toFixed(2)}`;
    document.getElementById('recAccuracy').textContent = `~${rec.estimated_accuracy || 87}%`;
    
    // Alternatives
    const altGrid = document.getElementById('alternativesGrid');
    altGrid.innerHTML = '';
    
    const alternatives = data.recommendation?.alternatives || [];
    alternatives.slice(0, 3).forEach(alt => {
        const div = document.createElement('div');
        div.className = 'alt-card';
        div.innerHTML = `
            <div class="alt-header">
                <span class="alt-name">${alt.model_name}</span>
                <span class="alt-score">${Math.round((alt.score || 0) * 100)}%</span>
            </div>
            <div class="alt-reasons">
                ${(alt.reasons || []).slice(0, 2).map(r => `<span class="alt-reason">${r}</span>`).join('')}
            </div>
        `;
        altGrid.appendChild(div);
    });
    
    // Dataset summary
    displayDatasetSummary(data.analysis);
}

function displayDatasetSummary(analysis) {
    const summaryGrid = document.getElementById('datasetSummary');
    summaryGrid.innerHTML = '';
    
    if (!analysis) return;
    
    const stats = analysis.stats || {};
    const chars = analysis.conversation_characteristics || {};
    
    const items = [
        { label: 'Examples', value: (stats.total_examples || 0).toLocaleString() },
        { label: 'Messages', value: (stats.total_messages || 0).toLocaleString() },
        { label: 'Avg/Example', value: stats.avg_messages_per_example || '—' },
        { label: 'Multi-turn', value: chars.is_multi_turn ? 'Yes' : 'No' },
        { label: 'System Prompts', value: chars.has_system_prompts ? 'Yes' : 'No' },
        { label: 'Classification', value: chars.looks_like_classification ? 'Likely' : 'No' },
    ];
    
    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'summary-item';
        div.innerHTML = `
            <span class="summary-label">${item.label}</span>
            <span class="summary-value">${item.value}</span>
        `;
        summaryGrid.appendChild(div);
    });
}

// ============================================================================
// STEP 3 → STEP 4: Training
// ============================================================================

document.getElementById('startTrainingBtn')?.addEventListener('click', () => {
    setStep(4);
    
    // Show training options
    document.getElementById('trainingOptions').classList.remove('hidden');
    document.getElementById('trainingProgress').classList.add('hidden');
    document.getElementById('trainingComplete').classList.add('hidden');
    document.getElementById('downloadPackage').classList.add('hidden');
});

document.getElementById('backToRec')?.addEventListener('click', () => {
    setStep(3);
});

// ============================================================================
// CLOUD TRAINING
// ============================================================================

const trainCloudBtn = document.getElementById('trainCloudBtn');
trainCloudBtn?.addEventListener('click', startCloudTraining);

async function startCloudTraining() {
    if (!sessionId || !recommendationData) {
        showError('Missing session data. Please start over.');
        return;
    }
    
    // Show loading
    trainCloudBtn.disabled = true;
    trainCloudBtn.querySelector('.btn-text').textContent = 'Starting...';
    trainCloudBtn.querySelector('.btn-loader').classList.remove('hidden');
    
    try {
        // First, create the plan
        const planResponse = await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_task: selectedTask,
                deployment_target: selectedDeployment
            })
        });
        
        if (!planResponse.ok) {
            throw new Error('Failed to create training plan');
        }
        
        // Start training
        const trainResponse = await fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        if (!trainResponse.ok) {
            let errorMsg = 'Training failed to start';
            try {
                const error = await trainResponse.json();
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = await trainResponse.text() || errorMsg;
            }
            throw new Error(errorMsg);
        }
        
        const data = await trainResponse.json();
        jobId = data.job_id;
        
        // Switch to progress view
        document.getElementById('trainingOptions').classList.add('hidden');
        document.getElementById('trainingProgress').classList.remove('hidden');
        
        // Update job info
        document.getElementById('jobIdDisplay').textContent = jobId;
        document.getElementById('startTimeDisplay').textContent = new Date().toLocaleTimeString();
        
        // Start polling
        startStatusPolling();
        
    } catch (error) {
        console.error('Training error:', error);
        showError(error.message);
    } finally {
        trainCloudBtn.disabled = false;
        trainCloudBtn.querySelector('.btn-text').textContent = 'Train on Cloud';
        trainCloudBtn.querySelector('.btn-loader').classList.add('hidden');
    }
}

function startStatusPolling() {
    if (statusPollInterval) clearInterval(statusPollInterval);
    
    statusPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/train/status/${jobId}`);
            if (!response.ok) return;
            
            const data = await response.json();
            updateTrainingStatus(data);
            
            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(statusPollInterval);
            }
        } catch (error) {
            console.error('Status poll error:', error);
        }
    }, 3000);
}

function updateTrainingStatus(data) {
    const statusDisplay = document.getElementById('statusDisplay');
    const logOutput = document.getElementById('logOutput');
    
    statusDisplay.textContent = data.status;
    
    if (data.status === 'completed') {
        document.getElementById('trainingProgress').classList.add('hidden');
        document.getElementById('trainingComplete').classList.remove('hidden');
        
        // Display metrics if available
        if (data.base_metrics && data.finetuned_metrics) {
            displayMetrics(data);
        }
        
        // Setup download
        if (data.model_url) {
            document.getElementById('downloadModelBtn').href = data.model_url;
        }
    } else if (data.status === 'failed') {
        statusDisplay.textContent = 'Failed';
        statusDisplay.style.color = 'var(--error)';
        
        const logLine = document.createElement('div');
        logLine.className = 'log-line log-error';
        logLine.textContent = data.error || 'Training failed';
        logOutput.appendChild(logLine);
    } else {
        // Add log line
        if (data.message) {
            const logLine = document.createElement('div');
            logLine.className = 'log-line';
            logLine.textContent = data.message;
            logOutput.appendChild(logLine);
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    }
}

function displayMetrics(data) {
    const baseMetrics = document.getElementById('baseMetrics');
    const tunedMetrics = document.getElementById('tunedMetrics');
    
    baseMetrics.innerHTML = '';
    tunedMetrics.innerHTML = '';
    
    // Display base metrics
    Object.entries(data.base_metrics || {}).forEach(([key, value]) => {
        const div = document.createElement('div');
        div.className = 'card-metric';
        div.innerHTML = `
            <span class="card-metric-label">${formatMetricName(key)}</span>
            <span class="card-metric-value">${formatMetricValue(value)}</span>
        `;
        baseMetrics.appendChild(div);
    });
    
    // Display fine-tuned metrics
    Object.entries(data.finetuned_metrics || {}).forEach(([key, value]) => {
        const div = document.createElement('div');
        div.className = 'card-metric';
        div.innerHTML = `
            <span class="card-metric-label">${formatMetricName(key)}</span>
            <span class="card-metric-value">${formatMetricValue(value)}</span>
        `;
        tunedMetrics.appendChild(div);
    });
    
    // Set base model name
    if (data.base_model) {
        document.getElementById('baseModelName').textContent = data.base_model;
    }
}

function formatMetricName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatMetricValue(value) {
    if (typeof value === 'number') {
        if (value < 1 && value > 0) {
            return (value * 100).toFixed(1) + '%';
        }
        return value.toFixed(2);
    }
    return value;
}

// ============================================================================
// LOCAL PACKAGE GENERATION
// ============================================================================

const generateBtn = document.getElementById('generateBtn');
generateBtn?.addEventListener('click', generatePackage);

async function generatePackage() {
    if (!sessionId) {
        showError('No session. Please upload a file first.');
        return;
    }
    
    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').textContent = 'Generating...';
    generateBtn.querySelector('.btn-loader').classList.remove('hidden');
    
    try {
        // Create plan first
        const planResponse = await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_task: selectedTask,
                deployment_target: selectedDeployment
            })
        });
        
        if (!planResponse.ok) {
            throw new Error('Failed to create training plan');
        }
        
        // Generate package
        const response = await fetch(`${API_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        if (!response.ok) {
            let errorMsg = 'Package generation failed';
            try {
                const error = await response.json();
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = await response.text() || errorMsg;
            }
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        
        // Show download section
        document.getElementById('trainingOptions').classList.add('hidden');
        document.getElementById('downloadPackage').classList.remove('hidden');
        
        // Setup download link
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.href = data.download_url;
        
    } catch (error) {
        console.error('Generate error:', error);
        showError(error.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').textContent = 'Download Package';
        generateBtn.querySelector('.btn-loader').classList.add('hidden');
    }
}

// ============================================================================
// START OVER
// ============================================================================

document.getElementById('startOver')?.addEventListener('click', () => {
    // Clear all state
    sessionId = null;
    jobId = null;
    selectedTask = null;
    selectedDeployment = null;
    analysisData = null;
    recommendationData = null;
    
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }
    
    // Reset file input
    clearFile();
    
    // Reset step 4 views
    document.getElementById('trainingOptions')?.classList.remove('hidden');
    document.getElementById('trainingProgress')?.classList.add('hidden');
    document.getElementById('trainingComplete')?.classList.add('hidden');
    document.getElementById('downloadPackage')?.classList.add('hidden');
    
    // Clear log
    const logOutput = document.getElementById('logOutput');
    if (logOutput) {
        logOutput.innerHTML = '<div class="log-line">Initializing training environment...</div>';
    }
    
    // Go to step 1
    setStep(1);
});

// ============================================================================
// INITIALIZATION
// ============================================================================

// Check if backend is running
async function checkBackend() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✓ Backend connected');
        }
    } catch (error) {
        console.warn('⚠️ Backend not reachable at', API_URL);
    }
}

checkBackend();
