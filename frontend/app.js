/**
 * TuneKit Frontend
 * ================
 * Handles API interactions and UI updates
 */

const API_URL = 'http://localhost:8000';

// State
let sessionId = null;
let currentStep = 1;
let jobId = null;
let statusPollInterval = null;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const removeFile = document.getElementById('removeFile');
const description = document.getElementById('description');
const analyzeBtn = document.getElementById('analyzeBtn');
const planBtn = document.getElementById('planBtn');
const generateBtn = document.getElementById('generateBtn');
const trainCloudBtn = document.getElementById('trainCloudBtn');
const downloadBtn = document.getElementById('downloadBtn');
const downloadModelBtn = document.getElementById('downloadModelBtn');
const startOver = document.getElementById('startOver');
const backToUpload = document.getElementById('backToUpload');
const backToAnalysis = document.getElementById('backToAnalysis');
const backToPlan = document.getElementById('backToPlan');
const continueToTrainBtn = document.getElementById('continueToTrainBtn');

// ============================================================================
// STEP NAVIGATION
// ============================================================================

function setStep(step) {
    currentStep = step;
    
    // Update progress bar
    document.querySelectorAll('.step').forEach((el, i) => {
        el.classList.remove('active', 'completed');
        if (i + 1 < step) {
            el.classList.add('completed');
        } else if (i + 1 === step) {
            el.classList.add('active');
        }
    });
    
    // Show/hide step content
    document.querySelectorAll('.step-content').forEach((el, i) => {
        el.classList.remove('active');
        if (i + 1 === step) {
            el.classList.add('active');
        }
    });
}

// ============================================================================
// FILE UPLOAD
// ============================================================================

// Drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelect(file);
});

removeFile.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['.csv', '.json', '.jsonl'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(ext)) {
        alert('Please upload a CSV, JSON, or JSONL file');
        return;
    }
    
    // Upload file
    uploadFile(file);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        setButtonLoading(analyzeBtn, true);
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData,
            // Don't set Content-Type header - browser will set it with boundary
        });
        
        if (!response.ok) {
            // Try to parse error as JSON, fallback to text
            let errorMsg = `HTTP ${response.status}: Upload failed`;
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
        
        // Show file info
        fileName.textContent = file.name;
        fileInfo.classList.remove('hidden');
        dropZone.style.display = 'none';
        
        updateAnalyzeButton();
        
    } catch (error) {
        console.error('Upload error:', error);
        // Check if it's a network error
        if (error.message === 'Failed to fetch' || error.message.includes('NetworkError')) {
            alert('Upload failed: Cannot connect to server. Is the backend running on http://localhost:8000?');
        } else {
            alert('Upload failed: ' + error.message);
        }
    } finally {
        setButtonLoading(analyzeBtn, false);
    }
}

function clearFile() {
    sessionId = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    dropZone.style.display = '';
    updateAnalyzeButton();
}

function updateAnalyzeButton() {
    analyzeBtn.disabled = !(sessionId && description.value.trim());
}

description.addEventListener('input', updateAnalyzeButton);

// ============================================================================
// ANALYZE
// ============================================================================

analyzeBtn.addEventListener('click', async () => {
    if (!sessionId || !description.value.trim()) return;
    
    try {
        setButtonLoading(analyzeBtn, true);
        
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_description: description.value.trim(),
            }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Analysis failed');
        }
        
        // Update UI
        document.getElementById('numRows').textContent = data.num_rows.toLocaleString();
        document.getElementById('numCols').textContent = data.columns.length;
        document.getElementById('quality').textContent = (data.quality_score * 100).toFixed(0) + '%';
        document.getElementById('taskType').textContent = formatTaskType(data.inferred_task_type);
        
        // Columns
        const columnsList = document.getElementById('columnsList');
        columnsList.innerHTML = data.columns.map(col => 
            `<span class="column-tag">${col}</span>`
        ).join('');
        
        // Issues
        const issuesList = document.getElementById('issuesList');
        const issuesSection = document.getElementById('issuesSection');
        
        if (data.quality_issues && data.quality_issues.length > 0 && 
            data.quality_issues[0] !== 'Data looks good!') {
            issuesList.innerHTML = data.quality_issues.map(issue => 
                `<li>${issue}</li>`
            ).join('');
            issuesSection.classList.remove('hidden');
        } else {
            issuesSection.classList.add('hidden');
        }
        
        setStep(2);
        
    } catch (error) {
        alert('Analysis failed: ' + error.message);
    } finally {
        setButtonLoading(analyzeBtn, false);
    }
});

// ============================================================================
// PLAN
// ============================================================================

planBtn.addEventListener('click', async () => {
    try {
        setButtonLoading(planBtn, true);
        
        const response = await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Planning failed');
        }
        
        // Update UI
        document.getElementById('planTask').textContent = formatTaskType(data.final_task_type);
        document.getElementById('planModel').textContent = data.base_model;
        document.getElementById('planReasoning').textContent = data.reasoning;
        
        // Config grid
        const configGrid = document.getElementById('configGrid');
        const configItems = [
            ['Learning Rate', data.training_config.learning_rate],
            ['Batch Size', data.training_config.batch_size],
            ['Epochs', data.training_config.num_epochs],
            ['Max Length', data.training_config.max_length],
        ];
        
        if (data.training_config.num_labels) {
            configItems.push(['Num Labels', data.training_config.num_labels]);
        }
        
        if (data.training_config.use_qlora) {
            configItems.push(['Method', 'QLoRA']);
            configItems.push(['LoRA Rank', data.training_config.lora_r]);
        } else {
            configItems.push(['Method', 'Full Fine-tune']);
        }
        
        configGrid.innerHTML = configItems.map(([key, value]) => `
            <div class="config-item">
                <span class="key">${key}</span>
                <span class="value">${value}</span>
            </div>
        `).join('');
        
        setStep(3);
        
    } catch (error) {
        alert('Planning failed: ' + error.message);
    } finally {
        setButtonLoading(planBtn, false);
    }
});

// ============================================================================
// CONTINUE TO TRAIN (Step 3 -> Step 4)
// ============================================================================

continueToTrainBtn.addEventListener('click', () => {
    setStep(4);
});

// ============================================================================
// TRAIN ON CLOUD (Modal)
// ============================================================================

trainCloudBtn.addEventListener('click', async () => {
    try {
        setButtonLoading(trainCloudBtn, true);
        
        const response = await fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to start training');
        }
        
        jobId = data.job_id;
        
        // Show training progress
        showTrainingProgress(data);
        setStep(5);
        
        // Start polling for status
        startStatusPolling();
        
    } catch (error) {
        alert('Training failed to start: ' + error.message);
    } finally {
        setButtonLoading(trainCloudBtn, false);
    }
});

let pollCount = 0;
let trainingStartTime = null;

function showTrainingProgress(data) {
    document.getElementById('trainingProgress').classList.remove('hidden');
    document.getElementById('trainingComplete').classList.add('hidden');
    document.getElementById('downloadPackage').classList.add('hidden');
    
    document.getElementById('jobIdDisplay').textContent = data.job_id;
    document.getElementById('statusDisplay').textContent = 'Running';
    document.getElementById('startTimeDisplay').textContent = new Date().toLocaleTimeString();
    
    // Clear previous logs
    document.getElementById('logOutput').innerHTML = '';
    pollCount = 0;
    trainingStartTime = Date.now();
    
    addLogLine('🎯 Job ID: ' + data.job_id);
    addLogLine('☁️ Spinning up GPU container on Modal...');
    addLogLine('⏳ Training will begin shortly...');
}

function addLogLine(message, type = 'normal') {
    const logOutput = document.getElementById('logOutput');
    const line = document.createElement('div');
    line.className = 'log-line';
    if (type === 'success') line.classList.add('log-success');
    if (type === 'error') line.classList.add('log-error');
    line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logOutput.appendChild(line);
    logOutput.scrollTop = logOutput.scrollHeight;
    
    // Keep only last 20 log lines to prevent overflow
    while (logOutput.children.length > 20) {
        logOutput.removeChild(logOutput.firstChild);
    }
}

function startStatusPolling() {
    // Poll every 3 seconds for more responsive updates
    statusPollInterval = setInterval(checkTrainingStatus, 3000);
    
    // Check immediately
    setTimeout(checkTrainingStatus, 1000);
}

function stopStatusPolling() {
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }
}

async function checkTrainingStatus() {
    if (!jobId) return;
    
    try {
        const response = await fetch(`${API_URL}/training-status/${jobId}`);
        const data = await response.json();
        
        document.getElementById('statusDisplay').textContent = capitalizeFirst(data.status);
        
        if (data.status === 'completed') {
            stopStatusPolling();
            const elapsed = Math.round((Date.now() - trainingStartTime) / 1000);
            addLogLine(`✅ Training completed in ${formatDuration(elapsed)}!`, 'success');
            showTrainingComplete(data);
        } else if (data.status === 'failed') {
            stopStatusPolling();
            addLogLine('❌ Training failed: ' + (data.error || 'Unknown error'), 'error');
            document.getElementById('statusDisplay').textContent = 'Failed';
            document.getElementById('statusDisplay').style.color = '#ef4444';
        } else if (data.status === 'running') {
            // Show elapsed time
            const elapsed = Math.round((Date.now() - trainingStartTime) / 1000);
            const lastLine = document.querySelector('#logOutput .log-line:last-child');
            
            // Update the last line with elapsed time instead of adding new lines
            if (lastLine && lastLine.textContent.includes('Training on GPU')) {
                lastLine.textContent = `[${new Date().toLocaleTimeString()}] ⚡ Training on GPU... (${formatDuration(elapsed)})`;
            } else if (pollCount === 0) {
                addLogLine('⚡ Training on GPU...');
            } else if (pollCount % 5 === 0) {
                // Add elapsed time update every 15 seconds (5 polls * 3s)
                addLogLine(`⚡ Training on GPU... (${formatDuration(elapsed)})`);
            }
            pollCount++;
        }
        
    } catch (error) {
        console.error('Status check failed:', error);
    }
}

function showTrainingComplete(data) {
    document.getElementById('trainingProgress').classList.add('hidden');
    document.getElementById('trainingComplete').classList.remove('hidden');
    
    console.log('[TuneKit] Training complete data:', data);
    
    // Show base model name
    if (data.base_model) {
        document.getElementById('baseModelName').textContent = data.base_model;
    }
    
    // Helper to format metric value
    const formatValue = (key, value) => {
        if (typeof value !== 'number') return value;
        if (key.includes('accuracy') || key.includes('f1') || key.includes('precision') || key.includes('recall')) {
            return (value * 100).toFixed(1) + '%';
        }
        return value.toFixed(4);
    };
    
    // Helper to render metrics
    const renderMetrics = (metrics, containerId, fallbackMetrics = null) => {
        const container = document.getElementById(containerId);
        
        // Use fallback if primary metrics not available
        const metricsToShow = metrics || fallbackMetrics;
        
        console.log(`[TuneKit] Rendering metrics for ${containerId}:`, metricsToShow);
        
        if (!metricsToShow || Object.keys(metricsToShow).length === 0) {
            container.innerHTML = '<p style="color: var(--text-muted); font-size: 12px;">No metrics available</p>';
            return;
        }
        
        const keyMetrics = ['accuracy', 'f1', 'loss'];
        let html = '';
        
        keyMetrics.forEach(key => {
            // Try both 'accuracy' and 'eval_accuracy' formats
            let value = metricsToShow[key];
            if (value === undefined) {
                value = metricsToShow[`eval_${key}`];
            }
            
            if (value !== undefined) {
                html += `
                    <div class="card-metric">
                        <span class="card-metric-label">${key}</span>
                        <span class="card-metric-value">${formatValue(key, value)}</span>
                    </div>
                `;
            }
        });
        
        container.innerHTML = html || '<p style="color: var(--text-muted);">-</p>';
    };
    
    // Check if this is an old training job (no comparison metrics)
    const hasComparisonMetrics = data.base_metrics && data.finetuned_metrics;
    
    if (!hasComparisonMetrics && data.metrics) {
        // Old training job - show a note and use regular metrics for fine-tuned
        const comparisonContainer = document.querySelector('.comparison-container');
        if (comparisonContainer && !comparisonContainer.previousElementSibling?.classList?.contains('comparison-note')) {
            const note = document.createElement('div');
            note.className = 'comparison-note';
            note.style.cssText = 'background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 16px; text-align: center; font-size: 13px; color: var(--text-secondary);';
            note.innerHTML = 'ℹ️ This training was done before comparison metrics were added. <strong>Retrain to see before/after comparison!</strong>';
            comparisonContainer.parentNode.insertBefore(note, comparisonContainer);
        }
    }
    
    // Render base metrics (or show "N/A" for old jobs)
    if (hasComparisonMetrics) {
        renderMetrics(data.base_metrics, 'baseMetrics');
    } else {
        document.getElementById('baseMetrics').innerHTML = '<p style="color: var(--text-muted); font-size: 12px;">Not available (retrain to see)</p>';
    }
    
    // Render fine-tuned metrics (use fallback for old jobs)
    // For old jobs, data.metrics should have the fine-tuned metrics
    const finetunedMetrics = data.finetuned_metrics || data.metrics;
    renderMetrics(finetunedMetrics, 'tunedMetrics');
    
    // Show improvement summary (in percentage points)
    const improvementSummary = document.getElementById('improvementSummary');
    if (data.improvement) {
        let html = '';
        Object.entries(data.improvement).forEach(([key, value]) => {
            const isPositive = value > 0;
            const icon = isPositive ? '📈' : '📉';
            const sign = isPositive ? '+' : '';
            // Value is already in percentage points (e.g., 44.1, not 0.441)
            html += `
                <div class="improvement-badge ${isPositive ? '' : 'negative'}">
                    <span class="improvement-icon">${icon}</span>
                    <span class="improvement-text">${key}: ${sign}${value.toFixed(1)} pp</span>
                </div>
            `;
        });
        improvementSummary.innerHTML = html;
    } else {
        improvementSummary.innerHTML = '';
    }
    
    // Setup "Try it out" comparison
    setupTryItOut();
    
    // Set download URL
    const downloadUrl = `${API_URL}/download-model-zip/${jobId}`;
    
    downloadModelBtn.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        const originalText = downloadModelBtn.innerHTML;
        downloadModelBtn.innerHTML = '⏳ Preparing download... (may take 30-60s)';
        downloadModelBtn.disabled = true;
        downloadModelBtn.style.pointerEvents = 'none';
        
        // Show countdown with elapsed time
        let countdown = 120; // 2 minutes max
        const startTime = Date.now();
        const countdownInterval = setInterval(() => {
            countdown--;
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            if (countdown > 0) {
                downloadModelBtn.innerHTML = `⏳ Preparing download... (${elapsed}s elapsed)`;
            } else {
                downloadModelBtn.innerHTML = `⏳ Still preparing... (${elapsed}s)`;
            }
        }, 1000);
        
        try {
            const downloadStartTime = Date.now();
            console.log('[Download] Starting download from:', downloadUrl);
            console.log('[Download] Start time:', new Date().toLocaleTimeString());
            
            const response = await fetch(downloadUrl);
            const fetchTime = ((Date.now() - downloadStartTime) / 1000).toFixed(1);
            console.log(`[Download] Response received in ${fetchTime}s - Status:`, response.status, response.statusText);
            
            if (!response.ok) {
                let errorMsg = `HTTP ${response.status}: Download failed`;
                try {
                    const error = await response.json();
                    errorMsg = error.detail || error.error || errorMsg;
                } catch {
                    try {
                        errorMsg = await response.text() || errorMsg;
                    } catch {
                        errorMsg = `HTTP ${response.status}`;
                    }
                }
                throw new Error(errorMsg);
            }
            
            const contentType = response.headers.get('content-type');
            console.log('[Download] Content-Type:', contentType);
            
            const blob = await response.blob();
            const totalTime = ((Date.now() - downloadStartTime) / 1000).toFixed(1);
            console.log(`[Download] Blob received: ${blob.size} bytes`);
            console.log(`[Download] Total time: ${totalTime}s`);
            
            if (blob.size === 0) {
                throw new Error('Downloaded file is empty (0 bytes)');
            }
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `tunekit_model_${jobId}.zip`;
            document.body.appendChild(a);
            
            console.log('[Download] Triggering browser download...');
            a.click();
            
            // Clean up after a delay
            setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                console.log('[Download] Cleanup complete');
            }, 1000);
            
            clearInterval(countdownInterval);
            const finalTime = ((Date.now() - downloadStartTime) / 1000).toFixed(1);
            downloadModelBtn.innerHTML = `✅ Downloaded! (${finalTime}s)`;
            console.log(`[Download] ✅ SUCCESS! Total time: ${finalTime}s`);
            setTimeout(() => {
                downloadModelBtn.innerHTML = originalText;
                downloadModelBtn.disabled = false;
                downloadModelBtn.style.pointerEvents = 'auto';
            }, 3000);
            
        } catch (error) {
            clearInterval(countdownInterval);
            console.error('[Download] Error:', error);
            alert('❌ Download failed: ' + error.message + '\n\nThe download may have timed out. Try again?');
            downloadModelBtn.innerHTML = originalText;
            downloadModelBtn.disabled = false;
            downloadModelBtn.style.pointerEvents = 'auto';
        }
    };
}

// ============================================================================
// TRY IT OUT - COMPARE MODELS
// ============================================================================

function setupTryItOut() {
    const tryBtn = document.getElementById('tryBtn');
    const tryInput = document.getElementById('tryInput');
    const tryResults = document.getElementById('tryResults');
    
    tryBtn.onclick = async () => {
        const text = tryInput.value.trim();
        if (!text) {
            alert('Please enter some text to classify');
            return;
        }
        
        tryBtn.disabled = true;
        tryBtn.textContent = 'Running...';
        
        try {
            // Clear previous results
            tryResults.classList.add('hidden');
            
            const response = await fetch(`${API_URL}/compare`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: jobId, text: text }),
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Show results
            tryResults.classList.remove('hidden');
            
            if (data.base) {
                document.getElementById('basePrediction').textContent = data.base.prediction || '-';
                document.getElementById('baseConfidence').textContent = 
                    data.base.confidence ? `${data.base.confidence}% confidence` : '';
            } else {
                document.getElementById('basePrediction').textContent = '-';
                document.getElementById('baseConfidence').textContent = '';
            }
            
            if (data.finetuned) {
                document.getElementById('tunedPrediction').textContent = data.finetuned.prediction || '-';
                document.getElementById('tunedConfidence').textContent = 
                    data.finetuned.confidence ? `${data.finetuned.confidence}% confidence` : '';
            } else {
                document.getElementById('tunedPrediction').textContent = '-';
                document.getElementById('tunedConfidence').textContent = '';
            }
            
        } catch (error) {
            console.error('Comparison error:', error);
            alert('❌ Comparison failed: ' + error.message + '\n\nThis might take 10-15 seconds to spin up the GPU. Try again?');
        } finally {
            tryBtn.disabled = false;
            tryBtn.textContent = 'Compare Models';
        }
    };
}

// ============================================================================
// GENERATE LOCAL PACKAGE
// ============================================================================

generateBtn.addEventListener('click', async () => {
    try {
        setButtonLoading(generateBtn, true);
        
        const response = await fetch(`${API_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Generation failed');
        }
        
        // Show download package section
        document.getElementById('trainingProgress').classList.add('hidden');
        document.getElementById('trainingComplete').classList.add('hidden');
        document.getElementById('downloadPackage').classList.remove('hidden');
        
        // Update download link
        downloadBtn.href = `${API_URL}${data.download_url}`;
        
        setStep(5);
        
    } catch (error) {
        alert('Generation failed: ' + error.message);
    } finally {
        setButtonLoading(generateBtn, false);
    }
});

// ============================================================================
// NAVIGATION
// ============================================================================

backToUpload.addEventListener('click', () => setStep(1));
backToAnalysis.addEventListener('click', () => setStep(2));
backToPlan.addEventListener('click', () => setStep(3));

startOver.addEventListener('click', () => {
    sessionId = null;
    jobId = null;
    stopStatusPolling();
    clearFile();
    description.value = '';
    
    // Reset step 5 views
    document.getElementById('trainingProgress').classList.add('hidden');
    document.getElementById('trainingComplete').classList.add('hidden');
    document.getElementById('downloadPackage').classList.add('hidden');
    document.getElementById('logOutput').innerHTML = '<div class="log-line">Initializing training environment...</div>';
    
    setStep(1);
});

// ============================================================================
// UTILITIES
// ============================================================================

function setButtonLoading(btn, loading) {
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    
    if (loading) {
        btn.disabled = true;
        if (text) text.style.opacity = '0.5';
        if (loader) loader.classList.remove('hidden');
    } else {
        btn.disabled = false;
        if (text) text.style.opacity = '1';
        if (loader) loader.classList.add('hidden');
    }
}

function formatTaskType(type) {
    const map = {
        'classification': 'Classification',
        'ner': 'NER',
        'instruction_tuning': 'Instruction Tuning',
    };
    return map[type] || type;
}

function formatMetricKey(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds}s`;
    }
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
}

// Initialize
setStep(1);
