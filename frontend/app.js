/**
 * TuneKit Frontend
 * ================
 * Handles API interactions and UI updates
 */

const API_URL = 'http://localhost:8000';

// State
let sessionId = null;
let currentStep = 1;

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
const downloadBtn = document.getElementById('downloadBtn');
const startOver = document.getElementById('startOver');
const backToUpload = document.getElementById('backToUpload');
const backToAnalysis = document.getElementById('backToAnalysis');

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
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }
        
        sessionId = data.session_id;
        
        // Show file info
        fileName.textContent = file.name;
        fileInfo.classList.remove('hidden');
        dropZone.style.display = 'none';
        
        updateAnalyzeButton();
        
    } catch (error) {
        alert('Upload failed: ' + error.message);
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
// GENERATE
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
        
        // Update download link
        downloadBtn.href = `${API_URL}${data.download_url}`;
        
        setStep(4);
        
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

startOver.addEventListener('click', () => {
    sessionId = null;
    clearFile();
    description.value = '';
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

// Initialize
setStep(1);

