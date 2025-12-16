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
let uploadedFileData = null; // Store file data for preview

// User configuration overrides (from Advanced Options)
let userConfig = {
    inputColumn: null,
    outputColumn: null,
    taskType: null
};

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileRows = document.getElementById('fileRows');
const fileSize = document.getElementById('fileSize');
const removeFile = document.getElementById('removeFile');
const continueBtn = document.getElementById('continueBtn');
const generateBtn = document.getElementById('generateBtn');
const trainCloudBtn = document.getElementById('trainCloudBtn');
const downloadBtn = document.getElementById('downloadBtn');
const downloadModelBtn = document.getElementById('downloadModelBtn');
const startOver = document.getElementById('startOver');
const backToUpload = document.getElementById('backToUpload');
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
    const validTypes = ['.jsonl', '.json', '.parquet'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(ext)) {
        alert('Please upload a JSONL, JSON, or Parquet file');
        return;
    }
    
    // Upload file
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
    
    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
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
        fileRows.textContent = `✓ ${data.rows?.toLocaleString() || ''} examples`;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.classList.remove('hidden');
        dropZone.style.display = 'none';
        
        // Enable continue button
        updateContinueButton();
        
    } catch (error) {
        console.error('Upload error:', error);
        if (error.message === 'Failed to fetch' || error.message.includes('NetworkError')) {
            alert('Upload failed: Cannot connect to server. Is the backend running on http://localhost:8000?');
        } else {
            alert('Upload failed: ' + error.message);
        }
    }
}

function clearFile() {
    sessionId = null;
    uploadedFileData = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    dropZone.style.display = '';
    updateContinueButton();
}

function updateContinueButton() {
    if (continueBtn) {
        continueBtn.disabled = !sessionId;
    }
}

// ============================================================================
// ANALYZE (called automatically when continuing)
// ============================================================================

let analysisData = null;

// (showInlineAnalysis removed - not needed in new design)

// Toggle Advanced Options Modal
const advancedModal = document.getElementById('advancedModal');
const closeAdvancedModal = document.getElementById('closeAdvancedModal');

// Open modal from Configure button on upload page
openAdvancedBtn?.addEventListener('click', () => {
    if (advancedModal) {
        advancedModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
});

// Apply and close modal - SAVE user configuration
applyAdvancedBtn?.addEventListener('click', () => {
    if (advancedModal) {
        // Get user's selected values
        const inputCol = document.getElementById('inputColumnSelect')?.value || '';
        const outputCol = document.getElementById('outputColumnSelect')?.value || '';
        const taskName = document.getElementById('detectedTaskName')?.textContent || '';
        
        // Store in userConfig - these will be sent to the backend
        userConfig.inputColumn = inputCol;
        userConfig.outputColumn = outputCol;
        // Task type is auto-detected, not user-configurable
        // (Removed task type override - it's dangerous to let users change task type
        // without understanding data format requirements)
        userConfig.taskType = null;
        
        console.log('[Config] User configuration saved:', userConfig);
        
        // Update inline display
        document.getElementById('inlineTaskType').textContent = taskName;
        document.getElementById('inlineInputCol').textContent = inputCol;
        document.getElementById('inlineOutputCol').textContent = outputCol;
        
        advancedModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
});

closeAdvancedModal?.addEventListener('click', () => {
    if (advancedModal) {
        advancedModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
});

// Close modal on overlay click
advancedModal?.addEventListener('click', (e) => {
    if (e.target === advancedModal) {
        advancedModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && advancedModal && !advancedModal.classList.contains('hidden')) {
        advancedModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
});

function populateAdvancedOptions(data) {
    // Task Detection Display
    const detectedTaskName = document.getElementById('detectedTaskName');
    
    if (detectedTaskName) {
        detectedTaskName.textContent = formatTaskType(data.inferred_task_type);
    }
    
    // Build label color map from actual data
    if (data.sample_rows && data.sample_rows.length > 0) {
        buildLabelColorMap(data.sample_rows, data.columns);
    }
    
    // Populate Column Selectors
    const inputColumnSelect = document.getElementById('inputColumnSelect');
    const outputColumnSelect = document.getElementById('outputColumnSelect');
    
    if (data.columns && inputColumnSelect && outputColumnSelect) {
        // Use column_candidates from API first (more accurate)
        let guessedInput = null;
        let guessedOutput = null;
        
        if (data.column_candidates) {
            const candidates = data.column_candidates;
            if (candidates.text_column && candidates.text_column.length > 0) {
                guessedInput = candidates.text_column[0];
            } else if (candidates.instruction_column && candidates.instruction_column.length > 0) {
                guessedInput = candidates.instruction_column[0];
            }
            
            if (candidates.label_column && candidates.label_column.length > 0) {
                guessedOutput = candidates.label_column[0];
            } else if (candidates.tags_column && candidates.tags_column.length > 0) {
                guessedOutput = candidates.tags_column[0];
            } else if (candidates.response_column && candidates.response_column.length > 0) {
                guessedOutput = candidates.response_column[0];
            }
        }
        
        // Fallback: pattern matching if API didn't provide candidates
        if (!guessedInput) {
            guessedInput = data.columns.find(col => {
                const colLower = col.toLowerCase();
                return colLower.includes('text') || colLower.includes('content') || 
                       colLower.includes('message') || colLower.includes('review') ||
                       colLower.includes('question') || colLower.includes('input') ||
                       colLower.includes('email') || colLower.includes('body');
            }) || data.columns[0];
        }
        
        if (!guessedOutput) {
            guessedOutput = data.columns.find(col => {
                const colLower = col.toLowerCase();
                return colLower.includes('label') || colLower.includes('category') ||
                       colLower.includes('class') || colLower.includes('sentiment') ||
                       colLower.includes('target') || colLower.includes('output') ||
                       colLower.includes('spam') || colLower.includes('type');
            }) || data.columns[data.columns.length - 1];
        }
        
        // Populate selectors
        inputColumnSelect.innerHTML = data.columns.map(col => 
            `<option value="${col}" ${col === guessedInput ? 'selected' : ''}>${col}</option>`
        ).join('');
        
        outputColumnSelect.innerHTML = data.columns.map(col => 
            `<option value="${col}" ${col === guessedOutput ? 'selected' : ''}>${col}</option>`
        ).join('');
        
        // Refresh sample data when columns change
        inputColumnSelect.addEventListener('change', fetchSampleData);
        outputColumnSelect.addEventListener('change', fetchSampleData);
    }
    
    // Task type is auto-detected and not user-changeable
    // (Removed task type dropdown - changing task type without proper data format is dangerous)
    
    // Quality Checks
    const qualityChecks = document.getElementById('qualityChecks');
    if (qualityChecks) {
        const checks = [];
        
        if (data.quality_score >= 0.8) {
            checks.push({ type: 'pass', text: 'Data quality is good' });
        }
        
        if (data.quality_issues && data.quality_issues.length > 0) {
            data.quality_issues.forEach(issue => {
                if (issue === 'Data looks good!') {
                    checks.push({ type: 'pass', text: 'No quality issues detected' });
                } else {
                    checks.push({ type: 'warn', text: issue });
                }
            });
        } else {
            checks.push({ type: 'pass', text: 'No missing values detected' });
            checks.push({ type: 'pass', text: 'No duplicate rows found' });
        }
        
        qualityChecks.innerHTML = checks.map(check => `
            <div class="quality-check check-${check.type}">${check.text}</div>
        `).join('');
    }
    
    // Fetch and display sample data
    fetchSampleData();
}

async function fetchSampleData() {
    const dataPreview = document.getElementById('dataPreview');
    if (!dataPreview || !sessionId) return;
    
    try {
        // Get currently selected columns
        const inputColSelect = document.getElementById('inputColumnSelect');
        const outputColSelect = document.getElementById('outputColumnSelect');
        
        if (!inputColSelect || !outputColSelect) {
            dataPreview.innerHTML = '<div class="preview-placeholder">Loading...</div>';
            return;
        }
        
        const inputCol = inputColSelect.value;
        const outputCol = outputColSelect.value;
        
        // Use real sample data from analysis response
        if (analysisData && analysisData.sample_rows && analysisData.sample_rows.length > 0) {
            const sampleData = analysisData.sample_rows.slice(0, 5); // First 5 rows
            
            // Rebuild color map when columns change
            buildLabelColorMap(analysisData.sample_rows, analysisData.columns);
            
            dataPreview.innerHTML = `
                <table class="preview-table">
                    <thead>
                        <tr>
                            <th>${inputCol}</th>
                            <th>${outputCol}</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${sampleData.map(row => {
                            const inputValue = row[inputCol] || row[inputCol.toLowerCase()] || '-';
                            const outputValue = row[outputCol] || row[outputCol.toLowerCase()] || '-';
                            return `
                                <tr>
                                    <td>${String(inputValue).substring(0, 100)}${String(inputValue).length > 100 ? '...' : ''}</td>
                                    <td>${formatLabelPill(String(outputValue))}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
                <p style="text-align: right; color: var(--text-muted); font-size: 13px; margin-top: 12px;">
                    <a href="#" class="view-all-rows" data-session="${sessionId}" style="color: var(--accent-secondary); text-decoration: none; cursor: pointer;">
                        View All ${analysisData.num_rows?.toLocaleString() || uploadedFileData?.rows?.toLocaleString() || '1,247'} Rows →
                    </a>
                </p>
            `;
            
            // Add click handler for "View All Rows"
            setTimeout(() => {
                const viewAllLink = dataPreview.querySelector('.view-all-rows');
                if (viewAllLink) {
                    viewAllLink.addEventListener('click', async (e) => {
                        e.preventDefault();
                        await showAllRows(sessionId, inputCol, outputCol);
                    });
                }
            }, 100);
        } else if (uploadedFileData && uploadedFileData.columns) {
            // Fallback: show placeholder if no sample data available
            dataPreview.innerHTML = `
                <div class="preview-placeholder">
                    Sample data will appear here once analysis is complete.
                </div>
            `;
        } else {
            dataPreview.innerHTML = '<div class="preview-placeholder">Upload a file to see preview</div>';
        }
    } catch (error) {
        console.error('Error fetching sample data:', error);
        dataPreview.innerHTML = '<div class="preview-placeholder">Unable to load sample data</div>';
    }
}

// Label color mapping - dynamically assigned
let labelColorMap = {};

function buildLabelColorMap(sampleRows, columns) {
    if (!sampleRows || sampleRows.length === 0 || !columns || columns.length === 0) {
        return;
    }
    
    // Get the currently selected output column
    const outputColSelect = document.getElementById('outputColumnSelect');
    const outputCol = outputColSelect ? outputColSelect.value : (columns[columns.length - 1]);
    
    // Collect unique labels from the output column
    const uniqueLabels = new Set();
    
    for (const row of sampleRows) {
        const value = row[outputCol] || row[outputCol.toLowerCase()];
        if (value !== null && value !== undefined) {
            const labelStr = String(value).trim();
            if (labelStr && labelStr.length < 100) { // Reasonable label length
                uniqueLabels.add(labelStr);
            }
        }
    }
    
    // Only assign colors if we have a reasonable number of unique labels (classification task)
    if (uniqueLabels.size > 0 && uniqueLabels.size <= 50) {
        // Color palette - 8 distinct colors that cycle
        const colorPalette = [
            { bg: 'rgba(124, 58, 237, 0.2)', color: '#a78bfa', border: 'rgba(124, 58, 237, 0.3)' }, // purple
            { bg: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa', border: 'rgba(59, 130, 246, 0.3)' }, // blue
            { bg: 'rgba(16, 185, 129, 0.2)', color: '#10b981', border: 'rgba(16, 185, 129, 0.3)' }, // green
            { bg: 'rgba(245, 158, 11, 0.2)', color: '#f59e0b', border: 'rgba(245, 158, 11, 0.3)' }, // yellow
            { bg: 'rgba(236, 72, 153, 0.2)', color: '#ec4899', border: 'rgba(236, 72, 153, 0.3)' }, // pink
            { bg: 'rgba(6, 182, 212, 0.2)', color: '#06b6d4', border: 'rgba(6, 182, 212, 0.3)' }, // cyan
            { bg: 'rgba(249, 115, 22, 0.2)', color: '#f97316', border: 'rgba(249, 115, 22, 0.3)' }, // orange
            { bg: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', border: 'rgba(239, 68, 68, 0.3)' }, // red
        ];
        
        // Sort labels for consistent color assignment
        const sortedLabels = Array.from(uniqueLabels).sort();
        labelColorMap = {};
        
        sortedLabels.forEach((label, index) => {
            const color = colorPalette[index % colorPalette.length];
            labelColorMap[label.toLowerCase()] = color;
        });
    }
}

function formatLabelPill(label) {
    if (!label) return '<span class="label-pill default">-</span>';
    
    const labelStr = String(label).trim();
    const labelKey = labelStr.toLowerCase();
    
    // Check if we have a color for this label
    if (labelColorMap[labelKey]) {
        const color = labelColorMap[labelKey];
        return `<span class="label-pill" style="background: ${color.bg}; color: ${color.color}; border: 1px solid ${color.border};">${labelStr}</span>`;
    }
    
    // Default styling
    return `<span class="label-pill default">${labelStr}</span>`;
}

async function showAllRows(sessionId, inputCol, outputCol) {
    const dataPreview = document.getElementById('dataPreview');
    if (!dataPreview || !sessionId) return;
    
    try {
        // Show loading state
        const originalContent = dataPreview.innerHTML;
        dataPreview.innerHTML = '<div class="preview-placeholder">Loading all rows...</div>';
        
        // Fetch all rows from API
        const response = await fetch(`${API_URL}/sample-data/${sessionId}?limit=600`);
        if (!response.ok) {
            throw new Error('Failed to fetch rows');
        }
        
        const data = await response.json();
        const allRows = data.rows || [];
        const totalRows = data.total_rows || allRows.length;
        
        // Rebuild color map from all rows to ensure all labels have colors
        if (allRows.length > 0) {
            const outputColSelect = document.getElementById('outputColumnSelect');
            const columns = outputColSelect ? Array.from(outputColSelect.options).map(opt => opt.value) : [outputCol];
            buildLabelColorMap(allRows, columns);
        }
        
        // Create scrollable table with all rows
        dataPreview.innerHTML = `
            <div class="all-rows-container">
                <div class="all-rows-header">
                    <span>Showing ${allRows.length.toLocaleString()} of ${totalRows.toLocaleString()} rows</span>
                    <button class="btn-close-preview" id="showLessBtn">Show Less</button>
                </div>
                <div class="all-rows-scroll">
                    <table class="preview-table">
                        <thead>
                            <tr>
                                <th>${inputCol}</th>
                                <th>${outputCol}</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${allRows.map(row => {
                                const inputValue = row[inputCol] || row[inputCol.toLowerCase()] || '-';
                                const outputValue = row[outputCol] || row[outputCol.toLowerCase()] || '-';
                                return `
                                    <tr>
                                        <td>${String(inputValue).substring(0, 200)}${String(inputValue).length > 200 ? '...' : ''}</td>
                                        <td>${formatLabelPill(String(outputValue))}</td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        // Add click handler for "Show Less" button
        const showLessBtn = document.getElementById('showLessBtn');
        if (showLessBtn) {
            showLessBtn.addEventListener('click', () => {
                fetchSampleData(); // Reload the 5-row preview
            });
        }
        
    } catch (error) {
        console.error('Error loading all rows:', error);
        dataPreview.innerHTML = '<div class="preview-placeholder">Unable to load all rows. Please try again.</div>';
    }
}

// ============================================================================
// CONTINUE (Analyze + Plan in one step)
// ============================================================================

continueBtn?.addEventListener('click', async () => {
    if (!sessionId) return;
    
    try {
        setButtonLoading(continueBtn, true);
        
        // Step 1: Analyze the dataset
        const analyzeResponse = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_description: 'Fine-tune model on this dataset',
            }),
        });
        
        const analyzeData = await analyzeResponse.json();
        
        if (!analyzeResponse.ok) {
            throw new Error(analyzeData.detail || 'Analysis failed');
        }
        
        analysisData = analyzeData;
        
        // Step 2: Generate training plan
        const planResponse = await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        });
        
        const planData = await planResponse.json();
        
        if (!planResponse.ok) {
            throw new Error(planData.detail || 'Planning failed');
        }
        
        // Update Plan UI and go to step 2
        updatePlanUI(planData);
        setStep(2);
        
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        setButtonLoading(continueBtn, false);
    }
});

// ============================================================================
// UPDATE PLAN UI
// ============================================================================

function updatePlanUI(data) {
    const config = data.training_config || {};
    const numRows = analysisData?.num_rows || 1000;
    
    // Top Info Grid
    document.getElementById('planTask').textContent = formatTaskType(data.final_task_type);
    document.getElementById('planClasses').textContent = config.num_labels ? `${config.num_labels} classes` : '';
    
    const modelName = data.base_model || 'distilbert-base-uncased';
    document.getElementById('planModel').textContent = modelName.split('/').pop();
    document.getElementById('planConfidence').textContent = `${Math.round((data.task_confidence || 0.9) * 100)}% confidence`;
    
    const strategy = config.use_qlora ? 'QLoRA' : 'Full Fine-tune';
    document.getElementById('planStrategy').textContent = strategy;
    const strategyMeta = [];
    if (config.use_class_weights) strategyMeta.push('Class Weights');
    if (config.use_early_stopping) strategyMeta.push('Early Stopping');
    document.getElementById('planStrategyMeta').textContent = strategyMeta.length > 0 ? strategyMeta.join(', ') : '';
    
    // Calculate training time (more realistic estimates)
    const epochs = config.num_epochs || 3;
    const batchSize = config.batch_size || 32;
    const stepsPerEpoch = Math.ceil(numRows * 0.8 / batchSize);
    const totalSteps = stepsPerEpoch * epochs;
    
    // Model-specific time per step (seconds)
    // distilbert: ~0.3-0.5s, bert-base: ~0.5-0.8s, roberta: ~0.6-1.0s
    let secondsPerStep = 0.5; // Default
    if (modelName.includes('distilbert')) {
        secondsPerStep = 0.4;
    } else if (modelName.includes('roberta') || modelName.includes('bert-large')) {
        secondsPerStep = 0.7;
    } else if (modelName.includes('bert-base')) {
        secondsPerStep = 0.6;
    }
    
    const trainSeconds = Math.ceil(totalSteps * secondsPerStep);
    const trainMinutes = Math.max(1, Math.ceil(trainSeconds / 60));
    
    // Add buffer for validation and overhead (validation is ~20% slower)
    const valSteps = Math.ceil(numRows * 0.1 / batchSize) * epochs;
    const valSeconds = Math.ceil(valSteps * secondsPerStep * 1.2);
    const valMinutes = Math.ceil(valSeconds / 60);
    
    // Total with overhead
    const totalMinutes = trainMinutes + valMinutes + 1; // +1 for load/save
    
    // Show range for more honesty
    const minTime = Math.max(1, Math.floor(totalMinutes * 0.7));
    const maxTime = Math.ceil(totalMinutes * 1.5);
    
    document.getElementById('trainTime').textContent = `(~${trainMinutes}-${trainMinutes + valMinutes} min)`;
    document.getElementById('totalTime').textContent = `~${minTime}-${maxTime} minutes`;
    
    // Config Grid
    const configGrid = document.getElementById('configGrid');
    const configItems = [
        ['Model', modelName.split('/').pop()],
        ['Learning Rate', config.learning_rate || '2e-5'],
        ['Batch Size', config.batch_size || '32'],
        ['Epochs', config.num_epochs || '3'],
        ['Max Length', config.max_length || '128'],
    ];
    
    if (config.num_labels) {
        configItems.push(['Num Labels', config.num_labels]);
    }
    
    configGrid.innerHTML = configItems.map(([key, value]) => `
        <div class="config-item-simple">
            <span class="key">${key}</span>
            <span class="value">${value}</span>
        </div>
    `).join('');
    
    // Reasoning (preview and full) - Format for readability
    const reasoning = data.reasoning || 'Analysis complete. Ready to train.';
    const formattedReasoning = formatReasoningText(reasoning);
    
    // For preview, strip HTML and truncate plain text
    const plainText = reasoning.replace(/<[^>]*>/g, '');
    const previewText = plainText.length > 200 ? plainText.substring(0, 200) + '...' : plainText;
    
    document.getElementById('planReasoning').textContent = previewText;
    document.getElementById('planReasoningFull').innerHTML = formattedReasoning;
    
    // Setup reasoning modal handlers
    setupReasoningModal();
}

// Track if modal handlers are already set up
let reasoningModalSetup = false;

function setupReasoningModal() {
    if (reasoningModalSetup) return; // Already set up
    
    const viewBtn = document.getElementById('viewReasoningBtn');
    const modal = document.getElementById('reasoningModal');
    const closeBtn = document.getElementById('closeReasoningBtn');
    
    if (!viewBtn || !modal || !closeBtn) return;
    
    viewBtn.addEventListener('click', () => {
        modal.classList.remove('hidden');
    });
    
    closeBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
    });
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });
    
    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            modal.classList.add('hidden');
        }
    });
    
    reasoningModalSetup = true;
}

// ============================================================================
// CONTINUE TO TRAIN (Step 2 -> Step 3)
// ============================================================================

continueToTrainBtn.addEventListener('click', () => {
    setStep(3); // Train is now step 3
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
        setStep(4); // Training/Done is now step 4
        
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
        
        setStep(4); // Done is now step 4
        
    } catch (error) {
        alert('Generation failed: ' + error.message);
    } finally {
        setButtonLoading(generateBtn, false);
    }
});

// ============================================================================
// NAVIGATION
// ============================================================================

backToUpload.addEventListener('click', () => {
    userConfig = { inputColumn: null, outputColumn: null, taskType: null };
    setStep(1);
});

backToPlan.addEventListener('click', () => setStep(2));

startOver.addEventListener('click', () => {
    sessionId = null;
    jobId = null;
    stopStatusPolling();
    clearFile();
    
    userConfig = { inputColumn: null, outputColumn: null, taskType: null };
    
    // Reset step 4 views
    document.getElementById('trainingProgress')?.classList.add('hidden');
    document.getElementById('trainingComplete')?.classList.add('hidden');
    document.getElementById('downloadPackage')?.classList.add('hidden');
    const logOutput = document.getElementById('logOutput');
    if (logOutput) logOutput.innerHTML = '<div class="log-line">Initializing...</div>';
    
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

function formatReasoningText(text) {
    if (!text) return '<p>Analysis complete. Ready to train.</p>';
    
    // Split into sentences
    const sentences = text.match(/[^\.!?]+[\.!?]+/g) || [text];
    
    // Group sentences into logical paragraphs (2-3 sentences each)
    const paragraphs = [];
    let currentPara = [];
    
    sentences.forEach((sentence, index) => {
        sentence = sentence.trim();
        if (!sentence) return;
        
        currentPara.push(sentence);
        
        // Start new paragraph every 2-3 sentences, or on topic shifts
        const isTopicShift = /(?:The|This|These|Based on|For|In|Using|Selected)/i.test(sentence);
        const shouldBreak = currentPara.length >= 2 && (isTopicShift || currentPara.length >= 3);
        
        if (shouldBreak || index === sentences.length - 1) {
            paragraphs.push(currentPara.join(' '));
            currentPara = [];
        }
    });
    
    // Format paragraphs - clean and simple, no aggressive highlighting
    return paragraphs.map(para => `<p>${para}</p>`).join('');
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

// ============================================================================
// THEME TOGGLE
// ============================================================================

const themeToggle = document.getElementById('themeToggle');

function initTheme() {
    // Check for saved preference or system preference
    const savedTheme = localStorage.getItem('tunekit-theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    } else if (!systemPrefersDark) {
        document.documentElement.setAttribute('data-theme', 'light');
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('tunekit-theme', newTheme);
}

themeToggle?.addEventListener('click', toggleTheme);

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('tunekit-theme')) {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
    }
});

// Initialize theme
initTheme();

// Initialize step
setStep(1);

// ============================================================================
// API KEY STATUS
// ============================================================================

function checkApiKeyStatus() {
    const apiStatus = document.getElementById('apiStatus');
    if (!apiStatus) return;
    
    const hasOpenAI = localStorage.getItem('tunekit_openai_key');
    const hasModal = localStorage.getItem('tunekit_modal_key');
    const hasModalSecret = localStorage.getItem('tunekit_modal_secret');
    
    const statusText = apiStatus.querySelector('.status-text');
    
    if (hasOpenAI && hasModal && hasModalSecret) {
        apiStatus.classList.add('connected');
        apiStatus.classList.remove('disconnected');
        apiStatus.title = 'API keys configured. Click to manage.';
        if (statusText) statusText.textContent = 'Connected';
    } else if (hasOpenAI || hasModal) {
        apiStatus.classList.remove('connected', 'disconnected');
        apiStatus.title = 'Some API keys missing. Click to configure.';
        if (statusText) statusText.textContent = 'Partial';
    } else {
        apiStatus.classList.add('disconnected');
        apiStatus.classList.remove('connected');
        apiStatus.title = 'No API keys configured. Click to set up.';
        if (statusText) statusText.textContent = 'Setup';
    }
    
    // Click to go to landing page for key management
    apiStatus.addEventListener('click', () => {
        window.location.href = '/#get-started';
    });
}

// Check API key status on load
checkApiKeyStatus();
