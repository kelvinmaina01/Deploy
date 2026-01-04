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
let selectedModelData = null; // The currently selected model (could be primary or alternative)
let allModelsData = []; // All models (primary + alternatives) for easy switching

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
    
    // Update top progress bar
    document.querySelectorAll('.progress-step-nav').forEach((el, i) => {
        el.classList.remove('active', 'completed');
        if (i + 1 < step) {
            el.classList.add('completed');
        } else if (i + 1 === step) {
            el.classList.add('active');
        }
    });
    
    // Update connector lines
    document.querySelectorAll('.progress-connector').forEach((el, i) => {
        el.classList.remove('completed');
        if (i + 1 < step) {
            el.classList.add('completed');
        }
    });
    
    // Scroll to top smoothly
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// QUESTION FADE TRANSITIONS & TYPEWRITER EFFECT
// ============================================================================

function typewriterEffect(element, text, callback) {
    // Clear any existing intervals
    if (element._typewriterInterval) {
        clearInterval(element._typewriterInterval);
    }

    let index = 0;
    element.classList.add('typing');
    element.textContent = '';

    // Start typing immediately
    element._typewriterInterval = setInterval(() => {
        if (index < text.length) {
            element.textContent += text[index];
            index++;
        } else {
            clearInterval(element._typewriterInterval);
            element._typewriterInterval = null;
            // Keep cursor briefly, then remove
            setTimeout(() => {
                element.classList.remove('typing');
                if (callback) callback();
            }, 300);
        }
    }, 45); // Smooth typing speed
}

function showQuestion(questionId) {
    const questionWrapper = document.getElementById(questionId);
    const allQuestions = document.querySelectorAll('.flow-question');

    // Hide all questions first
    allQuestions.forEach(q => {
        q.classList.remove('active');
    });

    // Clear the question text before showing
    const questionText = questionWrapper.querySelector('.flow-question-text');
    if (questionText) {
        questionText.textContent = '';
        questionText.classList.remove('typing');
    }

    // Reset option animations and selections for this question
    const options = questionWrapper.querySelectorAll('.flow-option');
    options.forEach((opt, i) => {
        // Clear selection state
        opt.classList.remove('selected');
        opt.removeAttribute('style');
        // Reset animations
        opt.style.animation = 'none';
        opt.offsetHeight; // Force reflow
        opt.style.animation = '';
        opt.style.setProperty('--delay', i);
    });

    // Show the question wrapper
    questionWrapper.classList.add('active');

    // Start typewriter immediately
    if (questionText && questionText.dataset.text) {
        typewriterEffect(questionText, questionText.dataset.text);
    }
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
    
    // Show loading state immediately without any delay
    continueBtn.disabled = true;
    continueBtn.textContent = 'Uploading...';
    
    // Add spinner after text change
    const spinner = document.createElement('span');
    spinner.className = 'btn-loader';
    continueBtn.appendChild(spinner);
        
    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            let errorMsg = `Upload failed`;
            const text = await response.text();
            try {
                const error = JSON.parse(text);
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
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

    // Training estimate section
    const totalExamples = stats.total_examples || 0;
    const avgTokens = Math.round((stats.avg_output_chars || 100) / 4);

    // Estimate training time: ~1 min per 50 examples on T4
    let trainingMin = Math.max(5, Math.round(totalExamples / 50));
    if (trainingMin > 30) trainingMin = '25-35';
    else if (trainingMin > 15) trainingMin = '15-25';
    else trainingMin = `${trainingMin}-${trainingMin + 5}`;

    html += '<div class="stats-estimate">';
    html += '<div class="estimate-row"><span class="estimate-label">Est. Training Time</span><span class="estimate-value">' + trainingMin + ' min</span></div>';
    html += '<div class="estimate-row"><span class="estimate-label">GPU Required</span><span class="estimate-value">Free T4 (Colab)</span></div>';
    html += '</div>';
    
    // Warnings if any
    if (stats.warnings && stats.warnings.length > 0) {
        html += '<div class="stats-warnings">';
        stats.warnings.forEach(w => {
            // Check if this is the system prompt warning
            if (w.includes('system prompt')) {
                html += `
                    <div class="warning-item with-action">
                        <span>⚠️ ${w}</span>
                        <button class="btn btn-secondary btn-small" id="openSystemPromptModal">
                            Add System Prompt
                        </button>
                    </div>`;
            } else {
                html += `<div class="warning-item">⚠️ ${w}</div>`;
            }
        });
        html += '</div>';
    }
    
    statsGrid.innerHTML = html;
    summaryCard.classList.remove('hidden');

    // Setup modal for system prompts if needed
    if (!stats.has_system_prompts && stats.total_examples >= 50) {
        setupSystemPromptModal(stats);

        // Add event listener to the button (it's dynamically created)
        setTimeout(() => {
            const modalBtn = document.getElementById('openSystemPromptModal');
            if (modalBtn) {
                modalBtn.addEventListener('click', () => openSystemPromptModal());
            }
        }, 100);
    }
}

// ============================================================================
// SYSTEM PROMPT FUNCTIONALITY
// ============================================================================

let selectedSystemPrompt = null;

function setupSystemPromptModal(stats) {
    // Just set up event listeners for the modal
    setupSystemPromptListeners();
}

function openSystemPromptModal() {
    const modal = document.getElementById('systemPromptModal');
    if (modal) {
        modal.classList.remove('hidden');
    }
}

function closeSystemPromptModal() {
    const modal = document.getElementById('systemPromptModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

function setupSystemPromptListeners() {
    // Close modal button
    const closeBtn = document.getElementById('closeModalBtn');
    if (closeBtn && !closeBtn.dataset.listenerAttached) {
        closeBtn.addEventListener('click', closeSystemPromptModal);
        closeBtn.dataset.listenerAttached = 'true';
    }

    // Cancel button
    const cancelBtn = document.getElementById('cancelPromptBtn');
    if (cancelBtn && !cancelBtn.dataset.listenerAttached) {
        cancelBtn.addEventListener('click', closeSystemPromptModal);
        cancelBtn.dataset.listenerAttached = 'true';
    }

    // Close modal when clicking outside
    const modal = document.getElementById('systemPromptModal');
    if (modal && !modal.dataset.listenerAttached) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeSystemPromptModal();
            }
        });
        modal.dataset.listenerAttached = 'true';
    }
    
    // Apply button
    const applyBtn = document.getElementById('applyPromptBtn');
    if (applyBtn && !applyBtn.dataset.listenerAttached) {
        applyBtn.addEventListener('click', async () => {
            // Check if sessionId exists
            if (!sessionId) {
                showError('No active session. Please upload a file first.');
                return;
            }

            const customInput = document.getElementById('customPromptInput');
            const prompt = customInput.value.trim();

            if (!prompt) {
                showError('Please enter a system prompt');
                return;
            }
            
            // Apply to dataset
            applyBtn.disabled = true;
            applyBtn.textContent = 'Applying...';
            
            try {
                console.log('Applying system prompt with sessionId:', sessionId);
                
                const response = await fetch(`${API_URL}/add-system-prompt`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        system_prompt: prompt
                    })
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(errorData.detail || `Server error: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('System prompt applied successfully:', data);
                
                // Update the stats display
                if (data.stats) {
                    showDatasetSummary(data.stats);
                }
                
                // Show success message with download option
                showSystemPromptSuccess(data.modified_count || 0, sessionId);

                // Close the modal
                closeSystemPromptModal();
                
            } catch (error) {
                console.error('Error applying system prompt:', error);
                showError(error.message || 'Failed to apply system prompt');
            } finally {
                applyBtn.disabled = false;
                applyBtn.textContent = 'Apply to Dataset';
            }
        });
        applyBtn.dataset.listenerAttached = 'true';
    }
}

// Task-specific prompt templates
const TASK_PROMPTS = {
    'classify': [
        {
            label: 'Classification Specialist',
            icon: '🎯',
            tagline: 'Optimized for accurate categorization',
            bestFor: 'Sentiment analysis, content categorization, intent detection',
            prompt: 'You are a classification model. Analyze inputs carefully and return the most appropriate category. Be consistent in your classifications and provide confident, direct responses.'
        },
        {
            label: 'Multi-Label Classifier',
            icon: '🏷️',
            tagline: 'For inputs with multiple categories',
            bestFor: 'Content tagging, feature detection, multi-class problems',
            prompt: 'You are a multi-label classification assistant. Analyze the input and identify all applicable categories. Maintain consistency in your labeling and explain your reasoning when multiple categories apply.'
        }
    ],
    'qa': [
        {
            label: 'Knowledge Expert',
            icon: '💡',
            tagline: 'Accurate answers with clear reasoning',
            bestFor: 'FAQ systems, knowledge bases, documentation',
            prompt: 'You are a knowledgeable assistant specializing in answering questions accurately. Provide clear, well-researched answers. If you\'re uncertain about something, acknowledge it. Include reasoning or sources when helpful.'
        },
        {
            label: 'Technical Support',
            icon: '🔧',
            tagline: 'Troubleshooting and problem-solving',
            bestFor: 'Customer support, technical assistance, debugging help',
            prompt: 'You are a technical support specialist. When answering questions, first understand the user\'s problem, then provide step-by-step solutions. Be patient and ask clarifying questions when needed.'
        }
    ],
    'conversation': [
        {
            label: 'Friendly Assistant',
            icon: '💬',
            tagline: 'Natural, engaging conversations',
            bestFor: 'Chatbots, customer service, virtual assistants',
            prompt: 'You are a friendly, conversational assistant. Engage naturally with users, maintain context throughout the conversation, and ask clarifying questions when needed. Be helpful while keeping responses concise.'
        },
        {
            label: 'Professional Consultant',
            icon: '👔',
            tagline: 'Formal, business-appropriate tone',
            bestFor: 'Enterprise chatbots, professional services, B2B interactions',
            prompt: 'You are a professional consultant assistant. Maintain a courteous and professional tone. Provide thorough, well-structured responses and prioritize accuracy and clarity in all communications.'
        }
    ],
    'generation': [
        {
            label: 'Creative Writer',
            icon: '✍️',
            tagline: 'Engaging, creative content',
            bestFor: 'Blog posts, marketing copy, creative writing',
            prompt: 'You are a creative content writer. Generate engaging, well-structured content that captures attention. Adapt your writing style to match the requested tone and format. Be original and compelling.'
        },
        {
            label: 'Technical Writer',
            icon: '📝',
            tagline: 'Clear, precise documentation',
            bestFor: 'Documentation, tutorials, technical guides',
            prompt: 'You are a technical writer. Create clear, precise, and well-organized content. Use appropriate technical terminology, provide step-by-step instructions when relevant, and ensure accuracy in all details.'
        }
    ],
    'extraction': [
        {
            label: 'Data Extraction Specialist',
            icon: '📊',
            tagline: 'Structured data extraction',
            bestFor: 'Entity extraction, form parsing, data structuring',
            prompt: 'You are a data extraction specialist. Analyze inputs and extract relevant information in a consistent, structured format. Be precise and thorough. If information is missing or unclear, indicate it explicitly.'
        },
        {
            label: 'Entity Recognition Expert',
            icon: '🔍',
            tagline: 'Identifying key entities and relationships',
            bestFor: 'Named entity recognition, relationship extraction',
            prompt: 'You are an entity recognition expert. Identify and extract key entities (people, organizations, locations, dates, etc.) from text. Maintain consistency in entity naming and format your output clearly.'
        }
    ]
};

function generateSuggestedPrompts(stats) {
    // Use ONLY dataset characteristics to suggest prompts (Step 1 happens before task selection)
    let prompts = [];

    const avgOutputTokens = Math.round((stats.avg_output_chars || 200) / 4);
    const isMultiTurn = (stats.multi_turn_pct || 0) > 50;
    const avgInputTokens = Math.round((stats.avg_input_chars || 100) / 4);

    // Detect task type from dataset patterns
    if (avgOutputTokens < 50) {
        // Short outputs suggest classification/extraction
        prompts.push(...TASK_PROMPTS.classify);
    } else if (isMultiTurn) {
        // Multi-turn conversations
        prompts.push(...TASK_PROMPTS.conversation);
    } else if (avgOutputTokens > 200) {
        // Long outputs suggest content generation
        prompts.push(...TASK_PROMPTS.generation);
                } else {
        // Medium outputs suggest Q&A
        prompts.push(...TASK_PROMPTS.qa);
    }

    // Always add a general-purpose option
    prompts.push({
        label: 'General Purpose',
        icon: '⚙️',
        tagline: 'Flexible assistant for various tasks',
        bestFor: 'Mixed use cases, general assistance',
        prompt: 'You are a helpful, accurate, and versatile assistant. Adapt your responses to the user\'s needs, provide clear explanations, and maintain a professional yet friendly tone.'
    });

    return prompts;
}

function hideSystemPromptCard() {
    const card = document.getElementById('systemPromptCard');
    if (card) card.classList.add('hidden');
}

function showSystemPromptSuccess(modifiedCount, sessionId) {
    // Create success banner
    const successBanner = document.createElement('div');
    successBanner.className = 'system-prompt-success';
    successBanner.innerHTML = `
        <div class="success-content">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            <span>System prompt added to ${modifiedCount} conversations</span>
        </div>
        <a href="${API_URL}/download-dataset/${sessionId}" class="btn btn-secondary btn-small" target="_blank">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Download Updated Dataset
        </a>
    `;
    
    // Insert after the system prompt card
    const promptCard = document.getElementById('systemPromptCard');
    if (promptCard && promptCard.parentNode) {
        promptCard.parentNode.insertBefore(successBanner, promptCard.nextSibling);
        
        // Auto-hide after 15 seconds
            setTimeout(() => {
            if (successBanner.parentNode) {
                successBanner.remove();
            }
        }, 15000);
    }
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
    hideSystemPromptCard();
    dropZone.style.display = '';
    continueBtn.disabled = true;
    resetContinueBtn(); // Reset button text when clearing file
    hideError();
}

// ============================================================================
// STEP 1 → STEP 2: Continue to Questions
// ============================================================================

continueBtn?.addEventListener('click', async () => {
    if (!sessionId) return;

    // Reset selections first
    selectedTask = null;
    selectedDeployment = null;

    // Reset all flow options
    document.querySelectorAll('.flow-option').forEach(opt => {
        opt.classList.remove('selected');
        opt.style.opacity = '';
        opt.style.pointerEvents = '';
    });

    // Clear ALL question texts and hide all questions BEFORE transitioning
    document.querySelectorAll('.flow-question').forEach(q => {
        q.classList.remove('active');
        const text = q.querySelector('.flow-question-text');
        if (text) {
            text.textContent = '';
            text.classList.remove('typing');
        }
    });

    // Move to questions step
    setStep(2);

    // Wait for step transition to complete, then show question
            setTimeout(() => {
        showQuestion('questionTask');
    }, 350);
});

// ============================================================================
// STEP 2: Task & Deployment Selection
// ============================================================================

// Task selection
document.querySelectorAll('.flow-option[data-task]').forEach(option => {
    option.addEventListener('click', () => {
        // Prevent if already selected
        if (option.classList.contains('selected')) return;

        // Reset all task options first
        document.querySelectorAll('.flow-option[data-task]').forEach(opt => {
            opt.classList.remove('selected');
            opt.style.opacity = '';
            opt.style.pointerEvents = '';
        });

        // Dim other options
        document.querySelectorAll('.flow-option[data-task]').forEach(opt => {
            if (opt !== option) {
                opt.style.opacity = '0.3';
                opt.style.pointerEvents = 'none';
            }
        });

        // Highlight selected option
        option.style.opacity = '1';
        option.classList.add('selected');
        selectedTask = option.dataset.task;

        // Fade out task question, fade in deployment question
        setTimeout(() => {
            hideQuestion('questionTask');
            setTimeout(() => {
                showQuestion('questionDeployment');
            }, 300);
        }, 500);
    });
});

// Deployment selection
document.querySelectorAll('.flow-option[data-deployment]').forEach(option => {
    option.addEventListener('click', async () => {
        // Prevent if already selected
        if (option.classList.contains('selected')) return;

        // Reset all deployment options first
        document.querySelectorAll('.flow-option[data-deployment]').forEach(opt => {
            opt.classList.remove('selected');
            opt.style.opacity = '';
            opt.style.pointerEvents = '';
        });

        // Dim other options
        document.querySelectorAll('.flow-option[data-deployment]').forEach(opt => {
            if (opt !== option) {
                opt.style.opacity = '0.3';
                opt.style.pointerEvents = 'none';
            }
        });

        // Highlight selected option
        option.style.opacity = '1';
        option.classList.add('selected');
        selectedDeployment = option.dataset.deployment;

        // Fade out deployment question, move to recommendation
        setTimeout(() => {
            hideQuestion('questionDeployment');
            setTimeout(() => {
                setStep(3);
                getRecommendation();
            }, 600);
        }, 500);
    });
});

// Back buttons
document.getElementById('backToUpload')?.addEventListener('click', () => {
    // Check if we're on question 2 (deployment)
    const deploymentQuestion = document.getElementById('questionDeployment');
    const taskQuestion = document.getElementById('questionTask');
    
    if (deploymentQuestion?.classList.contains('active')) {
        // Go back to question 1 (task)
        hideQuestion('questionDeployment');
        
        // Reset deployment selection
        selectedDeployment = null;
        document.querySelectorAll('.flow-option[data-deployment]').forEach(opt => {
            opt.classList.remove('selected');
            opt.removeAttribute('style');
        });
        
        // Reset task selection IMMEDIATELY and completely - do it NOW
        selectedTask = null;
        const taskOptions = document.querySelectorAll('.flow-option[data-task]');
        taskOptions.forEach(opt => {
            // Remove class
            opt.classList.remove('selected');
            // Remove ALL inline styles
            opt.removeAttribute('style');
            // Force reset computed styles by toggling
            opt.style.display = 'none';
            opt.offsetHeight; // Force reflow
            opt.style.display = '';
        });
        
        // Show task question with typewriter
            setTimeout(() => {
            showQuestion('questionTask');
            
            // Force reset AGAIN after question is visible to ensure it's clean
            setTimeout(() => {
                taskOptions.forEach(opt => {
                    opt.classList.remove('selected');
                    opt.removeAttribute('style');
                });
            }, 100);
        }, 200);
        } else {
        // Go back to step 1 (upload)
        setStep(1);
    }
});

document.getElementById('backToQuestions')?.addEventListener('click', () => {
    setStep(2);

    // Reset flow options state
    document.querySelectorAll('.flow-option').forEach(opt => {
        opt.style.opacity = '';
        opt.style.pointerEvents = '';
    });

    // Show deployment question (user was already past task) with typewriter
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
            const text = await response.text();
            try {
                const error = JSON.parse(text);
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = text || errorMsg;
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

// Store the original best match separately
let originalBestMatch = null;

function displayRecommendation(data) {
    const rec = data.recommendation?.primary_recommendation;
    
    if (!rec) {
        showError('No recommendation available');
        return;
    }
    
    // Store original best match (never changes)
    originalBestMatch = { ...rec, isOriginalBestMatch: true };
    
    // Store all models for later switching
    const alternatives = data.recommendation?.alternatives || [];
    allModelsData = [
        { ...rec, isOriginalBestMatch: true },
        ...alternatives.map(alt => ({ ...alt, isOriginalBestMatch: false }))
    ];
    
    // Set initial selected model to primary (which is also best match initially)
    selectedModelData = { ...rec, isOriginalBestMatch: true };
    
    // Display primary recommendation
    updatePrimaryDisplay(selectedModelData);
    
    // Render alternatives (excluding the selected model)
    updateAlternativesList();
}

function formatContextWindow(tokens) {
    if (!tokens) return '-';
    if (tokens >= 100000) {
        return `${Math.round(tokens / 1000)}K`;
    } else if (tokens >= 1000) {
        return `${Math.round(tokens / 1000)}K`;
    } else {
        return `${tokens}`;
    }
}

function updatePrimaryDisplay(model) {
    // Model name and size
    document.getElementById('recModelName').textContent = model.model_name || '-';
    document.getElementById('recModelSize').textContent = model.size || '-';
    
    // Score with animation
    const scorePercent = Math.round((model.score || 0) * 100);
    document.getElementById('recScore').textContent = scorePercent + '%';
    
    // Animate score ring
    const scoreCircle = document.getElementById('scoreCircle');
    scoreCircle.style.strokeDasharray = `${scorePercent}, 100`;
    
    // Reasons
    const reasonsContainer = document.getElementById('recReasons');
    reasonsContainer.innerHTML = '';
    (model.reasons || []).forEach(reason => {
        const div = document.createElement('div');
        div.className = 'rec-reason';
        div.innerHTML = `<span class="reason-icon">✓</span><span>${reason}</span>`;
        reasonsContainer.appendChild(div);
    });
    
    // Context Window - use formatted or format it
    const contextText = model.context_window_formatted || formatContextWindow(model.context_window || 0);
    document.getElementById('recContext').textContent = contextText;
    
    // Update badge based on whether this is the original best match
    const badge = document.querySelector('.rec-badge');
    if (badge) {
        if (model.isOriginalBestMatch) {
            badge.textContent = 'Best Match';
            badge.classList.remove('selected-alt');
        } else {
            badge.textContent = 'Your Selection';
            badge.classList.add('selected-alt');
        }
    }
}

function updateAlternativesList() {
    const altGrid = document.getElementById('alternativesGrid');
    altGrid.innerHTML = '';
    
    // Get all models except the currently selected one
    const currentModelId = selectedModelData?.model_id;
    const alternativesToShow = allModelsData.filter(m => m.model_id !== currentModelId);
    
    alternativesToShow.slice(0, 3).forEach((alt) => {
        const div = document.createElement('div');
        div.className = 'alt-card';
        
        // Add a special indicator if this is the original best match
        const isBestMatch = alt.isOriginalBestMatch;
        
        div.innerHTML = `
            <div class="alt-header">
                <div class="alt-name-container">
                    <span class="alt-name">${alt.model_name}</span>
                    ${isBestMatch ? '<span class="alt-best-match-tag">Best Match</span>' : ''}
                </div>
                <div class="alt-score-ring">
                    <svg viewBox="0 0 36 36" width="40" height="40">
                        <path class="score-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                        <path class="score-fill" stroke-dasharray="${Math.round((alt.score || 0) * 100)}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                    </svg>
                    <span class="alt-score-text">${Math.round((alt.score || 0) * 100)}%</span>
                </div>
            </div>
        `;
        
        // Add click handler to open modal
        div.addEventListener('click', () => openAltModelModal(alt));
        
        altGrid.appendChild(div);
    });
}

function openAltModelModal(model) {
    const modal = document.getElementById('altModelModal');
    
    // Populate modal
    document.getElementById('altModalModelName').textContent = model.model_name || '-';
    document.getElementById('altModalModelSize').textContent = model.size || '-';
    
    const scorePercent = Math.round((model.score || 0) * 100);
    document.getElementById('altModalScore').textContent = scorePercent + '%';
    document.getElementById('altModalScoreCircle').style.strokeDasharray = `${scorePercent}, 100`;
    
    // Reasons
    const reasonsContainer = document.getElementById('altModalReasons');
    reasonsContainer.innerHTML = '';
    (model.reasons || []).forEach(reason => {
        const div = document.createElement('div');
        div.className = 'modal-reason';
        div.innerHTML = `<span class="reason-icon">✓</span><span>${reason}</span>`;
        reasonsContainer.appendChild(div);
    });
    
    // Context
    document.getElementById('altModalContext').textContent = model.context_window_formatted || formatContextWindow(model.context_window || 0);
    
    // Store the model for confirmation
    modal.dataset.pendingModel = JSON.stringify(model);
    
    // Show modal
    modal.classList.remove('hidden');
}

function closeAltModelModal() {
    const modal = document.getElementById('altModelModal');
    modal.classList.add('hidden');
    modal.dataset.pendingModel = '';
}

function confirmAltModel() {
    const modal = document.getElementById('altModelModal');
    const modelData = JSON.parse(modal.dataset.pendingModel || '{}');
    
    if (modelData.model_name) {
        // Update selected model (preserve isOriginalBestMatch flag)
        selectedModelData = { ...modelData };
        
        // Update the primary display
        updatePrimaryDisplay(selectedModelData);
        
        // Update alternatives list (will now include the previous selection and exclude new one)
        updateAlternativesList();
        
        // Also update recommendationData so training uses the correct model
        if (recommendationData && recommendationData.recommendation) {
            recommendationData.recommendation.selected_model = selectedModelData;
        }
    }
    
    closeAltModelModal();
}

// Alt model modal event listeners
document.getElementById('altModalCancel')?.addEventListener('click', closeAltModelModal);
document.getElementById('altModalConfirm')?.addEventListener('click', confirmAltModel);

// Close modal on overlay click
document.getElementById('altModelModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'altModelModal') {
        closeAltModelModal();
    }
});

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

    // Show training options, hide result views
    document.getElementById('trainingOptions')?.classList.remove('hidden');
    document.getElementById('notebookReady')?.classList.add('hidden');
    document.getElementById('downloadPackage')?.classList.add('hidden');
});

document.getElementById('backToRec')?.addEventListener('click', () => {
    setStep(3);
});

// ============================================================================
// COLAB NOTEBOOK GENERATION
// ============================================================================

document.getElementById('openColabBtn')?.addEventListener('click', async () => {
    const btn = document.getElementById('openColabBtn');

    if (!sessionId || !recommendationData) {
        showError('Missing session data. Please start over.');
        return;
    }

    // Show loading
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Generating notebook...';
    btn.querySelector('.btn-loader')?.classList.remove('hidden');

    try {
        // First, create the plan (include selected model if user chose an alternative)
        const planBody = {
            session_id: sessionId,
            user_task: selectedTask,
            deployment_target: selectedDeployment
        };

        // If user selected an alternative model, include it
        if (selectedModelData && selectedModelData.model_id) {
            planBody.selected_model_id = selectedModelData.model_id;
        }

        const planResponse = await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(planBody)
        });

        if (!planResponse.ok) {
            throw new Error('Failed to create training plan');
        }

        // Generate Colab notebook
        const colabResponse = await fetch(`${API_URL}/generate-colab`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        if (!colabResponse.ok) {
            let errorMsg = 'Failed to generate notebook';
            const text = await colabResponse.text();
            try {
                const error = JSON.parse(text);
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = text || errorMsg;
            }
            throw new Error(errorMsg);
        }

        const data = await colabResponse.json();

        // Check if we have a direct Colab URL (via Gist)
        if (data.colab_url) {
            // Open Colab directly with notebook loaded!
            window.open(data.colab_url, '_blank');

            // Show success view
            document.getElementById('trainingOptions').classList.add('hidden');
            document.getElementById('notebookReady').classList.remove('hidden');

            // Update the UI to show it opened directly
            const notebookReady = document.getElementById('notebookReady');
            if (notebookReady) {
                notebookReady.querySelector('h2').textContent = 'Colab Opened!';
                notebookReady.querySelector('p').textContent = 'Your notebook is ready in the new tab';

                // Update instructions for direct open
                const instructions = notebookReady.querySelector('.colab-instructions');
                if (instructions) {
                    instructions.innerHTML = `
                        <div class="instruction-step">
                            <span class="step-num">1</span>
                            <span><strong>Runtime → Change runtime type → T4 GPU</strong></span>
                        </div>
                        <div class="instruction-step">
                            <span class="step-num">2</span>
                            <span>Click <strong>Runtime → Run all</strong></span>
                        </div>
                        <div class="instruction-step">
                            <span class="step-num">3</span>
                            <span>Wait for training to complete (~15 min)</span>
                        </div>
                    `;
                }
            }

            // Update buttons
            const downloadBtn = document.getElementById('downloadNotebookBtn');
            if (downloadBtn) {
                downloadBtn.href = `${API_URL}${data.notebook_url}`;
            }

            // Update "Open Colab Again" button to use direct URL
            const openColabAgainBtn = notebookReady.querySelector('a[href*="colab.research.google.com"]');
            if (openColabAgainBtn) {
                openColabAgainBtn.href = data.colab_url;
            }

        } else {
            // Fallback: Download notebook and open Colab manually
            const downloadUrl = `${API_URL}${data.notebook_url}`;
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = data.notebook_url.split('/').pop() || 'tunekit_training.ipynb';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Open Google Colab in new tab after short delay
            setTimeout(() => {
                window.open('https://colab.research.google.com/#create=true', '_blank');
            }, 500);

            // Show notebook ready view with upload instructions
            document.getElementById('trainingOptions').classList.add('hidden');
            document.getElementById('notebookReady').classList.remove('hidden');

            // Set download link for manual download
            const downloadBtn = document.getElementById('downloadNotebookBtn');
            if (downloadBtn) {
                downloadBtn.href = downloadUrl;
            }
        }

    } catch (error) {
        console.error('Colab generation error:', error);
        showError(error.message);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = 'Get Colab Notebook';
        btn.querySelector('.btn-loader')?.classList.add('hidden');
    }
});

// ============================================================================
// DOWNLOAD TRAINING PACKAGE
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
        // Create plan first (include selected model if user chose an alternative)
        const planBody = {
            session_id: sessionId,
            user_task: selectedTask,
            deployment_target: selectedDeployment
        };

        // If user selected an alternative model, include it
        if (selectedModelData && selectedModelData.model_id) {
            planBody.selected_model_id = selectedModelData.model_id;
        }

        const planResponse = await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(planBody)
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
            const text = await response.text();
            try {
                const error = JSON.parse(text);
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = text || errorMsg;
            }
            throw new Error(errorMsg);
        }

        const data = await response.json();

        // Show download section
        document.getElementById('trainingOptions').classList.add('hidden');
        document.getElementById('downloadPackage').classList.remove('hidden');

        // Setup download link
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.href = `${API_URL}${data.download_url}`;

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
    selectedModelData = null;
    allModelsData = [];
    originalBestMatch = null;

    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }

    // Reset file input
    clearFile();

    // Reset step 4 views
    document.getElementById('trainingOptions')?.classList.remove('hidden');
    document.getElementById('notebookReady')?.classList.add('hidden');
    document.getElementById('downloadPackage')?.classList.add('hidden');

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

// Initialize progress bar to step 1
setStep(1);

// ============================================================================
// TRAINING INFO TOGGLE
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    const trainingInfoToggle = document.getElementById('trainingInfoToggle');
    const trainingInfoCard = trainingInfoToggle?.closest('.training-info-card');
    
    if (trainingInfoToggle && trainingInfoCard) {
        trainingInfoToggle.addEventListener('click', () => {
            trainingInfoCard.classList.toggle('collapsed');
        });
    }
});
