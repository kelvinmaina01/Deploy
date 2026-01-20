import React, { useState } from 'react';
import { useSession } from '../context/SessionContext.tsx';
import { api } from '../services/api.ts';

const UploadStep: React.FC = () => {
    const { setSessionId, setAnalysisData, setStep, analysisData } = useSession();
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [file, setFile] = useState<File | null>(null);

    const handleFileSelect = async (selectedFile: File) => {
        if (!selectedFile.name.endsWith('.jsonl')) {
            setError('Please upload a JSONL file. Other formats are not supported.');
            return;
        }
        setError(null);
        setFile(selectedFile);
        setIsUploading(true);

        try {
            const response = await api.uploadFile(selectedFile);
            setSessionId(response.session_id);
            setAnalysisData(response);
        } catch (err: any) {
            setError(err.message || 'Upload failed');
            setFile(null);
        } finally {
            setIsUploading(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) handleFileSelect(droppedFile);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.currentTarget.classList.remove('dragover');
    };

    const clearFile = () => {
        setFile(null);
        setSessionId(null);
        setAnalysisData(null);
    };

    return (
        <section className="step-content active">
            <div className="content-header">
                <div className="header-badge">Step 1</div>
                <h1>Upload Training Data</h1>
                <p>Import your JSONL dataset to begin fine-tuning</p>
            </div>

            <div className="content-grid">
                <div className="content-main">
                    <div className="card upload-card">
                        {!file ? (
                            <div
                                className="upload-zone"
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                                onClick={() => document.getElementById('fileInput')?.click()}
                            >
                                <div className="upload-visual">
                                    <div className="upload-icon-bg">
                                        <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                            <polyline points="17 8 12 3 7 8"></polyline>
                                            <line x1="12" y1="3" x2="12" y2="15"></line>
                                        </svg>
                                    </div>
                                </div>
                                <div className="upload-text">
                                    <span className="upload-main">Drop your JSONL file here</span>
                                    <span className="upload-sub">or <span className="upload-browse">browse files</span></span>
                                </div>
                                <input
                                    type="file"
                                    id="fileInput"
                                    accept=".jsonl"
                                    hidden
                                    onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                                />
                            </div>
                        ) : null}

                        <div className={`file-info ${!file ? 'hidden' : ''}`}>
                            <div className="file-icon-wrapper">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                    <polyline points="14 2 14 8 20 8"></polyline>
                                </svg>
                            </div>
                            <div className="file-details">
                                <div className="file-name">{file?.name || 'dataset.jsonl'}</div>
                                <div className="file-meta">
                                    <span className="file-stat success">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <polyline points="20 6 9 17 4 12"></polyline>
                                        </svg>
                                        {analysisData?.rows || '--'} examples
                                    </span>
                                    <span className="file-stat">{file ? `${(file.size / 1024).toFixed(1)} KB` : '-- KB'}</span>
                                </div>
                            </div>
                            <button className="btn-remove" onClick={clearFile} title="Remove file">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <line x1="18" y1="6" x2="6" y2="18"></line>
                                    <line x1="6" y1="6" x2="18" y2="18"></line>
                                </svg>
                            </button>
                        </div>

                        {file && analysisData && (
                            <div className="validation-status">
                                <div className="validation-icon">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <polyline points="20 6 9 17 4 12"></polyline>
                                    </svg>
                                </div>
                                <div className="validation-details">
                                    <div className="validation-title">Format Validated</div>
                                    <div className="validation-meta">OpenAI Chat format detected</div>
                                </div>
                            </div>
                        )}

                        {file && analysisData && (
                            <div className="dataset-summary-card">
                                <div className="summary-header">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <line x1="18" y1="20" x2="18" y2="10"></line>
                                        <line x1="12" y1="20" x2="12" y2="4"></line>
                                        <line x1="6" y1="20" x2="6" y2="14"></line>
                                    </svg>
                                    <span>Analysis</span>
                                </div>
                                <div className="summary-stats-grid">
                                    <div className="stats-primary">
                                        <div className="primary-stat">
                                            <span className="primary-value">{analysisData.rows}</span>
                                            <span className="primary-label">Examples</span>
                                        </div>
                                        <div className="primary-stat">
                                            <span className="primary-value">{analysisData.stats?.avg_turns?.toFixed(1) || '--'}</span>
                                            <span className="primary-label">Avg Turns</span>
                                        </div>
                                        <div className="primary-stat">
                                            <span className="primary-value">{analysisData.stats?.avg_user_tokens?.toFixed(0) || '--'}</span>
                                            <span className="primary-label">Avg User Tokens</span>
                                        </div>
                                        <div className="primary-stat">
                                            <span className="primary-value">{analysisData.stats?.avg_assistant_tokens?.toFixed(0) || '--'}</span>
                                            <span className="primary-label">Avg Asst Tokens</span>
                                        </div>
                                    </div>
                                    <div className="stats-details">
                                        <div className="detail-row">
                                            <span className="detail-label">Has System Prompts</span>
                                            <span className="detail-value">{analysisData.stats?.has_system_prompts ? 'Yes' : 'No'}</span>
                                        </div>
                                        <div className="detail-row">
                                            <span className="detail-label">Quality</span>
                                            <span className={`quality-badge ${analysisData.stats?.quality === 'Good' ? 'success' : ''}`}>
                                                {analysisData.stats?.quality || 'Checking...'}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {error && (
                        <div className="error-banner">
                            <div className="error-content">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="12" r="10"></circle>
                                    <line x1="12" y1="8" x2="12" y2="12"></line>
                                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                                </svg>
                                <span className="error-text">{error}</span>
                            </div>
                            <button className="error-close" onClick={() => setError(null)}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <line x1="18" y1="6" x2="6" y2="18"></line>
                                    <line x1="6" y1="6" x2="18" y2="18"></line>
                                </svg>
                            </button>
                        </div>
                    )}
                </div>

                <aside className="content-sidebar">
                    <div className="card info-card">
                        <div className="info-header">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="16" x2="12" y2="12"></line>
                                <line x1="12" y1="8" x2="12.01" y2="8"></line>
                            </svg>
                            <span>Format Requirements</span>
                        </div>
                        <ul className="requirements-list">
                            <li><span className="req-check">✓</span><span>JSONL format with <code>messages</code> array</span></li>
                            <li><span className="req-check">✓</span><span>Minimum 50 examples <span className="req-note">(100-1000 recommended)</span></span></li>
                            <li><span className="req-check">✓</span><span>Each message: <code>role</code> and <code>content</code></span></li>
                            <li><span className="req-check">✓</span><span>Valid roles: <code>system</code>, <code>user</code>, <code>assistant</code></span></li>
                        </ul>
                    </div>

                    <div className="card example-card">
                        <div className="info-header">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <polyline points="16 18 22 12 16 6"></polyline>
                                <polyline points="8 6 2 12 8 18"></polyline>
                            </svg>
                            <span>Example Format</span>
                        </div>
                        <pre className="code-block"><code>{`{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}`}</code></pre>
                    </div>
                </aside>
            </div>

            <div className="action-bar">
                <button
                    className="btn btn-primary"
                    disabled={!file || isUploading || !analysisData}
                    onClick={() => setStep(2)}
                >
                    {isUploading ? 'Uploading...' : 'Continue'}
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                        <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                </button>
            </div>
        </section>
    );
};

export default UploadStep;
