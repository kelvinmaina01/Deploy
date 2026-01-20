import React, { useState } from 'react';
import { useSession } from '../context/SessionContext.tsx';
import { api } from '../services/api.ts';

const TrainingStep: React.FC = () => {
    const { sessionId, selectedModel, selectedTask, selectedDeployment, resetSession, setStep } = useSession();
    const [isGenerating, setIsGenerating] = useState(false);
    const [notebookReady, setNotebookReady] = useState(false);
    const [packageReady, setPackageReady] = useState(false);
    const [colabUrl, setColabUrl] = useState<string | null>(null);
    const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleColabGenerate = async () => {
        if (!sessionId) return;
        setIsGenerating(true);
        setError(null);
        try {
            await api.createPlan(sessionId, selectedTask!, selectedDeployment!, selectedModel?.model_id);
            const result = await api.generateColab(sessionId);
            setColabUrl(result.colab_url || null);
            setNotebookReady(true);
        } catch (err: any) {
            setError(err.message || 'Notebook generation failed');
        } finally {
            setIsGenerating(false);
        }
    };

    const handlePackageGenerate = async () => {
        if (!sessionId) return;
        setIsGenerating(true);
        setError(null);
        try {
            await api.createPlan(sessionId, selectedTask!, selectedDeployment!, selectedModel?.model_id);
            const result = await api.generatePackage(sessionId);
            setDownloadUrl(result.download_url);
            setPackageReady(true);
        } catch (err: any) {
            setError(err.message || 'Package generation failed');
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <section className="step-content active">
            <div className="content-header centered">
                <div className="header-badge">Step 4</div>
                <h1>Train Your Model</h1>
                <p>Choose how you want to train your model</p>
            </div>

            {/* Training Options */}
            {!notebookReady && !packageReady && (
                <div id="trainingOptions">
                    <div className="training-grid">
                        {/* Colab Option */}
                        <div className="card training-option">
                            <div className="training-icon">
                                <img src="https://colab.research.google.com/img/colab_favicon_256px.png" alt="Colab" width="48" height="48" />
                            </div>
                            <h3>Google Colab</h3>
                            <p>Train with free T4 GPU. No setup required.</p>
                            <ul className="training-features">
                                <li>Free T4 GPU (15GB VRAM)</li>
                                <li>Dataset embedded in notebook</li>
                                <li>Just click "Run All"</li>
                            </ul>
                            <button
                                className="btn btn-primary btn-full"
                                onClick={handleColabGenerate}
                                disabled={isGenerating}
                            >
                                <span className="btn-text">{isGenerating ? 'Generating...' : 'Get Colab Notebook'}</span>
                            </button>
                        </div>

                        {/* Download Package Option */}
                        <div className="card training-option">
                            <div className="training-icon">
                                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                    <polyline points="7 10 12 15 17 10"></polyline>
                                    <line x1="12" y1="15" x2="12" y2="3"></line>
                                </svg>
                            </div>
                            <h3>Download Package</h3>
                            <p>Get training scripts for your own hardware.</p>
                            <ul className="training-features">
                                <li>Full control over training</li>
                                <li>Use your own GPU</li>
                                <li>Customize as needed</li>
                            </ul>
                            <button
                                className="btn btn-secondary btn-full"
                                onClick={handlePackageGenerate}
                                disabled={isGenerating}
                            >
                                <span className="btn-text">{isGenerating ? 'Generating...' : 'Download Package'}</span>
                            </button>
                        </div>
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
                        </div>
                    )}

                    <div className="action-bar">
                        <button className="btn btn-ghost" onClick={() => setStep(3)}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="19" y1="12" x2="5" y2="12"></line>
                                <polyline points="12 19 5 12 12 5"></polyline>
                            </svg>
                            Back
                        </button>
                    </div>
                </div>
            )}

            {/* Notebook Ready */}
            {notebookReady && (
                <div id="notebookReady">
                    <div className="card success-card">
                        <div className="success-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                                <polyline points="22 4 12 14.01 9 11.01"></polyline>
                            </svg>
                        </div>
                        <h2>Notebook Ready!</h2>
                        <p>Your training notebook has been generated.</p>

                        <div className="notebook-actions">
                            {colabUrl && (
                                <a className="btn btn-primary" href={colabUrl} target="_blank" rel="noopener noreferrer">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                                        <polyline points="15 3 21 3 21 9"></polyline>
                                        <line x1="10" y1="14" x2="21" y2="3"></line>
                                    </svg>
                                    Open in Colab
                                </a>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Download Package Ready */}
            {packageReady && (
                <div id="downloadPackage">
                    <div className="card success-card">
                        <div className="success-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="7 10 12 15 17 10"></polyline>
                                <line x1="12" y1="15" x2="12" y2="3"></line>
                            </svg>
                        </div>
                        <h2>Package Ready!</h2>
                        <p>Your training package has been generated.</p>
                        <div className="package-contents">
                            <code>config.json</code>
                            <code>train.py</code>
                            <code>requirements.txt</code>
                            <code>README.md</code>
                        </div>
                        {downloadUrl && (
                            <a className="btn btn-primary" href={downloadUrl} download>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                    <polyline points="7 10 12 15 17 10"></polyline>
                                    <line x1="12" y1="15" x2="12" y2="3"></line>
                                </svg>
                                Download Package (.zip)
                            </a>
                        )}
                    </div>
                </div>
            )}

            <button className="btn btn-ghost start-over" onClick={resetSession}>Start New Project</button>
        </section>
    );
};

export default TrainingStep;
