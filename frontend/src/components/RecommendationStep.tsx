import React, { useState, useEffect } from 'react';
import { useSession } from '../context/SessionContext.tsx';
import { api } from '../services/api.ts';

const RecommendationStep: React.FC = () => {
    const {
        sessionId, selectedTask, selectedDeployment,
        recommendationData, setRecommendationData,
        selectedModel, setSelectedModel,
        allModels, setStep
    } = useSession();

    const [isLoading, setIsLoading] = useState(!recommendationData);

    useEffect(() => {
        if (!recommendationData && sessionId && selectedTask && selectedDeployment) {
            const fetchRecommendation = async () => {
                setIsLoading(true);
                try {
                    const data = await api.getRecommendation(sessionId, selectedTask, selectedDeployment);
                    setRecommendationData(data);
                } catch (err) {
                    console.error(err);
                } finally {
                    setIsLoading(false);
                }
            };
            fetchRecommendation();
        }
    }, [sessionId, selectedTask, selectedDeployment, recommendationData, setRecommendationData]);

    if (isLoading) {
        return (
            <section className="step-content active">
                <div className="content-header centered">
                    <div className="header-badge">Step 3</div>
                    <h1>Model Recommendation</h1>
                    <p>Based on your dataset and requirements</p>
                </div>
                <div className="loading-state">
                    <div className="loading-spinner"></div>
                    <p>Analyzing your dataset...</p>
                </div>
            </section>
        );
    }

    if (!selectedModel) return null;

    const scorePercent = Math.round((selectedModel.score || 0.85) * 100);

    return (
        <section className="step-content active">
            <div className="content-header centered">
                <div className="header-badge">Step 3</div>
                <h1>Model Recommendation</h1>
                <p>Based on your dataset and requirements</p>
            </div>

            <div className="rec-content">
                <div className="rec-grid">
                    {/* Primary Recommendation */}
                    <div className="card primary-rec">
                        <div className="rec-badge">Best Match</div>
                        <div className="rec-header">
                            <div className="rec-model-info">
                                <h3 className="rec-model-name">{selectedModel.model_name}</h3>
                                <span className="rec-model-size">{selectedModel.size}</span>
                            </div>
                            <div className="rec-score-ring">
                                <svg viewBox="0 0 36 36">
                                    <defs>
                                        <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                            <stop offset="0%" style={{ stopColor: '#3b82f6' }} />
                                            <stop offset="100%" style={{ stopColor: '#2563eb' }} />
                                        </linearGradient>
                                    </defs>
                                    <path className="score-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                    <path
                                        className="score-fill"
                                        strokeDasharray={`${scorePercent}, 100`}
                                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                    />
                                </svg>
                                <span className="score-text">{scorePercent}%</span>
                            </div>
                        </div>
                        <div className="rec-reasons">
                            {selectedModel.reasons?.map((reason, i) => (
                                <div key={i} className="rec-reason">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <polyline points="20 6 9 17 4 12"></polyline>
                                    </svg>
                                    <span>{reason}</span>
                                </div>
                            ))}
                        </div>
                        <div className="rec-metrics">
                            <div className="rec-metric">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                    <polyline points="14 2 14 8 20 8"></polyline>
                                </svg>
                                <span>Context Window: <strong>{selectedModel.context_window || '8K'}</strong></span>
                            </div>
                        </div>
                    </div>

                    {/* Alternatives */}
                    <div className="card alternatives-card">
                        <h4 className="card-title">Alternative Models</h4>
                        <p className="alternatives-hint">Click a model to learn more</p>
                        <div className="alternatives-list">
                            {allModels.map((model) => (
                                <button
                                    key={model.model_id}
                                    className={`alternative-item ${selectedModel.model_id === model.model_id ? 'active' : ''}`}
                                    onClick={() => setSelectedModel(model)}
                                >
                                    <div className="alt-model-info">
                                        <span className="alt-model-name">{model.model_name}</span>
                                        <span className="alt-model-size">{model.size}</span>
                                    </div>
                                    <div className="alt-score">{Math.round((model.score || 0.8) * 100)}%</div>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            <div className="action-bar">
                <button className="btn btn-ghost" onClick={() => setStep(2)}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="19" y1="12" x2="5" y2="12"></line>
                        <polyline points="12 19 5 12 12 5"></polyline>
                    </svg>
                    Back
                </button>
                <button className="btn btn-primary" onClick={() => setStep(4)}>
                    Start Training
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                        <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                </button>
            </div>
        </section>
    );
};

export default RecommendationStep;
