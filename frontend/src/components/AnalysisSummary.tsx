import React from 'react';
import { AlertTriangle, ShieldCheck, Clock, Zap } from 'lucide-react';
import { useSession } from '../context/SessionContext.tsx';

const AnalysisSummary: React.FC = () => {
    const { analysisData } = useSession();

    if (!analysisData || !analysisData.stats) return null;

    const { stats } = analysisData;
    const qualityClass = stats.quality === 'excellent' ? 'success' : stats.quality === 'good' ? 'accent' : 'warning';

    return (
        <div className="dashboard-card dataset-summary-card">
            <div className="card-header">
                <div className="validation-status">
                    <div className="status-icon success">
                        <ShieldCheck size={20} />
                    </div>
                    <div>
                        <h3>Format Validated</h3>
                        <p>Dataset ready for analysis</p>
                    </div>
                </div>
            </div>

            <div className="summary-stats-grid">
                <div className="stats-details">
                    <div className="detail-row">
                        <span className="detail-label">Conversation Type</span>
                        <span className="detail-value">{stats.single_turn_pct}% single-turn, {stats.multi_turn_pct}% multi-turn</span>
                    </div>
                    <div className="detail-row">
                        <span className="detail-label">System Prompts</span>
                        <span className="detail-value">{stats.system_prompt_pct > 0 ? `${stats.system_prompt_pct}% of examples` : 'None'}</span>
                    </div>
                    <div className="detail-row">
                        <span className="detail-label">Avg Response Length</span>
                        <span className="detail-value">~{Math.round(stats.avg_output_chars / 4)} tokens</span>
                    </div>
                    <div className="detail-row">
                        <span className="detail-label">Dataset Quality</span>
                        <span className={`detail-value quality-badge ${qualityClass}`}>
                            {stats.quality.charAt(0).toUpperCase() + stats.quality.slice(1)}
                        </span>
                    </div>
                </div>

                <div className="stats-estimate">
                    <div className="estimate-row">
                        <Clock size={16} />
                        <span className="estimate-label">Est. Training Time</span>
                        <span className="estimate-value">15-20 min</span>
                    </div>
                    <div className="estimate-row">
                        <Zap size={16} />
                        <span className="estimate-label">GPU Required</span>
                        <span className="estimate-value">Free T4 (Colab)</span>
                    </div>
                </div>

                {stats.warnings && stats.warnings.length > 0 && (
                    <div className="stats-warnings">
                        {stats.warnings.map((w: string, i: number) => (
                            <div key={i} className="warning-item">
                                <AlertTriangle size={14} />
                                <span>{w}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default AnalysisSummary;
