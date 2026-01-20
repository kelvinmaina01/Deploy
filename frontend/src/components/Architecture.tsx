import React from 'react';
import { Database, Cpu, Brain, Zap } from 'lucide-react';

const Architecture: React.FC = () => {
    return (
        <section className="section-container">
            <div className="architecture-box glass">
                <div className="architecture-content">
                    <span className="section-tag-premium">Differentiator</span>
                    <h2 className="architecture-title">Separation of truth and reasoning.</h2>
                    <p className="architecture-copy">
                        Deploy AI never lets LLMs invent predictions. <br />
                        <strong>Models generate truth. LLMs interpret it.</strong>
                    </p>
                </div>

                <div className="architecture-diagram">
                    <div className="diag-node">
                        <div className="diag-icon-wrapper blue"><Database size={24} /></div>
                        <span>Input Data</span>
                    </div>
                    <div className="diag-arrow">↓</div>
                    <div className="diag-node">
                        <div className="diag-icon-wrapper purple"><Cpu size={24} /></div>
                        <span>ML Model (Predictions)</span>
                    </div>
                    <div className="diag-arrow">↓</div>
                    <div className="diag-node">
                        <div className="diag-icon-wrapper indigo"><Brain size={24} /></div>
                        <span>LLM (Reasoning + Interface)</span>
                    </div>
                    <div className="diag-arrow">↓</div>
                    <div className="diag-node">
                        <div className="diag-icon-wrapper success"><Zap size={24} /></div>
                        <span>Human-readable Output</span>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Architecture;
