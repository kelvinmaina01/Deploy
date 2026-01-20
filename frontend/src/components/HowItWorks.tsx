import React from 'react';
import { Upload, Settings, Zap, ArrowRight } from 'lucide-react';

const HowItWorks: React.FC = () => {
    return (
        <section className="how-it-works">
            <div className="section-header">
                <span className="section-tag">Simple Process</span>
                <h2 className="section-title">Three steps to your custom model</h2>
            </div>

            <div className="steps">
                <div className="step">
                    <div className="step-icon step-icon-accent">
                        <Upload size={24} />
                    </div>
                    <div className="step-number">01</div>
                    <h3>Upload Data</h3>
                    <p>Drop your JSONL file. We validate format and analyze patterns automatically.</p>
                </div>

                <div className="step-arrow">
                    <ArrowRight size={40} strokeWidth={1.5} />
                </div>

                <div className="step">
                    <div className="step-icon step-icon-accent">
                        <Settings size={24} />
                    </div>
                    <div className="step-number">02</div>
                    <h3>Configure</h3>
                    <p>AI recommends the best model and hyperparameters for your specific task.</p>
                </div>

                <div className="step-arrow">
                    <ArrowRight size={40} strokeWidth={1.5} />
                </div>

                <div className="step">
                    <div className="step-icon step-icon-accent">
                        <Zap size={24} />
                    </div>
                    <div className="step-number">03</div>
                    <h3>Train</h3>
                    <p>One-click Colab notebook. Hit "Run All" and get your model in 15 minutes.</p>
                </div>
            </div>
        </section>
    );
};

export default HowItWorks;
