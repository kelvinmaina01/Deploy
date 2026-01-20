import React from 'react';
import { Zap, Layout, Download } from 'lucide-react';

const Features: React.FC = () => {
    return (
        <section className="features">
            <div className="section-header">
                <span className="section-tag">Why Deploy</span>
                <h2 className="section-title">From your data to your model, effortlessly.</h2>
            </div>

            <div className="features-grid">
                <div className="feature-card">
                    <div className="feature-icon feature-icon-purple">
                        <Zap size={24} />
                    </div>
                    <h3>2x Faster Training</h3>
                    <p>Powered by Unsloth optimization. Train models in half the time on free GPUs.</p>
                </div>

                <div className="feature-card">
                    <div className="feature-icon feature-icon-colab">
                        <img src="https://colab.research.google.com/img/colab_favicon_256px.png" alt="Colab" width="28" height="28" />
                    </div>
                    <h3>Free GPU Training</h3>
                    <p>Train on Google Colab's free T4 GPU. No credit card or setup required.</p>
                </div>

                <div className="feature-card">
                    <div className="feature-icon feature-icon-blue">
                        <Layout size={24} />
                    </div>
                    <h3>Smart Model Selection</h3>
                    <p>AI analyzes your data and recommends the best model for your task.</p>
                </div>

                <div className="feature-card">
                    <div className="feature-icon feature-icon-green">
                        <Download size={24} />
                    </div>
                    <h3>Export Anywhere</h3>
                    <p>GGUF for Ollama, merged weights for HuggingFace, or LoRA adapters.</p>
                </div>
            </div>
        </section>
    );
};

export default Features;
