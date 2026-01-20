import React from 'react';
import { CheckCircle2 } from 'lucide-react';

const Philosophy: React.FC = () => {
    return (
        <section className="section-container">
            <div className="philosophy-box glass">
                <div className="philosophy-content">
                    <span className="section-tag-premium">Values</span>
                    <h2 className="philosophy-title">Free-first philosophy.</h2>
                    <h3 className="philosophy-subtitle">Start free. Stay in control.</h3>

                    <ul className="philosophy-list">
                        <li>
                            <CheckCircle2 size={20} className="text-success" />
                            <span>Google Colab by default</span>
                        </li>
                        <li>
                            <CheckCircle2 size={20} className="text-success" />
                            <span>Open models first</span>
                        </li>
                        <li>
                            <CheckCircle2 size={20} className="text-success" />
                            <span>No forced cloud lock-in</span>
                        </li>
                        <li>
                            <CheckCircle2 size={20} className="text-success" />
                            <span>Download everything you build</span>
                        </li>
                    </ul>
                </div>
                <div className="philosophy-visual">
                    {/* Abstract visual or icon */}
                    <div className="philosophy-blob"></div>
                </div>
            </div>
        </section>
    );
};

export default Philosophy;
