import React from 'react';
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const CTASection: React.FC = () => {
    return (
        <section className="cta-section">
            <div className="cta-card">
                <div className="cta-glow"></div>
                <h2>Ready to build?</h2>
                <p>Join developers using Deploy to ship fine-tuned models faster.</p>
                <Link to="/dashboard" className="btn-primary btn-large btn-glow">
                    <span>Open Dashboard</span>
                    <ArrowRight size={20} />
                </Link>
            </div>
        </section>
    );
};

export default CTASection;
