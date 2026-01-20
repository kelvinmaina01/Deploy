import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';

const FinalCTA: React.FC = () => {
    return (
        <section className="section-container text-center">
            <div className="cta-premium-box glass">
                <h2 className="cta-premium-title">Build, link, and ship AI — fast.</h2>
                <p className="cta-subtitle">Sync your models directly to Hugging Face Hub automatically.</p>
                <div className="cta-premium-buttons">
                    <Link to="/dashboard" className="btn-premium btn-large">
                        <span>Start your project</span>
                        <ArrowRight size={20} />
                    </Link>
                </div>
            </div>
        </section>
    );
};

export default FinalCTA;
