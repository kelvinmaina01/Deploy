import React from 'react';

const WhatIf: React.FC = () => {
    return (
        <section className="section-container text-center">
            <div className="what-if-wrapper">
                <span className="section-tag-premium">Concept</span>
                <h2 className="what-if-title">
                    What if you could just <span className="highlight-text-accent">upload data and ship real AI?</span>
                </h2>
                <h3 className="what-if-subtitle">
                    What if models trained where they already work best? <br />
                    Free-first training using Google Colab and optimized runtimes.
                </h3>
                <p className="what-if-description">
                    No pipelines to design. No infrastructure to manage. <br />
                    Just data in — deployed AI out.
                </p>
                <div className="divider-glow"></div>
            </div>
        </section>
    );
};

export default WhatIf;
