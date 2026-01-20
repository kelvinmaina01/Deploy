import React from 'react';

const StatsBar: React.FC = () => {
    return (
        <section className="stats-bar">
            <div className="stat-item">
                <span className="stat-value">2x</span>
                <span className="stat-label">Faster Training</span>
            </div>
            <div className="stat-divider"></div>
            <div className="stat-item">
                <span className="stat-value">70%</span>
                <span className="stat-label">Less VRAM</span>
            </div>
            <div className="stat-divider"></div>
            <div className="stat-item">
                <span className="stat-value">$0</span>
                <span className="stat-label">Cloud Bills</span>
            </div>
            <div className="stat-divider"></div>
            <div className="stat-item">
                <span className="stat-value">&lt;15m</span>
                <span className="stat-label">To Production</span>
            </div>
        </section>
    );
};

export default StatsBar;
