import React from 'react';
import { Bot, BrainCircuit, LineChart, FileText, Eye, Hammer } from 'lucide-react';

const cases = [
    { icon: <Bot />, label: "AI agents" },
    { icon: <BrainCircuit />, label: "Decision engines" },
    { icon: <LineChart />, label: "Predictive analytics" },
    { icon: <FileText />, label: "NLP pipelines" },
    { icon: <Eye />, label: "Vision systems" },
    { icon: <Hammer />, label: "Internal AI tools" }
];

const UseCases: React.FC = () => {
    return (
        <section className="section-container">
            <div className="section-header-centered">
                <span className="section-tag-premium">Applications</span>
                <h2 className="section-title-large">Built for real-world systems.</h2>
            </div>
            <div className="use-cases-grid">
                {cases.map((c, index) => (
                    <div key={index} className="use-case-card glass">
                        <div className="use-case-icon">{c.icon}</div>
                        <span className="use-case-label">{c.label}</span>
                    </div>
                ))}
            </div>
        </section>
    );
};

export default UseCases;
