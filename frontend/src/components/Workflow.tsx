import React from 'react';
import { Upload, Search, Activity, Link2, Send } from 'lucide-react';

const steps = [
    { icon: <Upload size={20} />, label: "Upload your data" },
    { icon: <Search size={20} />, label: "We analyze and infer" },
    { icon: <Activity size={20} />, label: "Train on Colab" },
    { icon: <Link2 size={20} />, label: "Connect to a base LLM" },
    { icon: <Send size={20} />, label: "Deploy instantly" }
];

const Workflow: React.FC = () => {
    return (
        <section className="section-container">
            <div className="section-header-centered">
                <span className="section-tag-premium">Process</span>
                <h2 className="section-title-large">From data to deployed AI — end to end.</h2>
            </div>
            <div className="workflow-steps-horizontal">
                {steps.map((step, index) => (
                    <div key={index} className="workflow-step-item" style={{ animationDelay: `${index * 0.8}s` }}>
                        <div className="workflow-icon-box glass">
                            {step.icon}
                        </div>
                        <span className="workflow-label">{step.label}</span>
                        {index < steps.length - 1 && <div className="workflow-divider"></div>}
                    </div>
                ))}
            </div>
        </section>
    );
};

export default Workflow;
