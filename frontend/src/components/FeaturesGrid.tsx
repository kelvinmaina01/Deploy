import React from 'react';
import { Rocket, Box, Link2, ShieldCheck, Zap, BarChart3 } from 'lucide-react';

const features = [
    {
        icon: <Rocket className="feature-icon-accent" />,
        title: "Fast-First Training",
        description: "Leverage Unsloth and Google Colab for training that’s 2× faster with up to 70% less memory usage."
    },
    {
        icon: <Box className="feature-icon-accent" />,
        title: "Automated ML Ingestion",
        description: "From JSONL to CSV, we infer task types, detect targets, and validate data quality automatically."
    },
    {
        icon: <Link2 className="feature-icon-accent" />,
        title: "Model Connection",
        description: "Orchestrate your models. Link fine-tuned SLMs with base LLMs for task-specific reasoning."
    },
    {
        icon: <ShieldCheck className="feature-icon-accent" />,
        title: "Enterprise Ready",
        description: "Privacy-first by design. High-performance models. Your infrastructure. Your control."
    },
    {
        icon: <Zap className="feature-icon-accent" />,
        title: "One-Click Deployment",
        description: "Deploy to Hugging Face Spaces or export production-ready APIs instantly."
    },
    {
        icon: <BarChart3 className="feature-icon-accent" />,
        title: "Real-Time Monitoring",
        description: "Track loss, accuracy, and performance as your model trains in the cloud."
    }
];

const FeaturesGrid: React.FC = () => {
    return (
        <section className="section-container">
            <div className="section-header-centered">
                <span className="section-tag-premium">Capabilities</span>
                <h2 className="section-title-large">Enterprise-ready AI tools</h2>
            </div>
            <div className="features-bento-grid">
                {features.map((feature, index) => (
                    <div key={index} className="feature-card-premium glass">
                        <div className="feature-icon-wrapper-alt">
                            {feature.icon}
                        </div>
                        <h3 className="feature-title-alt">{feature.title}</h3>
                        <p className="feature-description-alt">{feature.description}</p>
                    </div>
                ))}
            </div>
        </section>
    );
};

export default FeaturesGrid;
