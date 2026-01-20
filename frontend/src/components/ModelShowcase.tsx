import React from 'react';

const models = [
    {
        name: "Phi-4 Mini",
        creator: "MICROSOFT",
        description: "Best for classification, extraction & structured output. Excellent reasoning.",
        size: "3.8B"
    },
    {
        name: "Llama 3.2",
        creator: "META",
        description: "Top performer for Q&A and conversational AI. Great context tracking.",
        size: "3B"
    },
    {
        name: "Mistral 7B",
        creator: "MISTRAL AI",
        description: "Best for long-form generation and complex reasoning. Strong knowledge base.",
        size: "7B"
    },
    {
        name: "Qwen 2.5",
        creator: "ALIBABA",
        description: "Strong multilingual support. Great for JSON and structured outputs.",
        size: "3B"
    },
    {
        name: "Gemma 2",
        creator: "GOOGLE",
        description: "Smallest and fastest. Ideal for edge devices and mobile deployment.",
        size: "2B"
    }
];

const ModelShowcase: React.FC = () => {
    return (
        <section className="section-container">
            <div className="section-header-centered">
                <span className="section-tag-premium">Model Zoo</span>
                <h2 className="section-title-large">Train the best open models</h2>
            </div>
            <div className="model-grid-curated">
                {models.map((model, index) => (
                    <div key={index} className="model-card-premium glass">
                        <div className="model-card-header">
                            <span className="model-creator">{model.creator}</span>
                            <span className="model-size-badge">{model.size}</span>
                        </div>
                        <h3 className="model-name-alt">{model.name}</h3>
                        <p className="model-description-alt">{model.description}</p>
                    </div>
                ))}
            </div>
        </section>
    );
};

export default ModelShowcase;
