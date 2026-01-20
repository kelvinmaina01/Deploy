import React from 'react';
import { ExternalLink } from 'lucide-react';

const models = [
    {
        name: 'Phi-4 Mini',
        org: 'MICROSOFT',
        description: 'Best for classification, extraction & structured output. Excellent reasoning.',
        size: '3.8B',
        link: 'https://huggingface.co/microsoft/Phi-4-mini-instruct'
    },
    {
        name: 'Llama 3.2',
        org: 'META',
        description: 'Top performer for Q&A and conversational AI. Great context tracking.',
        size: '3B',
        link: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct'
    },
    {
        name: 'Mistral 7B',
        org: 'MISTRAL AI',
        description: 'Best for long-form generation and complex reasoning. Strong knowledge base.',
        size: '7B',
        link: 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3'
    },
    {
        name: 'Qwen 2.5',
        org: 'ALIBABA',
        description: 'Strong multilingual support. Great for JSON and structured outputs.',
        size: '3B',
        link: 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct'
    },
    {
        name: 'Gemma 2',
        org: 'GOOGLE',
        description: 'Smallest and fastest. Ideal for edge devices and mobile deployment.',
        size: '2B',
        link: 'https://huggingface.co/google/gemma-2-2b-it'
    }
];

const ModelList: React.FC = () => {
    return (
        <section className="models">
            <div className="section-header">
                <span className="section-tag">Supported Models</span>
                <h2 className="section-title">Train the best open models</h2>
            </div>

            <div className="models-list">
                {models.map((model, i) => (
                    <a key={i} href={model.link} target="_blank" rel="noopener noreferrer" className="model-row">
                        <div className="model-info">
                            <div className="model-header">
                                <span className="model-name">{model.name}</span>
                                <span className="model-badge">{model.org}</span>
                            </div>
                            <p className="model-desc">{model.description}</p>
                        </div>
                        <div className="model-meta">
                            <span className="model-size">{model.size}</span>
                            <ExternalLink size={16} className="model-link-icon" />
                        </div>
                    </a>
                ))}
            </div>
        </section>
    );
};

export default ModelList;
