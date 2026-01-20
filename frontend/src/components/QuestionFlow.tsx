import React, { useState, useEffect } from 'react';
import { Target, Cpu, Send, Layout, Search, ArrowRight, ArrowLeft } from 'lucide-react';
import { useSession } from '../context/SessionContext.tsx';

const tasks = [
    { id: 'classify', label: 'Classification', icon: Target, desc: 'Categorize inputs into predefined labels' },
    { id: 'qa', label: 'Q&A', icon: Send, desc: 'Answer questions based on provided context' },
    { id: 'conversation', label: 'Chatbot', icon: Cpu, desc: 'Natural back-and-forth multi-turn chat' },
    { id: 'generation', label: 'Content Gen', icon: Layout, desc: 'Writing blogs, articles, or creative text' },
    { id: 'extraction', label: 'Extraction', icon: Search, desc: 'Pulling specific data fields from text' },
];

const deployments = [
    { id: 'cloud_api', label: 'Cloud API', desc: 'Scalable backend for web/mobile apps' },
    { id: 'mobile_app', label: 'Mobile App', desc: 'Directly on iOS/Android devices' },
    { id: 'edge_device', label: 'Edge/IOT', desc: 'Embedded hardware with limited VRAM' },
    { id: 'web_browser', label: 'Browser', desc: 'Local inference via Transformers.js' },
    { id: 'desktop_app', label: 'Desktop', desc: 'Standalone Windows/Mac/Linux apps' },
    { id: 'not_sure', label: 'Not Sure', desc: 'Give me the overall best performer' },
];

const QuestionFlow: React.FC = () => {
    const {
        selectedTask, setSelectedTask,
        selectedDeployment, setSelectedDeployment,
        setStep
    } = useSession();

    const [currentSubStep, setCurrentSubStep] = useState(1);
    const [typewriterText, setTypewriterText] = useState('');

    const question1 = "What's the primary task your model will perform?";
    const question2 = "Where do you plan to deploy this model?";

    useEffect(() => {
        const text = currentSubStep === 1 ? question1 : question2;
        let i = 0;
        setTypewriterText('');
        const interval = setInterval(() => {
            if (i < text.length) {
                setTypewriterText(prev => prev + text[i]);
                i++;
            } else {
                clearInterval(interval);
            }
        }, 45);
        return () => clearInterval(interval);
    }, [currentSubStep]);

    const handleNext = () => {
        if (currentSubStep === 1 && selectedTask) {
            setCurrentSubStep(2);
        } else if (currentSubStep === 2 && selectedDeployment) {
            setStep(3);
        }
    };

    const handleBack = () => {
        if (currentSubStep === 2) {
            setCurrentSubStep(1);
        } else {
            setStep(1);
        }
    };

    return (
        <div className="step-wrapper">
            <div className="step-header">
                <span className="step-tag">Step 2</span>
                <h2 className="flow-question-text">{typewriterText}<span className="cursor">|</span></h2>
            </div>

            <div className="flow-options-grid">
                {currentSubStep === 1 ? (
                    tasks.map(task => (
                        <button
                            key={task.id}
                            className={`flow-option ${selectedTask === task.id ? 'selected' : ''}`}
                            onClick={() => setSelectedTask(task.id)}
                        >
                            <div className="option-icon"><task.icon size={24} /></div>
                            <div className="option-content">
                                <span className="option-label">{task.label}</span>
                                <span className="option-desc">{task.desc}</span>
                            </div>
                        </button>
                    ))
                ) : (
                    deployments.map(dep => (
                        <button
                            key={dep.id}
                            className={`flow-option ${selectedDeployment === dep.id ? 'selected' : ''}`}
                            onClick={() => setSelectedDeployment(dep.id)}
                        >
                            <div className="option-content">
                                <span className="option-label">{dep.label}</span>
                                <span className="option-desc">{dep.desc}</span>
                            </div>
                        </button>
                    ))
                )}
            </div>

            <div className="flow-actions">
                <button className="btn-secondary" onClick={handleBack}>
                    <ArrowLeft size={18} />
                    <span>Back</span>
                </button>
                <button
                    className="btn-primary"
                    disabled={currentSubStep === 1 ? !selectedTask : !selectedDeployment}
                    onClick={handleNext}
                >
                    <span>Continue</span>
                    <ArrowRight size={18} />
                </button>
            </div>
        </div>
    );
};

export default QuestionFlow;
