import React, { useState, useEffect } from 'react';
import { useSession } from '../context/SessionContext.tsx';

const QuestionStep: React.FC = () => {
    const { selectedTask, setSelectedTask, selectedDeployment, setSelectedDeployment, setStep } = useSession();
    const [currentQuestion, setCurrentQuestion] = useState<'task' | 'deployment'>('task');
    const [typewriterText, setTypewriterText] = useState('');

    const taskQuestion = "What will this model do?";
    const deploymentQuestion = "Where will you deploy it?";

    useEffect(() => {
        const text = currentQuestion === 'task' ? taskQuestion : deploymentQuestion;
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
    }, [currentQuestion]);

    const handleTaskSelect = (task: string) => {
        setSelectedTask(task);
        setTimeout(() => setCurrentQuestion('deployment'), 300);
    };

    const handleDeploymentSelect = (deployment: string) => {
        setSelectedDeployment(deployment);
        setTimeout(() => setStep(3), 300);
    };

    return (
        <section className="step-content active">
            <div className="interactive-flow">
                {/* Question 1: Task Type */}
                <div className={`flow-question ${currentQuestion === 'task' ? 'active' : ''}`} style={{ display: currentQuestion === 'task' ? 'block' : 'none' }}>
                    <div className="flow-question-header">
                        <span className="question-number">1/2</span>
                        <h2 className="flow-question-text">{currentQuestion === 'task' ? typewriterText : taskQuestion}</h2>
                    </div>

                    <div className="flow-options five-items">
                        {[
                            { id: 'classify', title: 'Classify Text', desc: 'Sentiment, categories, spam' },
                            { id: 'qa', title: 'Answer Questions', desc: 'Q&A, knowledge, FAQ' },
                            { id: 'conversation', title: 'Conversations', desc: 'Chatbots, assistants' },
                            { id: 'generation', title: 'Generate Content', desc: 'Writing, summaries' },
                            { id: 'extraction', title: 'Extract Info', desc: 'Entities, parsing' }
                        ].map((task, i) => (
                            <button
                                key={task.id}
                                className={`flow-option ${selectedTask === task.id ? 'selected' : ''}`}
                                data-task={task.id}
                                style={{ '--delay': i } as React.CSSProperties}
                                onClick={() => handleTaskSelect(task.id)}
                            >
                                <span className="flow-option-title">{task.title}</span>
                                <span className="flow-option-desc">{task.desc}</span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Question 2: Deployment Target */}
                <div className={`flow-question ${currentQuestion === 'deployment' ? 'active' : ''}`} style={{ display: currentQuestion === 'deployment' ? 'block' : 'none' }}>
                    <div className="flow-question-header">
                        <span className="question-number">2/2</span>
                        <h2 className="flow-question-text">{currentQuestion === 'deployment' ? typewriterText : deploymentQuestion}</h2>
                    </div>

                    <div className="flow-options">
                        {[
                            { id: 'cloud_api', title: 'Cloud API', desc: 'AWS, GCP, Azure' },
                            { id: 'mobile_app', title: 'Mobile App', desc: 'iOS & Android' },
                            { id: 'edge_device', title: 'Edge Device', desc: 'Raspberry Pi, Jetson' },
                            { id: 'web_browser', title: 'Web Browser', desc: 'WebGPU, WASM' },
                            { id: 'desktop_app', title: 'Desktop App', desc: 'Windows, macOS, Linux' },
                            { id: 'not_sure', title: 'Not Sure', desc: 'Show all options' }
                        ].map((dep, i) => (
                            <button
                                key={dep.id}
                                className={`flow-option ${selectedDeployment === dep.id ? 'selected' : ''}`}
                                data-deployment={dep.id}
                                style={{ '--delay': i } as React.CSSProperties}
                                onClick={() => handleDeploymentSelect(dep.id)}
                            >
                                <span className="flow-option-title">{dep.title}</span>
                                <span className="flow-option-desc">{dep.desc}</span>
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            <button className="btn btn-ghost flow-back-btn" onClick={() => currentQuestion === 'deployment' ? setCurrentQuestion('task') : setStep(1)}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="19" y1="12" x2="5" y2="12"></line>
                    <polyline points="12 19 5 12 12 5"></polyline>
                </svg>
                Back
            </button>
        </section>
    );
};

export default QuestionStep;
