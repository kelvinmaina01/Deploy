import React from 'react';
import { ArrowRight, Github, Check, Terminal as TerminalIcon } from 'lucide-react';
import { Link } from 'react-router-dom';

const Hero: React.FC = () => {
    return (
        <section className="hero-centered">
            <div className="hero-content-full">
                <div className="badge-row justify-center">
                    <span className="badge badge-premium">
                        <span className="badge-dot"></span>
                        Free during Beta
                    </span>
                </div>

                <h1 className="hero-title-main">
                    <span className="hero-line-1">Turn raw data into</span><br />
                    <span className="hero-line-2 gradient-text">production-ready</span><br />
                    <span className="hero-line-3 typing-container">
                        <span className="typing-text">AI systems without writing code.</span>
                    </span>
                </h1>

                <p className="hero-subtitle-large">
                    Upload your dataset, and we'll build, train, and deploy your model automatically. <br />
                    <span className="highlight-white">No coding. No guesswork. No infrastructure costs.</span><br />
                    Get a production-ready model in under 15 minutes.
                </p>

                <div className="hero-cta-centered">
                    <Link to="/dashboard" className="btn-premium">
                        <span>Start Your Project</span>
                        <ArrowRight size={18} />
                    </Link>
                    <a href="https://github.com/riyanshibohra/Deploy" className="btn-outline" target="_blank" rel="noopener noreferrer">
                        <Github size={18} />
                        <span>Star on GitHub</span>
                    </a>
                </div>
            </div>

            {/* Terminal Preview Below */}
            <div className="hero-visual-container">
                <div className="terminal-glass">
                    <div className="terminal-header-dark">
                        <div className="terminal-dots">
                            <span className="dot dot-red"></span>
                            <span className="dot dot-yellow"></span>
                            <span className="dot dot-green"></span>
                        </div>
                        <div className="terminal-tab">
                            <TerminalIcon size={12} className="mr-2" />
                            <span>deploy.sh</span>
                        </div>
                    </div>
                    <div className="terminal-body-dark">
                        <div className="code-line">
                            <span className="code-muted">$</span> <span className="code-cmd">deploy analyze</span> <span className="code-string">./data/*</span>
                        </div>
                        <div className="code-line"></div>
                        <div className="code-line code-output-dim">
                            <span className="code-stage">[STAGE 1]</span> Detecting file formats and structure...
                        </div>
                        <div className="code-line code-output-dim">
                            <span className="code-stage">[STAGE 2]</span> Inferring task: <span className="code-highlight">Classification</span> (text + metadata)
                        </div>
                        <div className="code-line code-output-dim">
                            <span className="code-stage">[STAGE 3]</span> Recommending model: <span className="code-highlight">Llama-3.2-3B</span> (Unsloth optimized)
                        </div>
                        <div className="code-line"></div>
                        <div className="code-line code-ready">
                            <span className="code-success"><Check size={14} style={{ display: 'inline' }} /></span> READY: Start training on Google Colab
                        </div>
                        <div className="terminal-cursor-alt"></div>
                    </div>
                </div>
                <div className="hero-glow-under"></div>
            </div>
        </section>
    );
};

export default Hero;
