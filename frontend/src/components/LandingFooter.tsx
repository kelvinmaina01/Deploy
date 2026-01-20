import React from 'react';
import { Github } from 'lucide-react';
import { Link } from 'react-router-dom';

const LandingFooter: React.FC = () => {
    return (
        <footer className="footer-minimal glass">
            <div className="footer-minimal-content">
                <div className="footer-minimal-left">
                    <Link to="/" className="footer-logo">
                        <img src="/logo.png" alt="Deploy" width="24" height="24" />
                        <span>Deploy</span>
                    </Link>
                </div>

                <div className="footer-minimal-right">
                    <div className="footer-minimal-links">
                        <a href="https://github.com/riyanshibohra/Deploy" target="_blank" rel="noopener noreferrer">
                            <Github size={16} />
                            <span>GitHub</span>
                        </a>
                        <Link to="/dashboard">Dashboard</Link>
                    </div>
                </div>
            </div>
            <div className="footer-minimal-bottom">
                &copy; {new Date().getFullYear()} Deploy AI. All rights reserved.
            </div>
        </footer>
    );
};

export default LandingFooter;
