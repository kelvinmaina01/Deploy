import React from 'react';
import { Github, Twitter, Linkedin } from 'lucide-react';
import { Link } from 'react-router-dom';

const LandingFooter: React.FC = () => {
    return (
        <footer className="footer-flow">
            <div className="footer-content">
                <div className="footer-grid">
                    <div className="footer-brand">
                        <div className="footer-logo-group">
                            <span className="footer-logo-text">Deploy</span>
                        </div>
                        <p className="footer-tagline">
                            Turn raw data into production-ready AI systems.
                        </p>
                        <div className="footer-socials">
                            <a href="#" className="social-link"><Twitter size={20} /></a>
                            <a href="#" className="social-link"><Github size={20} /></a>
                            <a href="#" className="social-link"><Linkedin size={20} /></a>
                        </div>
                    </div>

                    <div className="footer-links-column">
                        <h4>Product</h4>
                        <Link to="/models">Models</Link>
                        <Link to="/features">Features</Link>
                        <Link to="/pricing">Pricing</Link>
                    </div>

                    <div className="footer-links-column">
                        <h4>Resources</h4>
                        <Link to="/docs">Documentation</Link>
                        <Link to="/blog">Blog</Link>
                        <Link to="/community">Community</Link>
                    </div>

                    <div className="footer-links-column">
                        <h4>Company</h4>
                        <Link to="/about">About</Link>
                        <Link to="/careers">Careers</Link>
                        <Link to="/legal">Legal</Link>
                    </div>
                </div>

                <div className="footer-bottom">
                    <p>&copy; {new Date().getFullYear()} Deploy AI. All rights reserved.</p>
                </div>
            </div>
        </footer>
    );
};

export default LandingFooter;
