import React from 'react';
import { Github, Twitter, Linkedin, Mail, ArrowUp } from 'lucide-react';
import { Link } from 'react-router-dom';
import Logo from './Logo.tsx';

const DetailedFooter: React.FC = () => {
    const scrollToTop = () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    return (
        <footer className="detailed-footer">
            <div className="footer-top">
                <div className="footer-brand-column">
                    <Logo />
                    <p className="footer-tagline">
                        AI-powered research intelligence platform transforming how teams analyze data and make discoveries.
                    </p>
                    <div className="footer-socials">
                        <a href="#" aria-label="Twitter" className="social-link"><Twitter size={18} /></a>
                        <a href="https://github.com/riyanshibohra/Deploy" aria-label="GitHub" className="social-link"><Github size={18} /></a>
                        <a href="#" aria-label="LinkedIn" className="social-link"><Linkedin size={18} /></a>
                        <a href="mailto:hello@deploy.ai" aria-label="Email" className="social-link"><Mail size={18} /></a>
                    </div>
                </div>

                <div className="footer-links-grid">
                    <div className="footer-column">
                        <h4 className="footer-heading">Product</h4>
                        <ul className="footer-list">
                            <li><Link to="/dashboard">Dashboard</Link></li>
                            <li><Link to="/dashboard">Upload Data</Link></li>
                            <li><Link to="/models">AI Models</Link></li>
                            <li><Link to="/pricing">Pricing</Link></li>
                        </ul>
                    </div>
                    <div className="footer-column">
                        <h4 className="footer-heading">Features</h4>
                        <ul className="footer-list">
                            <li><Link to="/features/analytics">Analytics</Link></li>
                            <li><Link to="/features/insights">AI Insights</Link></li>
                            <li><Link to="/features/reports">Reports</Link></li>
                            <li><Link to="/features/automation">Automation</Link></li>
                        </ul>
                    </div>
                    <div className="footer-column">
                        <h4 className="footer-heading">Company</h4>
                        <ul className="footer-list">
                            <li><Link to="/about">About Us</Link></li>
                            <li><Link to="/blog">Blog</Link></li>
                            <li><Link to="/careers">Careers</Link></li>
                            <li><Link to="/contact">Contact</Link></li>
                        </ul>
                    </div>
                </div>
            </div>

            <div className="footer-bottom">
                <div className="footer-bottom-content">
                    <p className="footer-copyright">
                        &copy; {new Date().getFullYear()} <strong>Deploy AI</strong>. All rights reserved.
                    </p>
                    <div className="footer-legal-links">
                        <Link to="/privacy">Privacy Policy</Link>
                        <Link to="/terms">Terms of Service</Link>
                        <Link to="/security">Security</Link>
                    </div>
                    <button className="back-to-top" onClick={scrollToTop} aria-label="Back to top">
                        <ArrowUp size={20} />
                    </button>
                </div>
            </div>
        </footer>
    );
};

export default DetailedFooter;
