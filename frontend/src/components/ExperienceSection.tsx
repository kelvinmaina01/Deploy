import React from 'react';
import { Calendar, Play, Zap, ArrowRight, CheckCircle2 } from 'lucide-react';
import { Link } from 'react-router-dom';

const ExperienceSection: React.FC = () => {
    return (
        <section className="section-container">
            <div className="section-header-centered">
                <span className="section-tag-premium">Experience</span>
                <h2 className="section-title-large">Deploy in Action</h2>
                <p className="section-subtitle-centered">
                    See how leading research teams are transforming their data analysis workflow. <br />
                    Choose the best way to get started.
                </p>
            </div>

            <div className="experience-grid">
                {/* Book a Demo */}
                <div className="experience-card glass">
                    <div className="experience-card-content">
                        <div className="experience-icon-row">
                            <div className="experience-icon-box blue">
                                <Calendar size={24} />
                            </div>
                        </div>
                        <h3 className="experience-card-title">Book a Demo</h3>
                        <p className="experience-card-text">
                            See Deploy's full capabilities in a personalized walkthrough with our team.
                        </p>
                        <ul className="experience-features">
                            <li><CheckCircle2 size={16} /> 30-minute personalized demo</li>
                            <li><CheckCircle2 size={16} /> Q&A with product experts</li>
                            <li><CheckCircle2 size={16} /> Custom use-case discussion</li>
                        </ul>
                    </div>
                    <a href="#" className="btn-experience outline">
                        Schedule Demo <ArrowRight size={16} />
                    </a>
                </div>

                {/* Watch Demo */}
                <div className="experience-card glass featured">
                    <div className="experience-card-content">
                        <div className="experience-icon-row">
                            <div className="experience-icon-box purple">
                                <Play size={24} />
                            </div>
                        </div>
                        <h3 className="experience-card-title">Watch Demo</h3>
                        <p className="experience-card-text">
                            See a quick walkthrough of how Deploy transforms research data into insights.
                        </p>
                        <ul className="experience-features">
                            <li><CheckCircle2 size={16} /> 5-minute overview video</li>
                            <li><CheckCircle2 size={16} /> Real use-case examples</li>
                            <li><CheckCircle2 size={16} /> Key features showcase</li>
                        </ul>
                    </div>
                    <a href="#" className="btn-experience primary">
                        Watch Now <Play size={16} fill="currentColor" />
                    </a>
                </div>

                {/* Start Free */}
                <div className="experience-card glass">
                    <div className="experience-card-content">
                        <div className="experience-icon-row">
                            <div className="experience-icon-box green">
                                <Zap size={24} />
                            </div>
                        </div>
                        <h3 className="experience-card-title">Start Free</h3>
                        <p className="experience-card-text">
                            Jump right in and explore Deploy with your own data. No credit card required.
                        </p>
                        <ul className="experience-features">
                            <li><CheckCircle2 size={16} /> Instant access</li>
                            <li><CheckCircle2 size={16} /> Upload your data immediately</li>
                            <li><CheckCircle2 size={16} /> Upgrade anytime</li>
                        </ul>
                    </div>
                    <Link to="/dashboard" className="btn-experience outline">
                        Get Started <ArrowRight size={16} />
                    </Link>
                </div>
            </div>
        </section>
    );
};

export default ExperienceSection;
