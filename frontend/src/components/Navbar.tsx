import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useSession } from '../context/SessionContext.tsx';
import { User, Bell, Settings, LogOut, ChevronDown, Sparkles } from 'lucide-react';
import Logo from './Logo.tsx';

const Navbar: React.FC = () => {
    const { toggleTheme, currentStep, isAuthenticated, user, login, logout } = useSession();
    const [showUserMenu, setShowUserMenu] = useState(false);
    const location = useLocation();
    const isDashboard = location.pathname === '/dashboard';

    const handleStartTraining = (e: React.MouseEvent) => {
        if (!isAuthenticated) {
            e.preventDefault();
            login();
        }
    };

    return (
        <header className="top-nav">
            <div className="nav-container">
                <Logo />

                <div className="nav-main-links">
                    {!isDashboard && (
                        <>
                            <Link to="/models" className="nav-text-link">Models</Link>
                            <Link to="/pricing" className="nav-text-link">Pricing</Link>
                            <Link to="/docs" className="nav-text-link">Docs</Link>
                        </>
                    )}
                </div>

                {isDashboard && (
                    <nav className="nav-steps">
                        <div className="progress-bar-container">
                            {[
                                { step: 1, label: 'Upload Data' },
                                { step: 2, label: 'Configure' },
                                { step: 3, label: 'Recommendation' },
                                { step: 4, label: 'Training' }
                            ].map((item, index) => (
                                <React.Fragment key={item.step}>
                                    <div
                                        className={`progress-step-nav ${currentStep > item.step ? 'completed' : ''} ${currentStep === item.step ? 'active' : ''}`}
                                        data-step={item.step}
                                    >
                                        <div className="progress-dot-nav">
                                            <svg className="check-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                                <polyline points="20 6 9 17 4 12"></polyline>
                                            </svg>
                                            <div className="pulse-ring-nav"></div>
                                        </div>
                                        <span className="step-label-nav">{item.label}</span>
                                    </div>
                                    {index < 3 && (
                                        <div className={`progress-connector ${currentStep > item.step ? 'completed' : ''}`}></div>
                                    )}
                                </React.Fragment>
                            ))}
                        </div>
                    </nav>
                )}

                <div className="nav-actions">
                    <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
                        <svg className="sun-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="5"></circle>
                            <line x1="12" y1="1" x2="12" y2="3"></line>
                            <line x1="12" y1="21" x2="12" y2="23"></line>
                            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                            <line x1="1" y1="12" x2="3" y2="12"></line>
                            <line x1="21" y1="12" x2="23" y2="12"></line>
                            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                        </svg>
                        <svg className="moon-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                        </svg>
                    </button>

                    {isAuthenticated ? (
                        <div className="user-menu-container">
                            <button
                                className="user-profile-btn"
                                onClick={() => setShowUserMenu(!showUserMenu)}
                            >
                                <div className="user-avatar">
                                    <img src={user?.avatar || '/avatar-placeholder.png'} alt={user?.name} />
                                </div>
                                <span className="user-name">{user?.name.split(' ')[0]}</span>
                                <ChevronDown size={14} className={`chevron ${showUserMenu ? 'open' : ''}`} />
                            </button>

                            {showUserMenu && (
                                <div className="user-dropdown glass animate-in">
                                    <div className="dropdown-header">
                                        <span className="dropdown-title">My Account</span>
                                        <span className="user-email">{user?.email}</span>
                                    </div>
                                    <div className="dropdown-divider"></div>
                                    <Link to="/profile" className="dropdown-item">
                                        <User size={16} />
                                        <span>Profile</span>
                                    </Link>
                                    <Link to="/notifications" className="dropdown-item">
                                        <Bell size={16} />
                                        <span>Notifications</span>
                                        <span className="badge-new">3</span>
                                    </Link>
                                    <Link to="/settings" className="dropdown-item">
                                        <Settings size={16} />
                                        <span>Settings</span>
                                    </Link>
                                    <div className="dropdown-divider"></div>
                                    <button onClick={logout} className="dropdown-item logout">
                                        <LogOut size={16} />
                                        <span>Sign Out</span>
                                    </button>
                                </div>
                            )}
                        </div>
                    ) : (
                        <Link to="/dashboard" className="btn btn-primary" onClick={handleStartTraining}>
                            <span>Start Training</span>
                            <Sparkles size={16} />
                        </Link>
                    )}
                </div>
            </div>
        </header>
    );
};

export default Navbar;
