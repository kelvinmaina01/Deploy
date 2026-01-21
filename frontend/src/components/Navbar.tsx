import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useSession } from '../context/SessionContext.tsx';
import { User, Bell, Settings, LogOut, ChevronDown, ArrowRight } from 'lucide-react';
import Logo from './Logo.tsx';

const Navbar: React.FC = () => {
    const { isAuthenticated, user, login, logout } = useSession();
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
                            <div className="nav-dropdown-trigger">
                                <span className="nav-text-link">Solutions</span>
                            </div>
                            <Link to="/pricing" className="nav-text-link">Pricing</Link>
                            <Link to="/dashboard" className="nav-text-link">Dashboard</Link>
                        </>
                    )}
                </div>

                <div className="nav-actions">
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
                        <Link to="/dashboard" className="btn-black" onClick={handleStartTraining}>
                            <span>Get Started</span>
                            <ArrowRight size={16} />
                        </Link>
                    )}
                </div>
            </div>
        </header>
    );
};

export default Navbar;
