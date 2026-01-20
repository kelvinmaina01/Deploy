import React from 'react';
import Navbar from './Navbar.tsx';

interface LayoutProps {
    children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
    return (
        <div className="app-wrapper">
            <Navbar />
            <main className="main-content">
                {children}
            </main>
        </div>
    );
};

export default Layout;
