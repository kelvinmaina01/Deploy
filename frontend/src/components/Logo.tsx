import React from 'react';
import { Link } from 'react-router-dom';

const Logo: React.FC = () => {
    return (
        <Link to="/" className="logo-group">
            <img src="/logo.png" alt="Deploy" className="logo-image-large" />
            <span className="logo-brand-text">Deploy</span>
        </Link>
    );
};

export default Logo;
