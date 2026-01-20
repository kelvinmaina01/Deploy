import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { useSession } from '../context/SessionContext';

const ThemeToggle: React.FC = () => {
    const { theme, toggleTheme } = useSession();

    return (
        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
            {theme === 'dark' ? (
                <Sun className="sun-icon" size={18} />
            ) : (
                <Moon className="moon-icon" size={18} />
            )}
        </button>
    );
};

export default ThemeToggle;
