import React, { createContext, useContext, useState, useEffect } from 'react';
import type { ModelRecommendation, RecommendationResponse } from '../types/api';

interface SessionState {
    sessionId: string | null;
    currentStep: number;
    analysisData: any | null;
    recommendationData: RecommendationResponse | null;
    selectedModel: ModelRecommendation | null;
    allModels: ModelRecommendation[];
    selectedTask: string | null;
    selectedDeployment: string | null;
    theme: 'light' | 'dark';
    isAuthenticated: boolean;
    user: { name: string; email: string; avatar?: string } | null;
}

interface SessionContextType extends SessionState {
    setSessionId: (id: string | null) => void;
    setStep: (step: number) => void;
    setAnalysisData: (data: any) => void;
    setRecommendationData: (data: RecommendationResponse) => void;
    setSelectedModel: (model: ModelRecommendation) => void;
    setSelectedTask: (task: string | null) => void;
    setSelectedDeployment: (target: string | null) => void;
    toggleTheme: () => void;
    resetSession: () => void;
    login: () => void;
    logout: () => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export const SessionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [state, setState] = useState<SessionState>({
        sessionId: null,
        currentStep: 1,
        analysisData: null,
        recommendationData: null,
        selectedModel: null,
        allModels: [],
        selectedTask: null,
        selectedDeployment: null,
        theme: (localStorage.getItem('deploy-theme') as 'light' | 'dark') ||
            (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
        isAuthenticated: false,
        user: null,
    });

    useEffect(() => {
        const html = document.documentElement;
        if (state.theme === 'dark') {
            html.removeAttribute('data-theme');
        } else {
            html.setAttribute('data-theme', 'light');
        }
        localStorage.setItem('deploy-theme', state.theme);
    }, [state.theme]);

    const setSessionId = (id: string | null) => setState(s => ({ ...s, sessionId: id }));
    const setStep = (step: number) => setState(s => ({ ...s, currentStep: step }));
    const setAnalysisData = (data: any) => setState(s => ({ ...s, analysisData: data }));

    const setRecommendationData = (data: RecommendationResponse) => {
        const primary = data.recommendation.primary_recommendation;
        const alternatives = data.recommendation.alternatives || [];
        setState(s => ({
            ...s,
            recommendationData: data,
            selectedModel: { ...primary, isOriginalBestMatch: true },
            allModels: [
                { ...primary, isOriginalBestMatch: true },
                ...alternatives.map(alt => ({ ...alt, isOriginalBestMatch: false }))
            ]
        }));
    };

    const setSelectedModel = (model: ModelRecommendation) => setState(s => ({ ...s, selectedModel: model }));
    const setSelectedTask = (task: string | null) => setState(s => ({ ...s, selectedTask: task }));
    const setSelectedDeployment = (target: string | null) => setState(s => ({ ...s, selectedDeployment: target }));
    const toggleTheme = () => setState(s => ({ ...s, theme: s.theme === 'light' ? 'dark' : 'light' }));

    const login = () => setState(s => ({
        ...s,
        isAuthenticated: true,
        user: { name: 'Alex Researcher', email: 'alex@labiq.ai', avatar: '/avatar-placeholder.png' }
    }));

    const logout = () => setState(s => ({
        ...s,
        isAuthenticated: false,
        user: null
    }));

    const resetSession = () => setState({
        sessionId: null,
        currentStep: 1,
        analysisData: null,
        recommendationData: null,
        selectedModel: null,
        allModels: [],
        selectedTask: null,
        selectedDeployment: null,
        theme: state.theme,
        isAuthenticated: state.isAuthenticated,
        user: state.user
    });

    return (
        <SessionContext.Provider value={{
            ...state,
            setSessionId,
            setStep,
            setAnalysisData,
            setRecommendationData,
            setSelectedModel,
            setSelectedTask,
            setSelectedDeployment,
            toggleTheme,
            resetSession,
            login,
            logout
        }}>
            {children}
        </SessionContext.Provider>
    );
};

export const useSession = () => {
    const context = useContext(SessionContext);
    if (!context) throw new Error('useSession must be used within a SessionProvider');
    return context;
};
