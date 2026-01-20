import type {
    SessionResponse,
    RecommendationResponse,
    PlanResponse,
    ColabResponse
} from '../types/api';

const API_BASE = ''; // Backend is served from the same host in production

export const api = {
    async uploadFile(file: File): Promise<SessionResponse> {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },

    async getRecommendation(
        sessionId: string,
        userTask: string,
        deploymentTarget: string
    ): Promise<RecommendationResponse> {
        const res = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_task: userTask,
                deployment_target: deploymentTarget,
            }),
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },

    async addSystemPrompt(sessionId: string, prompt: string): Promise<any> {
        const res = await fetch(`${API_BASE}/add-system-prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                system_prompt: prompt,
            }),
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },

    async createPlan(
        sessionId: string,
        userTask: string,
        deploymentTarget: string,
        selectedModelId?: string
    ): Promise<PlanResponse> {
        const res = await fetch(`${API_BASE}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_task: userTask,
                deployment_target: deploymentTarget,
                selected_model_id: selectedModelId,
            }),
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },

    async generateColab(sessionId: string): Promise<ColabResponse> {
        const res = await fetch(`${API_BASE}/generate-colab`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },

    async generatePackage(sessionId: string): Promise<{ download_url: string }> {
        const res = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },
};
