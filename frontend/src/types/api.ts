export interface DatasetStats {
    total_examples: number;
    total_messages: number;
    avg_messages_per_example: number;
    single_turn_pct: number;
    multi_turn_pct: number;
    avg_input_chars: number;
    avg_output_chars: number;
    system_prompt_pct: number;
    has_system_prompts: boolean;
    quality: 'excellent' | 'good' | 'minimal';
    warnings: string[];
}

export interface ModelRecommendation {
    model_id: string;
    model_name: string;
    size: string;
    score: number;
    reasons: string[];
    context_window: number;
    context_window_formatted?: string;
    isOriginalBestMatch?: boolean;
}

export interface RecommendationResponse {
    session_id: string;
    recommendation: {
        primary_recommendation: ModelRecommendation;
        alternatives: ModelRecommendation[];
    };
    analysis: {
        stats: DatasetStats;
        quality_score: number;
        quality_issues: string[];
        conversation_characteristics: any;
    };
}

export interface SessionResponse {
    session_id: string;
    status: string;
    message: string;
    rows?: number;
    stats?: DatasetStats;
}

export interface ColabResponse {
    session_id: string;
    notebook_url: string;
    message: string;
    status: string;
    colab_url?: string;
    gist_url?: string;
}

export interface PlanResponse {
    session_id: string;
    final_task_type: string;
    base_model: string;
    reasoning: string;
    training_config: any;
}
