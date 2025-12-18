"""
Model Recommendation Tool
=========================
Simple 3-factor scoring: Task (50) + Size (30) + Output (20) = 100 points
"""

from typing import Dict, List

# ════════════════════════════════════════════════════════════════════════════
# MODEL METADATA
# ════════════════════════════════════════════════════════════════════════════

MODELS = {
    'phi-4-mini': {
        'id': 'microsoft/Phi-4-mini-instruct',
        'name': 'Phi-4 Mini',
        'size': '3.8B',
        'training_time_base': 3,
        'cost_base': 0.18,
        'accuracy_baseline': 87
    },
    'gemma-3-2b': {
        'id': 'google/gemma-3-2b-it',
        'name': 'Gemma 3 2B',
        'size': '2B',
        'training_time_base': 2,
        'cost_base': 0.12,
        'accuracy_baseline': 82
    },
    'llama-3.2-3b': {
        'id': 'meta-llama/Llama-3.2-3B-Instruct',
        'name': 'Llama 3.2 3B',
        'size': '3B',
        'training_time_base': 4,
        'cost_base': 0.20,
        'accuracy_baseline': 89
    },
    'qwen-2.5-3b': {
        'id': 'Qwen/Qwen2.5-3B-Instruct',
        'name': 'Qwen 2.5 3B',
        'size': '3B',
        'training_time_base': 4,
        'cost_base': 0.20,
        'accuracy_baseline': 88
    },
    'mistral-7b': {
        'id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'name': 'Mistral 7B',
        'size': '7B',
        'training_time_base': 6,
        'cost_base': 0.35,
        'accuracy_baseline': 91
    }
}

# ════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT FILTERS
# ════════════════════════════════════════════════════════════════════════════

DEPLOYMENT_FILTERS = {
    'cloud_api': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'qwen-2.5-3b', 'mistral-7b'],
    'desktop_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'qwen-2.5-3b', 'mistral-7b'],
    'mobile_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],
    'ios_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],
    'android_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],
    'web_browser': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],
    'edge_device': ['gemma-3-2b', 'phi-4-mini'],
    'not_sure': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'qwen-2.5-3b', 'mistral-7b']
}

# ════════════════════════════════════════════════════════════════════════════
# SCORING TABLES
# ════════════════════════════════════════════════════════════════════════════

# Factor 1: TASK TYPE (50 points max)
TASK_SCORES = {
    'classify': {
        'phi-4-mini': 50,
        'gemma-3-2b': 35,
        'llama-3.2-3b': 30,
        'qwen-2.5-3b': 25,
        'mistral-7b': 20
    },
    'qa': {
        'llama-3.2-3b': 50,
        'mistral-7b': 45,
        'phi-4-mini': 35,
        'qwen-2.5-3b': 30,
        'gemma-3-2b': 20
    },
    'conversation': {
        'llama-3.2-3b': 50,
        'mistral-7b': 45,
        'qwen-2.5-3b': 35,
        'phi-4-mini': 25,
        'gemma-3-2b': 20
    },
    'generation': {
        'mistral-7b': 50,
        'llama-3.2-3b': 45,
        'qwen-2.5-3b': 30,
        'phi-4-mini': 20,
        'gemma-3-2b': 20
    },
    'extraction': {
        'phi-4-mini': 50,
        'llama-3.2-3b': 40,
        'mistral-7b': 35,
        'qwen-2.5-3b': 30,
        'gemma-3-2b': 25
    }
}

# Factor 2: DATASET SIZE (30 points max)
SIZE_SCORES_SMALL = {  # < 500 examples
    'gemma-3-2b': 30,
    'phi-4-mini': 20,
    'llama-3.2-3b': 10,
    'qwen-2.5-3b': 10,
    'mistral-7b': 5
}

SIZE_SCORES_MEDIUM = {  # 500-2000 examples
    'llama-3.2-3b': 30,
    'phi-4-mini': 25,
    'mistral-7b': 20,
    'qwen-2.5-3b': 20,
    'gemma-3-2b': 15
}

SIZE_SCORES_LARGE = {  # > 2000 examples
    'llama-3.2-3b': 30,
    'mistral-7b': 25,
    'phi-4-mini': 10,
    'qwen-2.5-3b': 10,
    'gemma-3-2b': 5
}

# Factor 3: OUTPUT CHARACTERISTICS (20 points max)
OUTPUT_SCORES_LONG = {  # avg_response_length > 200
    'mistral-7b': 20,
    'llama-3.2-3b': 15,
    'qwen-2.5-3b': 10,
    'phi-4-mini': 5,
    'gemma-3-2b': 5
}

OUTPUT_SCORES_JSON = {  # JSON output detected
    'phi-4-mini': 20,
    'llama-3.2-3b': 15,
    'mistral-7b': 10,
    'qwen-2.5-3b': 10,
    'gemma-3-2b': 5
}


# ════════════════════════════════════════════════════════════════════════════
# MAIN RECOMMENDATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def recommend_model(
    user_task: str,
    conversation_characteristics: Dict,
    num_examples: int,
    deployment_target: str = 'not_sure'
) -> Dict:
    """
    Recommend best SLM model using simple 3-factor scoring.
    
    Args:
        user_task: 'classify', 'qa', 'conversation', 'generation', 'extraction'
        conversation_characteristics: Dict with is_multilingual, avg_response_length, looks_like_json_output
        num_examples: Dataset size
        deployment_target: 'cloud_api', 'mobile_app', 'edge_device', etc.
    
    Returns:
        Dict with primary_recommendation, alternatives, all_scores
    """
    
    # Extract characteristics with safe defaults
    is_multilingual = conversation_characteristics.get('is_multilingual', False)
    avg_response_length = conversation_characteristics.get('avg_response_length', 50)
    looks_like_json = conversation_characteristics.get('looks_like_json_output', False)
    is_multi_turn = conversation_characteristics.get('is_multi_turn', False)
    
    print(f"\n🎯 Model Recommendation")
    print(f"   Task: {user_task}")
    print(f"   Examples: {num_examples}")
    print(f"   Deployment: {deployment_target}")
    print(f"   Multilingual: {is_multilingual}")
    print(f"   Multi-turn: {is_multi_turn}")
    print(f"   Avg response: {avg_response_length} chars")
    print(f"   JSON output: {looks_like_json}")
    
    # ════════════════════════════════════════════════════════════
    # STEP 1: HARD OVERRIDES (Skip scoring for obvious cases)
    # ════════════════════════════════════════════════════════════
    
    if is_multilingual:
        print("\n⚡ OVERRIDE: Multilingual detected → Qwen 2.5 3B")
        return build_response(
            primary_key='qwen-2.5-3b',
            score=100,
            all_scores={'qwen-2.5-3b': 100},
            reasons=['Best multilingual support (29 languages)', 'Optimized for non-English text'],
            num_examples=num_examples
        )
    
    if deployment_target == 'edge_device':
        print("\n⚡ OVERRIDE: Edge device → Gemma 3 2B")
        return build_response(
            primary_key='gemma-3-2b',
            score=100,
            all_scores={'gemma-3-2b': 100, 'phi-4-mini': 80},
            reasons=['Smallest model (2B params)', 'Optimized for low-power devices'],
            num_examples=num_examples,
            alternatives=[{'model': 'phi-4-mini', 'score': 80}]
        )
    
    # ════════════════════════════════════════════════════════════
    # STEP 2: FILTER MODELS BY DEPLOYMENT
    # ════════════════════════════════════════════════════════════
    
    allowed_models = DEPLOYMENT_FILTERS.get(deployment_target, DEPLOYMENT_FILTERS['not_sure'])
    print(f"\n📱 Allowed models for {deployment_target}: {allowed_models}")
    
    # ════════════════════════════════════════════════════════════
    # STEP 3: SCORE EACH MODEL (3 factors)
    # ════════════════════════════════════════════════════════════
    
    scores = {}
    score_breakdown = {}
    
    for model_key in allowed_models:
        task_score = 0
        size_score = 0
        output_score = 0
        multi_turn_bonus = 0
        
        # Factor 1: TASK TYPE (50 points)
        task_scores = TASK_SCORES.get(user_task, TASK_SCORES['classify'])
        task_score = task_scores.get(model_key, 25)
        
        # Factor 2: DATASET SIZE (30 points)
        if num_examples < 500:
            size_score = SIZE_SCORES_SMALL.get(model_key, 15)
        elif num_examples > 2000:
            size_score = SIZE_SCORES_LARGE.get(model_key, 15)
        else:
            size_score = SIZE_SCORES_MEDIUM.get(model_key, 15)
        
        # Factor 3: OUTPUT CHARACTERISTICS (20 points)
        if avg_response_length > 200:
            output_score = OUTPUT_SCORES_LONG.get(model_key, 10)
        elif looks_like_json:
            output_score = OUTPUT_SCORES_JSON.get(model_key, 10)
        else:
            output_score = 10  # Neutral
        
        # BONUS: Multi-turn conversations (+10 points)
        if is_multi_turn:
            MULTI_TURN_BONUS = {
                'llama-3.2-3b': 10,    # Best at context tracking
                'mistral-7b': 8,       # Good at multi-turn
                'qwen-2.5-3b': 6,      # Decent
                'phi-4-mini': 3,       # Okay
                'gemma-3-2b': 2        # Weak
            }
            multi_turn_bonus = MULTI_TURN_BONUS.get(model_key, 0)
        
        total = task_score + size_score + output_score + multi_turn_bonus
        scores[model_key] = total
        score_breakdown[model_key] = {
            'task': task_score,
            'size': size_score,
            'output': output_score,
            'multi_turn': multi_turn_bonus,
            'total': total
        }
        
        max_score = 110 if is_multi_turn else 100
        print(f"\n   {MODELS[model_key]['name']}: {total}/{max_score}")
        bonus_str = f", Multi-turn: +{multi_turn_bonus}" if is_multi_turn else ""
        print(f"      Task: {task_score}/50, Size: {size_score}/30, Output: {output_score}/20{bonus_str}")
    
    # ════════════════════════════════════════════════════════════
    # STEP 4: PICK WINNER
    # ════════════════════════════════════════════════════════════
    
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_key = sorted_models[0][0]
    primary_score = sorted_models[0][1]
    
    print(f"\n✅ Winner: {MODELS[primary_key]['name']} ({primary_score}/100)")
    
    # Generate reasons
    reasons = generate_reasons(primary_key, user_task, num_examples, avg_response_length, looks_like_json, is_multi_turn, deployment_target)
    
    # Build alternatives
    alternatives = []
    for model_key, score in sorted_models[1:3]:
        model = MODELS[model_key]
        alt_reasons = []
        if score >= primary_score - 10:
            alt_reasons.append(f"Close match ({score}/100)")
        if model['cost_base'] < MODELS[primary_key]['cost_base']:
            alt_reasons.append(f"Lower cost (${model['cost_base']})")
        if model['training_time_base'] < MODELS[primary_key]['training_time_base']:
            alt_reasons.append(f"Faster training")
        alternatives.append({
            'model': model_key,
            'score': score,
            'reasons': alt_reasons if alt_reasons else ['Good alternative']
        })
    
    return build_response(
        primary_key=primary_key,
        score=primary_score,
        all_scores=scores,
        reasons=reasons,
        num_examples=num_examples,
        alternatives=alternatives
    )


def generate_reasons(model_key: str, task: str, num_examples: int, avg_response: int, is_json: bool, is_multi_turn: bool, deployment: str) -> List[str]:
    """Generate human-readable reasons for the recommendation."""
    reasons = []
    model = MODELS[model_key]
    
    # Task-based reason
    task_reasons = {
        ('phi-4-mini', 'classify'): 'Best for classification tasks',
        ('phi-4-mini', 'extraction'): 'Excellent at structured extraction',
        ('llama-3.2-3b', 'qa'): 'Top performer for Q&A tasks',
        ('llama-3.2-3b', 'conversation'): 'Best for conversational AI',
        ('mistral-7b', 'generation'): 'Best for long-form generation',
        ('gemma-3-2b', 'classify'): 'Fast and efficient for classification',
    }
    reason = task_reasons.get((model_key, task))
    if reason:
        reasons.append(reason)
    else:
        reasons.append(f"Strong performance for {task} tasks")
    
    # Multi-turn reason
    if is_multi_turn and model_key in ['llama-3.2-3b', 'mistral-7b']:
        reasons.append('Excellent at multi-turn context tracking')
    
    # Size-based reason
    if num_examples < 500:
        if model_key == 'gemma-3-2b':
            reasons.append('Ideal for small datasets')
        elif model_key == 'phi-4-mini':
            reasons.append('Works well with limited data')
    elif num_examples > 2000:
        if model_key in ['llama-3.2-3b', 'mistral-7b']:
            reasons.append(f'Scales well with {num_examples:,} examples')
    
    # Output-based reason
    if avg_response > 200 and model_key in ['mistral-7b', 'llama-3.2-3b']:
        reasons.append('Optimized for longer outputs')
    elif is_json and model_key == 'phi-4-mini':
        reasons.append('Excellent JSON/structured output')
    
    # Deployment reason
    if deployment in ['mobile_app', 'ios_app', 'android_app'] and model_key in ['phi-4-mini', 'gemma-3-2b']:
        reasons.append(f'Optimized for mobile deployment')
    elif deployment == 'web_browser' and model_key in ['gemma-3-2b', 'phi-4-mini']:
        reasons.append('Runs efficiently in browser')
    
    return reasons[:4]  # Max 4 reasons


def build_response(
    primary_key: str,
    score: int,
    all_scores: Dict,
    reasons: List[str],
    num_examples: int,
    alternatives: List[Dict] = None
) -> Dict:
    """Build the final response with scaled time/cost estimates."""
    
    model = MODELS[primary_key]
    
    # Scale time/cost based on dataset size (base is ~200 examples)
    scale_factor = max(0.5, min(10.0, num_examples / 200))
    
    # Model size multiplier
    size_mult = {'2B': 0.7, '3B': 1.0, '3.8B': 1.1, '7B': 2.0}.get(model['size'], 1.0)
    
    scaled_time = max(2, round(model['training_time_base'] * scale_factor * size_mult))
    scaled_cost = round(model['cost_base'] * scale_factor * size_mult, 2)
    
    # Confidence based on score
    confidence = 'high' if score >= 80 else ('medium' if score >= 60 else 'low')
    
    # Accuracy estimate
    accuracy_boost = min(score / 20, 5)
    estimated_accuracy = min(model['accuracy_baseline'] + accuracy_boost, 95)
    
    # Build alternatives with scaled costs
    formatted_alternatives = []
    if alternatives:
        for alt in alternatives:
            alt_model = MODELS.get(alt.get('model'))
            if alt_model:
                alt_scale = max(0.5, min(10.0, num_examples / 200))
                alt_size_mult = {'2B': 0.7, '3B': 1.0, '3.8B': 1.1, '7B': 2.0}.get(alt_model['size'], 1.0)
                formatted_alternatives.append({
                    'model_name': alt_model['name'],
                    'score': round(alt.get('score', 0) / 100, 2),
                    'reasons': alt.get('reasons', ['Good alternative']),
                    'training_time_min': max(2, round(alt_model['training_time_base'] * alt_scale * alt_size_mult)),
                    'cost_usd': round(alt_model['cost_base'] * alt_scale * alt_size_mult, 2)
                })
    
    return {
        'primary_recommendation': {
            'model_id': model['id'],
            'model_name': model['name'],
            'size': model['size'],
            'score': round(score / 100, 2),
            'confidence': confidence,
            'reasons': reasons,
            'training_time_min': scaled_time,
            'cost_usd': scaled_cost,
            'estimated_accuracy': round(estimated_accuracy, 1)
        },
        'alternatives': formatted_alternatives,
        'all_scores': {k: round(v / 100, 2) for k, v in all_scores.items()}
    }
