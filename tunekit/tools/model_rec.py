"""
Model Recommendation Tool
=========================
Recommends Small Language Models (SLMs) based on task type and dataset characteristics.
"""

from typing import Dict, List, Literal

# Model metadata
MODELS = {
    'phi-4-mini': {
        'id': 'microsoft/Phi-4-mini-instruct',
        'name': 'Phi-4 Mini',
        'size': '3.8B',
        'strengths': ['classification', 'reasoning', 'structured_output', 'ios'],
        'training_time_min': 3,
        'cost_usd': 0.18,
        'accuracy_baseline': 87
    },
    'gemma-3-2b': {
        'id': 'google/gemma-3-2b-it',
        'name': 'Gemma 3 2B',
        'size': '2B',
        'strengths': ['speed', 'small_datasets', 'edge_deployment'],
        'training_time_min': 2,
        'cost_usd': 0.12,
        'accuracy_baseline': 82
    },
    'llama-3.2-3b': {
        'id': 'meta-llama/Llama-3.2-3B-Instruct',
        'name': 'Llama 3.2 3B',
        'size': '3B',
        'strengths': ['quality', 'multi_turn', 'large_datasets'],
        'training_time_min': 4,
        'cost_usd': 0.20,
        'accuracy_baseline': 89
    },
    'qwen-2.5-3b': {
        'id': 'Qwen/Qwen2.5-3B-Instruct',
        'name': 'Qwen 2.5 3B',
        'size': '3B',
        'strengths': ['multilingual', 'chinese', '29_languages'],
        'training_time_min': 4,
        'cost_usd': 0.20,
        'accuracy_baseline': 88
    },
    'mistral-7b': {
        'id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'name': 'Mistral 7B',
        'size': '7B',
        'strengths': ['long_generation', 'versatile', 'complex_tasks'],
        'training_time_min': 6,
        'cost_usd': 0.35,
        'accuracy_baseline': 91
    }
}

# Deployment target filters - which models are viable for each deployment
DEPLOYMENT_FILTERS = {
    'cloud_api': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'qwen-2.5-3b', 'mistral-7b'],  # All models supported
    
    'mobile_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],  # Cross-platform mobile (iOS + Android)
    
    'ios_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],  # Core ML optimized, <4GB
    
    'android_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],  # TFLite/MediaPipe/NNAPI
    
    'edge_device': ['gemma-3-2b', 'phi-4-mini'],  # Low power (Pi, Jetson, embedded)
    
    'web_browser': ['gemma-3-2b', 'phi-4-mini', 'llama-3.2-3b'],  # WebGPU/Transformers.js
    
    'desktop_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'qwen-2.5-3b', 'mistral-7b'],  # Full hardware, 8GB+ RAM
    
    'not_sure': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'qwen-2.5-3b', 'mistral-7b']  # Show all options
}


def score_task_match(user_task: str, model_key: str) -> float:
    """Score based on task type match (0-20 points)."""
    task_mapping = {
        'classify': {
            'phi-4-mini': 18,
            'gemma-3-2b': 15,
            'llama-3.2-3b': 16,
            'qwen-2.5-3b': 14,
            'mistral-7b': 12
        },
        'qa': {
            'phi-4-mini': 12,
            'gemma-3-2b': 8,
            'llama-3.2-3b': 20,
            'qwen-2.5-3b': 15,
            'mistral-7b': 18
        },
        'conversation': {
            'phi-4-mini': 10,
            'gemma-3-2b': 8,
            'llama-3.2-3b': 20,
            'qwen-2.5-3b': 15,
            'mistral-7b': 18
        },
        'generation': {
            'phi-4-mini': 8,
            'gemma-3-2b': 10,
            'llama-3.2-3b': 15,
            'qwen-2.5-3b': 12,
            'mistral-7b': 20
        },
        'extraction': {
            'phi-4-mini': 18,
            'llama-3.2-3b': 16,
            'mistral-7b': 15,
            'qwen-2.5-3b': 14,
            'gemma-3-2b': 12
        }
    }
    
    return task_mapping.get(user_task, {}).get(model_key, 0)


def score_dataset_size(num_examples: int, model_key: str) -> float:
    """Score based on dataset size (0-15 points)."""
    if num_examples < 500:
        # Small dataset - Gemma is best
        scores = {
            'gemma-3-2b': 15,
            'phi-4-mini': 8,
            'llama-3.2-3b': 5,
            'qwen-2.5-3b': 5,
            'mistral-7b': 3
        }
    elif num_examples > 2000:
        # Large dataset - Llama benefits most
        scores = {
            'llama-3.2-3b': 15,
            'mistral-7b': 12,
            'phi-4-mini': 10,
            'qwen-2.5-3b': 10,
            'gemma-3-2b': 5
        }
    else:
        # Medium dataset - More balanced, slight edge to Llama for quality
        scores = {
            'llama-3.2-3b': 15,
            'phi-4-mini': 12,
            'mistral-7b': 10,
            'qwen-2.5-3b': 10,
            'gemma-3-2b': 8
        }
    
    return scores.get(model_key, 0)


def score_output_characteristics(characteristics: Dict, model_key: str) -> float:
    """Score based on output characteristics (0-15 points)."""
    score = 0.0
    
    avg_response_length = characteristics.get('avg_response_length', 0)
    looks_like_json = characteristics.get('looks_like_json_output', False)
    
    # Short responses favor smaller models
    if avg_response_length < 50:
        if model_key == 'phi-4-mini':
            score += 12
        elif model_key == 'gemma-3-2b':
            score += 12
        elif model_key == 'llama-3.2-3b':
            score += 10
        else:
            score += 8
    
    # Long responses favor Mistral
    elif avg_response_length > 200:
        if model_key == 'mistral-7b':
            score += 15
        elif model_key == 'llama-3.2-3b':
            score += 12
        else:
            score += 5
    
    # JSON output favors structured output models
    if looks_like_json:
        if model_key == 'phi-4-mini':
            score += 12
        elif model_key == 'llama-3.2-3b':
            score += 12
        elif model_key == 'mistral-7b':
            score += 10
        else:
            score += 8
    
    return min(score, 15)  # Cap at 15


def score_language(characteristics: Dict, model_key: str) -> float:
    """Score based on language requirements (0-15 points)."""
    is_multilingual = characteristics.get('is_multilingual', False)
    
    if is_multilingual:
        if model_key == 'qwen-2.5-3b':
            return 15
        elif model_key == 'mistral-7b':
            return 10
        else:
            return 5
    
    return 0  # No bonus for English-only


def score_conversation_complexity(characteristics: Dict, model_key: str) -> float:
    """Score based on conversation complexity (0-10 points)."""
    is_multi_turn = characteristics.get('is_multi_turn', False)
    
    if is_multi_turn:
        if model_key == 'llama-3.2-3b':
            return 10
        elif model_key == 'mistral-7b':
            return 8
        elif model_key == 'qwen-2.5-3b':
            return 7
        else:
            return 5
    
    return 0  # Single-turn doesn't need special handling


def score_classification_patterns(characteristics: Dict, model_key: str) -> float:
    """Score based on classification patterns (0-10 points)."""
    score = 0.0
    
    looks_like_classification = characteristics.get('looks_like_classification', False)
    num_unique_responses = characteristics.get('num_unique_assistant_responses', 0)
    
    if looks_like_classification:
        if model_key == 'phi-4-mini':
            score += 8
        elif model_key == 'gemma-3-2b':
            score += 8
        elif model_key == 'llama-3.2-3b':
            score += 7
        else:
            score += 6
    
    if num_unique_responses < 10:
        if model_key == 'phi-4-mini':
            score += 7
        elif model_key == 'gemma-3-2b':
            score += 7
        elif model_key == 'llama-3.2-3b':
            score += 6
        else:
            score += 5
    
    return min(score, 10)  # Cap at 10


def score_speed_efficiency(num_examples: int, model_key: str) -> float:
    """Score based on speed/efficiency needs (0-10 points)."""
    if num_examples < 500:
        if model_key == 'gemma-3-2b':
            return 10
        elif model_key == 'phi-4-mini':
            return 8
        elif model_key == 'llama-3.2-3b':
            return 5
        else:
            return 4
    
    # For larger datasets, give slight bonus to faster models
    elif num_examples < 1000:
        if model_key == 'gemma-3-2b':
            return 5
        elif model_key == 'phi-4-mini':
            return 4
        else:
            return 2
    
    return 0  # Very large datasets don't prioritize speed


def score_deployment_match(deployment_target: str, model_key: str) -> float:
    """Score based on deployment target match (0-15 points)."""
    if deployment_target == 'not_sure' or deployment_target == 'cloud_api':
        return 0  # No bonus, all models viable
    
    deployment_scores = {
        'mobile_app': {
            'phi-4-mini': 15,  # Best cross-platform support (iOS + Android)
            'gemma-3-2b': 12,  # Works on both platforms
            'llama-3.2-3b': 10,  # Core ML + MediaPipe compatible
            'qwen-2.5-3b': 0,
            'mistral-7b': 0
        },
        'ios_app': {
            'phi-4-mini': 15,  # Best iOS optimization
            'gemma-3-2b': 12,  # Core ML compatible
            'llama-3.2-3b': 10,  # Core ML compatible
            'qwen-2.5-3b': 0,
            'mistral-7b': 0
        },
        'android_app': {
            'phi-4-mini': 15,  # Excellent Android support
            'gemma-3-2b': 12,  # TFLite optimized
            'llama-3.2-3b': 10,  # MediaPipe/NNAPI compatible
            'qwen-2.5-3b': 0,
            'mistral-7b': 0
        },
        'edge_device': {
            'gemma-3-2b': 15,  # Smallest, most efficient
            'phi-4-mini': 12,  # Good for Pi 5, Jetson
            'llama-3.2-3b': 0,
            'qwen-2.5-3b': 0,
            'mistral-7b': 0
        },
        'web_browser': {
            'gemma-3-2b': 15,  # Lightest for WebGPU
            'phi-4-mini': 12,  # WebGPU compatible
            'llama-3.2-3b': 10,  # Transformers.js support
            'qwen-2.5-3b': 0,
            'mistral-7b': 0
        },
        'desktop_app': {
            'mistral-7b': 15,  # Best quality if hardware allows
            'llama-3.2-3b': 14,  # High quality
            'qwen-2.5-3b': 12,  # Multilingual option
            'phi-4-mini': 12,  # Balanced performance
            'gemma-3-2b': 10  # Fast inference
        }
    }
    
    return deployment_scores.get(deployment_target, {}).get(model_key, 0)


def calculate_model_scores(
    user_task: str,
    conversation_characteristics: Dict,
    num_examples: int,
    deployment_target: str = 'not_sure',
    allowed_models: List[str] = None
) -> Dict[str, float]:
    """Calculate scores for all models (0-100 points each)."""
    scores = {}
    
    # Filter models by deployment if needed
    models_to_score = allowed_models if allowed_models else list(MODELS.keys())
    
    # Guard against empty allowed_models
    if not models_to_score:
        models_to_score = list(MODELS.keys())
    
    for model_key in models_to_score:
        # Skip invalid model keys
        if model_key not in MODELS:
            continue
        score = 0.0
        
        # 1. Task match (20 points)
        task_score = score_task_match(user_task, model_key)
        score += task_score
        
        # 2. Dataset size (15 points)
        size_score = score_dataset_size(num_examples, model_key)
        score += size_score
        
        # 3. Output characteristics (15 points)
        output_score = score_output_characteristics(conversation_characteristics, model_key)
        score += output_score
        
        # 4. Language (15 points)
        lang_score = score_language(conversation_characteristics, model_key)
        score += lang_score
        
        # 5. Conversation complexity (10 points)
        complexity_score = score_conversation_complexity(conversation_characteristics, model_key)
        score += complexity_score
        
        # 6. Classification patterns (10 points)
        classification_score = score_classification_patterns(conversation_characteristics, model_key)
        score += classification_score
        
        # 7. Speed/efficiency (10 points)
        speed_score = score_speed_efficiency(num_examples, model_key)
        score += speed_score
        
        # 8. Deployment match (15 points)
        deployment_score = score_deployment_match(deployment_target, model_key)
        score += deployment_score
        
        # 9. Versatility fallback (5 points)
        score += 5
        
        scores[model_key] = score
        
        # Print scoring breakdown
        print(f"\n📊 {MODELS[model_key]['name']} Score: {score:.1f}/100")
        print(f"   Task match: {task_score:.1f} | Size: {size_score:.1f} | Output: {output_score:.1f}")
        print(f"   Language: {lang_score:.1f} | Complexity: {complexity_score:.1f} | Classification: {classification_score:.1f}")
        print(f"   Speed: {speed_score:.1f} | Deployment: {deployment_score:.1f} | Base: 5.0")
    
    return scores


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to 0-1 range."""
    if not scores:
        return {}
    
    max_score = max(scores.values())
    if max_score == 0:
        return {k: 0.0 for k in scores.keys()}
    
    return {k: v / max_score for k, v in scores.items()}


def generate_reasons(
    model_key: str,
    user_task: str,
    conversation_characteristics: Dict,
    num_examples: int,
    normalized_score: float,
    deployment_target: str = 'not_sure'
) -> List[str]:
    """Generate human-readable reasons for recommendation."""
    if model_key not in MODELS:
        return ["Model recommendation available"]
    
    reasons = []
    model = MODELS[model_key]
    
    # Deployment-based reason (highest priority)
    deployment_reasons = {
        'mobile_app': 'Cross-platform mobile deployment (iOS + Android)',
        'ios_app': 'Optimized for iOS deployment',
        'android_app': 'Lightweight for mobile apps',
        'edge_device': 'Smallest model for edge devices',
        'web_browser': 'Lightweight for browser deployment',
        'desktop_app': 'Well-suited for desktop applications',
        'cloud_api': 'Excellent for cloud API deployment'
    }
    if deployment_target in deployment_reasons and deployment_target != 'not_sure':
        reasons.append(deployment_reasons[deployment_target])
    
    # Task-based reason
    task_reasons = {
        'classify': 'Best for classification tasks',
        'qa': 'Excellent for question answering',
        'conversation': 'Optimized for multi-turn conversations',
        'generation': 'Superior long-form generation',
        'extraction': 'Strong structured data extraction'
    }
    if user_task in task_reasons:
        reasons.append(task_reasons[user_task])
    
    # Dataset size reason
    if num_examples < 500:
        reasons.append(f'Optimal for small datasets ({num_examples} examples)')
    elif num_examples > 2000:
        reasons.append(f'Best performance on large datasets ({num_examples} examples)')
    else:
        reasons.append(f'Well-suited for your dataset size ({num_examples} examples)')
    
    # Output characteristics
    if conversation_characteristics.get('looks_like_json_output'):
        reasons.append('Excellent with structured JSON outputs')
    
    if conversation_characteristics.get('avg_response_length', 0) < 50:
        reasons.append('Optimized for short, concise responses')
    elif conversation_characteristics.get('avg_response_length', 0) > 200:
        reasons.append('Handles long-form responses well')
    
    # Language
    if conversation_characteristics.get('is_multilingual'):
        reasons.append('Supports 29 languages including Chinese')
    
    # Classification
    if conversation_characteristics.get('looks_like_classification'):
        reasons.append('Strong classification performance')
    
    # Multi-turn
    if conversation_characteristics.get('is_multi_turn'):
        reasons.append('Excellent multi-turn conversation handling')
    
    # Speed
    if num_examples < 500:
        reasons.append(f'Fastest training time ({model["training_time_min"]} min)')
    
    return reasons[:3]  # Return top 3 reasons


def recommend_model(
    user_task: str,
    conversation_characteristics: Dict,
    num_examples: int,
    deployment_target: str = 'not_sure'
) -> Dict:
    """
    Recommend the best SLM model based on task, dataset characteristics, and deployment target.
    
    Args:
        user_task: "classify" | "qa" | "conversation" | "generation" | "extraction"
        conversation_characteristics: Dict from analyze.py with:
            - looks_like_classification
            - num_unique_assistant_responses
            - avg_response_length
            - is_multi_turn
            - is_multilingual
            - looks_like_json_output
            - output_variance
            - has_system_prompts
        num_examples: Dataset size
        deployment_target: "cloud_api" | "mobile_app" | "ios_app" | "android_app" | 
                          "edge_device" | "web_browser" | "desktop_app" | "not_sure"
    
    Returns:
        Dict with primary_recommendation, alternatives, and all_scores
    """
    print(f"\n🎯 Model Recommendation Analysis")
    print(f"   Task: {user_task}")
    print(f"   Deployment: {deployment_target}")
    print(f"   Dataset size: {num_examples} examples")
    print(f"   Characteristics: {conversation_characteristics}")
    
    # Filter models based on deployment target
    allowed_models = DEPLOYMENT_FILTERS.get(deployment_target, DEPLOYMENT_FILTERS['not_sure'])
    
    # Guard against empty allowed_models
    if not allowed_models:
        allowed_models = DEPLOYMENT_FILTERS['not_sure']
    
    if deployment_target != 'not_sure' and deployment_target != 'cloud_api':
        print(f"\n🔍 Filtering models for {deployment_target}: {', '.join([MODELS[k]['name'] for k in allowed_models])}")
    
    # Calculate raw scores (only for allowed models)
    raw_scores = calculate_model_scores(
        user_task, 
        conversation_characteristics, 
        num_examples,
        deployment_target,
        allowed_models
    )
    
    # Normalize to 0-1
    normalized_scores = normalize_scores(raw_scores)
    
    # Guard against empty scores
    if not normalized_scores:
        raise ValueError("No models available for recommendation. Check deployment target and model filters.")
    
    # Sort by score
    sorted_models = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Primary recommendation
    primary_key, primary_score = sorted_models[0]
    primary_model = MODELS[primary_key]
    
    confidence = 'high' if primary_score >= 0.8 else ('medium' if primary_score >= 0.6 else 'low')
    
    # Estimate accuracy (baseline + boost from good fit)
    accuracy_boost = min(primary_score * 5, 5)  # Up to 5% boost
    estimated_accuracy = min(primary_model['accuracy_baseline'] + accuracy_boost, 95)
    
    primary_recommendation = {
        'model_id': primary_model['id'],
        'model_name': primary_model['name'],
        'size': primary_model['size'],
        'score': round(primary_score, 2),
        'confidence': confidence,
        'reasons': generate_reasons(primary_key, user_task, conversation_characteristics, num_examples, primary_score, deployment_target),
        'training_time_min': primary_model['training_time_min'],
        'cost_usd': primary_model['cost_usd'],
        'estimated_accuracy': round(estimated_accuracy, 1)
    }
    
    # Alternatives (top 2-3)
    alternatives = []
    for model_key, score in sorted_models[1:4]:  # Next 3 models
        model = MODELS[model_key]
        alt_reasons = []
        
        # Add reasons based on strengths
        if score >= 0.7:
            alt_reasons.append(f"Strong alternative ({score:.0%} match)")
        if model['cost_usd'] < primary_model['cost_usd']:
            alt_reasons.append(f"Lower cost (${model['cost_usd']})")
        if model['training_time_min'] < primary_model['training_time_min']:
            alt_reasons.append(f"Faster training ({model['training_time_min']} min)")
        
        alternatives.append({
            'model_name': model['name'],
            'score': round(score, 2),
            'reasons': alt_reasons[:2] if alt_reasons else ['Good alternative option']
        })
    
    print(f"\n✅ Recommended: {primary_model['name']} (score: {primary_score:.2f}, confidence: {confidence})")
    
    return {
        'primary_recommendation': primary_recommendation,
        'alternatives': alternatives,
        'all_scores': {k: round(v, 2) for k, v in normalized_scores.items()}
    }
