"""
Analyze Dataset Tool
====================
Analyze JSONL chat format dataset for statistics and characteristics.
"""

import statistics
from typing import TYPE_CHECKING, List, Dict, Set

if TYPE_CHECKING:
    from tunekit.state import TuneKitState


def analyze_dataset(state: "TuneKitState") -> dict:
    """
    Analyze chat format dataset and extract statistics.
    
    Inputs (from state):
        - raw_data: List[Dict] - conversation data
        - num_rows: int
        - stats: Dict (from ingest)
    
    Outputs (to state):
        - dataset_stats: Dict - comprehensive statistics
        - conversation_characteristics: Dict - patterns detected
    """
    raw_data = state.get("raw_data", [])
    num_rows = state.get("num_rows", 0)
    
    if not raw_data:
        return {
            "dataset_stats": {},
            "conversation_characteristics": {}
        }
    
    # --- Analyze conversation patterns ---
    
    # Count message types
    total_user_msgs = 0
    total_assistant_msgs = 0
    total_system_msgs = 0
    
    # Track conversation lengths
    conversation_lengths = []
    user_message_lengths = []
    assistant_message_lengths = []
    
    # Track unique assistant responses (for classification detection)
    unique_assistant_responses: Set[str] = set()
    
    # Track conversation structures
    single_turn_count = 0
    multi_turn_count = 0
    has_system_count = 0
    
    for entry in raw_data:
        messages = entry.get("messages", [])
        conversation_lengths.append(len(messages))
        
        # Count by role
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            content_len = len(content)
            
            if role == "user":
                total_user_msgs += 1
                user_message_lengths.append(content_len)
            elif role == "assistant":
                total_assistant_msgs += 1
                assistant_message_lengths.append(content_len)
                unique_assistant_responses.add(content.strip().lower())
            elif role == "system":
                total_system_msgs += 1
        
        # Check structure
        non_system = [m for m in messages if m.get("role") != "system"]   #messages without system prompts
        if len(non_system) == 2:
            single_turn_count += 1
        else:
            multi_turn_count += 1
        
        if any(m.get("role") == "system" for m in messages):
            has_system_count += 1
    
    # Calculate averages
    avg_conversation_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
    avg_user_length = sum(user_message_lengths) / len(user_message_lengths) if user_message_lengths else 0
    avg_assistant_length = sum(assistant_message_lengths) / len(assistant_message_lengths) if assistant_message_lengths else 0
    
    # Detect if this looks like classification (few unique responses)
    num_unique_responses = len(unique_assistant_responses)
    looks_like_classification = num_unique_responses < 20 and num_unique_responses > 0
    
    # Detect if responses are very short (classification-like)
    avg_response_short = avg_assistant_length < 50
    
    # Detect language (check for non-ASCII)
    non_ascii_count = 0
    sample_responses = list(unique_assistant_responses)[:50]  # Sample first 50
    for response in sample_responses:
        if any(ord(c) > 127 for c in response):
            non_ascii_count += 1
    
    sample_size = min(50, len(unique_assistant_responses))
    is_multilingual = (non_ascii_count / sample_size) > 0.3 if sample_size > 0 else False
    
    # Calculate output variance (stdev / mean of assistant message lengths)
    output_variance = 0.0
    if assistant_message_lengths and len(assistant_message_lengths) > 1:
        mean_length = statistics.mean(assistant_message_lengths)
        if mean_length > 0:
            stdev_length = statistics.stdev(assistant_message_lengths)
            output_variance = round(stdev_length / mean_length, 2)
    
    # Detect JSON output (check if responses start with { or [)
    json_like_count = 0
    sample_assistant_responses = []
    for entry in raw_data:
        for msg in entry.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").strip()
                if content:
                    sample_assistant_responses.append(content)
                    if len(sample_assistant_responses) >= 50:
                        break
        if len(sample_assistant_responses) >= 50:
            break
    
    for response in sample_assistant_responses[:50]:
        stripped = response.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            json_like_count += 1
    
    looks_like_json_output = (json_like_count / len(sample_assistant_responses)) > 0.5 if sample_assistant_responses else False
    
    # Calculate total tokens (rough estimate: 1 token ≈ 4 chars)
    total_chars = sum(
        sum(len(msg.get("content", "")) for msg in entry.get("messages", []))
        for entry in raw_data
    )
    estimated_tokens = total_chars // 4
    
    # Build comprehensive stats
    total_messages = total_user_msgs + total_assistant_msgs + total_system_msgs
    
    dataset_stats = {
        "num_examples": num_rows,
        "total_messages": total_messages,
        "avg_messages_per_example": round(avg_conversation_length, 1),
        "message_counts": {
            "user": total_user_msgs,
            "assistant": total_assistant_msgs,
            "system": total_system_msgs
        },
        "avg_lengths": {
            "user_messages": round(avg_user_length),
            "assistant_messages": round(avg_assistant_length),
            "conversation": round(avg_conversation_length)
        },
        "conversation_structure": {
            "single_turn": single_turn_count,
            "multi_turn": multi_turn_count,
            "with_system": has_system_count,
            "without_system": num_rows - has_system_count
        },
        "estimated_tokens": estimated_tokens,
        "has_system_prompts": has_system_count > 0
    }
    
    # Conversation characteristics for model selection
    conversation_characteristics = {
        "looks_like_classification": looks_like_classification,
        "num_unique_assistant_responses": num_unique_responses,
        "avg_response_short": avg_response_short,
        "avg_response_length": round(avg_assistant_length),
        "is_multi_turn": multi_turn_count > single_turn_count,
        "has_system_prompts": has_system_count > 0,
        "avg_conversation_length": round(avg_conversation_length),
        "is_multilingual": is_multilingual,
        "output_variance": output_variance,
        "looks_like_json_output": looks_like_json_output
    }
    
    print(f"\n📊 Dataset Analysis:")
    print(f"   Examples: {num_rows}")
    print(f"   Total messages: {dataset_stats['total_messages']}")
    print(f"   Avg per conversation: {dataset_stats['avg_messages_per_example']}")
    print(f"   Structure: {single_turn_count} single-turn, {multi_turn_count} multi-turn")
    print(f"   System prompts: {has_system_count}/{num_rows}")
    print(f"   Unique assistant responses: {num_unique_responses}")
    if looks_like_classification:
        print(f"   💡 Appears to be classification-style data")
    if is_multilingual:
        print(f"   🌍 Multilingual dataset detected")
    print(f"   Estimated tokens: ~{estimated_tokens:,}")
    
    return {
        "dataset_stats": dataset_stats,
        "conversation_characteristics": conversation_characteristics
    }
