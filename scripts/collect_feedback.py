"""
Feedback Collection Script
========================

Collects human feedback on model responses for DPO/GRPO training.
Run this after evaluating the model to collect preference data.

Usage:
    python scripts/collect_feedback.py
"""

import json
import os
from datetime import datetime

FEEDBACK_FILE = "data/processed/llm/feedback.jsonl"

def save_feedback(feedback: dict):
    """Save feedback to file."""
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    
    with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(feedback, ensure_ascii=False) + '\n')

def collect_feedback_pair(
    prompt: str,
    response_a: str,
    response_b: str,
    model_name: str = "SFT"
):
    """Collect feedback for a pair of responses."""
    
    print("\n" + "="*60)
    print("FEEDBACK COLLECTION")
    print("="*60)
    print(f"\nPrompt: {prompt[:200]}...")
    
    print("\n" + "-"*40)
    print("Response A:")
    print("-"*40)
    print(response_a[:500] + "..." if len(response_a) > 500 else response_a)
    
    print("\n" + "-"*40)
    print("Response B:")
    print("-"*40)
    print(response_b[:500] + "..." if len(response_b) > 500 else response_b)
    
    print("\n" + "="*60)
    print("Which response is better?")
    print("  [a] - Response A is better")
    print("  [b] - Response B is better")  
    print("  [t] - Both are equally good")
    print("  [b] - Both are equally bad")
    print("  [s] - Skip")
    print("  [q] - Quit")
    print("="*60)
    
    choice = input("\nYour choice: ").strip().lower()
    
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "model_name": model_name,
        "choice": choice
    }
    
    if choice == 'a':
        feedback["winner"] = "A"
        feedback["chosen"] = response_a
        feedback["rejected"] = response_b
        print("\n✓ You chose Response A")
    elif choice == 'b':
        feedback["winner"] = "B"
        feedback["chosen"] = response_b
        feedback["rejected"] = response_a
        print("\n✓ You chose Response B")
    elif choice == 't':
        feedback["winner"] = "tie_good"
        print("\n✓ Marked as equally good")
    elif choice == 'n':
        feedback["winner"] = "tie_bad"
        print("\n✓ Marked as equally bad")
    elif choice == 's':
        print("\n⏭ Skipped")
        return None
    elif choice == 'q':
        print("\n👋 Goodbye!")
        return None
    else:
        print("\n❌ Invalid choice")
        return None
    
    # Save feedback
    save_feedback(feedback)
    print(f"💾 Saved to {FEEDBACK_FILE}")
    
    return feedback

def show_feedback_stats():
    """Show feedback statistics."""
    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback collected yet.")
        return
    
    with open(FEEDBACK_FILE, 'r') as f:
        feedback_data = [json.loads(line) for line in f]
    
    print("\n" + "="*60)
    print("FEEDBACK STATISTICS")
    print("="*60)
    print(f"Total feedback: {len(feedback_data)}")
    
    # Count choices
    choices = {}
    for fb in feedback_data:
        choice = fb.get('choice', 'unknown')
        choices[choice] = choices.get(choice, 0) + 1
    
    print("\nChoice distribution:")
    for choice, count in sorted(choices.items()):
        print(f"  {choice}: {count}")
    
    # Convert to DPO format
    dpo_pairs = []
    for fb in feedback_data:
        if 'chosen' in fb and 'rejected' in fb:
            dpo_pairs.append({
                "prompt": fb["prompt"],
                "chosen": fb["chosen"],
                "rejected": fb["rejected"]
            })
    
    print(f"\nDPO-compatible pairs: {len(dpo_pairs)}")
    
    return dpo_pairs

def export_dpo_dataset():
    """Export collected feedback as DPO dataset."""
    dpo_pairs = show_feedback_stats()
    
    if dpo_pairs:
        output_file = FEEDBACK_FILE.replace('.jsonl', '_dpo.jsonl')
        with open(output_file, 'w') as f:
            for pair in dpo_pairs:
                f.write(json.dumps(pair) + '\n')
        print(f"\n💾 Exported DPO dataset to: {output_file}")

def main():
    print("="*60)
    print("Feedback Collection System")
    print("="*60)
    print("\nOptions:")
    print("  [1] Show feedback statistics")
    print("  [2] Export as DPO dataset")
    print("  [3] Interactive feedback collection")
    print("  [q] Quit")
    
    choice = input("\nChoice: ").strip().lower()
    
    if choice == '1':
        show_feedback_stats()
    elif choice == '2':
        export_dpo_dataset()
    elif choice == '3':
        # Example interactive session
        example_prompt = "What causes EDGE_RING defects?"
        example_response_a = "EDGE_RING defects are caused by particle contamination in the CVD chamber due to chemical purity issues. This can be identified by high particle counts and requires chamber cleaning."
        example_response_b = "Weather is sunny."
        
        collect_feedback_pair(example_prompt, example_response_a, example_response_b)
    elif choice == 'q':
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
