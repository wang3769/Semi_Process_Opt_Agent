"""
Dataset Module for LLM Fine-tuning
=================================

Creates datasets for:
- SFT (Supervised Fine-tuning)
- DPO (Direct Preference Optimization)
- GRPO (RL from Your Feedback)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SFT Dataset
# =============================================================================

def create_sft_instruction_prompt(
    instruction: str,
    context: str = "",
    response: str = ""
) -> Dict[str, str]:
    """
    Create an SFT prompt in instruction format.
    
    Args:
        instruction: The task instruction
        context: Additional context (optional)
        response: The expected response (optional, for training)
    
    Returns:
        Dictionary with 'prompt' and 'response' keys
    """
    if context:
        prompt = f"""<|im_start|>system
You are an expert semiconductor process engineer specializing in yield optimization, defect analysis, and root cause analysis.<|im_end|>
<|im_start|>user
Context:
{context}

Instruction: {instruction}<|im_end|>"""
    else:
        prompt = f"""<|im_start|>system
You are an expert semiconductor process engineer specializing in yield optimization, defect analysis, and root cause analysis.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>"""
    
    if response:
        prompt += f"\n<|im_start|>assistant\n{response}<|im_end|>"
    
    return {
        'prompt': prompt,
        'response': response
    }


def generate_rca_sft_examples() -> List[Dict[str, Any]]:
    """
    Generate SFT training examples for RCA tasks.
    These are based on the RAG knowledge base.
    """
    examples = []
    
    # Example 1: Defect classification
    examples.append({
        'instruction': "Classify the wafer defect pattern based on the description: 'Circular pattern of defects in the center region, about 2cm in diameter'",
        'context': "WM-811K dataset contains 9 defect patterns: Center (C), Donut (D), Edge-Ring (ER), Edge-Loc (EL), Local (L), Random (R), Scratch (S), Near-Full (NF), None (OK)",
        'response': "This defect pattern is classified as CENTER (C). The circular pattern in the center region is a characteristic signature of center defects. Root causes include: film deposition issues, center-only process variations, uneven temperature distribution. Common in PVD and CVD processes. Recommended actions: check deposition uniformity, verify temperature distribution across wafer, inspect center-specific process parameters."
    })
    
    # Example 2: Feature importance interpretation
    examples.append({
        'instruction': "Interpret the following SECOM feature analysis: Feature 47 (deposition temperature) has a SHAP value of +0.85, Feature 203 (film thickness) has SHAP value of -0.42, Feature 156 (particle count) has SHAP value of +0.31",
        'context': "SECOM dataset contains 590 sensor measurements including process parameters, optical measurements, thickness, and electrical properties. Positive SHAP values indicate the feature increases yield loss probability.",
        'response': "Based on the SHAP analysis:\n\n1. **Feature 47 (Deposition Temperature)** - Strong positive SHAP (+0.85): This is the PRIMARY driver of yield loss. Higher deposition temperatures correlate with increased defect probability. Recommend investigating temperature uniformity and calibration.\n\n2. **Feature 203 (Film Thickness)** - Negative SHAP (-0.42): This actually has a protective effect - thicker films correlate with better yield. May indicate optimal thickness threshold.\n\n3. **Feature 156 (Particle Count)** - Moderate positive SHAP (+0.31): Particle contamination contributes to yield loss. Secondary priority for investigation.\n\n**Primary Root Cause**: Deposition temperature variance is the key issue to address."
    })
    
    # Example 3: Root cause analysis
    examples.append({
        'instruction': "A wafer lot shows 15% yield drop. Wafer map shows ring-shaped defects at the edge. Process data shows chamber pressure variance of ±5 Torr. Provide root cause analysis.",
        'context': "Edge-Ring (ER) defects are typically caused by edge exposure issues, clamp marks, or uneven coating. Chamber pressure variance can cause uneven film deposition, especially at wafer edges where gas flow is less uniform.",
        'response': """**Root Cause Analysis Report**

**Defect Pattern**: Edge-Ring (ER)
**Observed**: Ring-shaped defects at wafer edge

**Primary Root Cause**: Chamber pressure instability (±5 Torr variance)
- Pressure fluctuations cause uneven film deposition
- Effect is most pronounced at wafer edge where gas flow dynamics differ
- This explains the ring pattern concentrated at edges

**Secondary Factors to Investigate**:
1. Edge ring/chuck condition - may need replacement
2. Pressure transducer calibration
3. Gas flow uniformity at chamber edges
4. Wafer clamping force consistency

**Recommended Actions**:
1. Perform pressure transducer calibration
2. Inspect edge ring for wear or particles
3. Review chamber conditioning recipes
4. Consider chamber pressure PID tuning

**Expected Impact**: Fixing pressure variance should recover 10-12% yield based on the correlation between ER defects and pressure instability."""
    })
    
    # Example 4: Process optimization
    examples.append({
        'instruction': "Current yield is 72%. Target yield is 85%. Main defect types are Local (L) and Random (R). Available process parameters: temperature (600-650°C), pressure (10-50 Torr), deposition time (30-120s). Suggest optimization strategy.",
        'context': "Local and Random defects are typically caused by particle contamination, systemic process issues, or chemical purity problems. These are scattered randomly across wafers rather than having a spatial pattern.",
        'response': """**Process Optimization Strategy**

**Current State**: 72% yield, Local/Random defects dominant

**Analysis**:
- Local/Random defects indicate particle contamination or systemic issues
- No spatial pattern suggests equipment or environment, not process parameters

**Optimization Approach**:
1. **Immediate Actions** (High Impact):
   - Clean/checks for particle sources in chamber
   - Verify chemical purity levels
   - Review wafer handling procedures
   
2. **Parameter Tuning** (Moderate Impact):
   - Lower temperature (620°C) to reduce thermal gradients
   - Increase pressure slightly (30 Torr) for better film uniformity
   - Extend deposition time (60s) for better coverage
   
3. **Expected Outcome**:
   - Target: 82-85% yield
   - Timeline: 2-3 optimization cycles

**Next Experiment**:
Run DOE with temperature (620 vs 640°C) and pressure (25 vs 35 Torr) while monitoring particle counts."""
    })
    
    # Example 5: Wafer map interpretation
    examples.append({
        'instruction': "Analyze this wafer map: Dense defects in center, sparse at edges, total defect count 47. What is the defect pattern and recommended action?",
        'context': "WM-811K defect patterns: Center (C), Donut (D), Edge-Ring (ER), Edge-Loc (EL), Local (L), Random (R), Scratch (S), Near-Full (NF)",
        'response': """**Wafer Map Analysis**

**Defect Count**: 47 defects
**Pattern**: Center (C) - Dense center, sparse edges

**Interpretation**:
- This is a classic CENTER defect pattern
- 47 defects is in the moderate range
- Suggests center-specific process issue

**Root Causes to Investigate**:
1. Center-specific temperature variation
2. Center spray nozzle blockage or malfunction
3. Film deposition uniformity at center
4. Wafer chuck centering

**Recommended Actions**:
1. Check deposition system for center-specific issues
2. Verify temperature uniformity (TC map)
3. Inspect center showerhead/nozzle
4. Review center-specific process recipes

**Priority**: HIGH - Center defects can rapidly degrade yield if unchecked"""
    })
    
    return examples


def create_sft_dataset(
    num_examples: int = 100,
    output_file: str = "data/processed/llm/sft_train.jsonl"
) -> List[Dict]:
    """
    Create a synthetic SFT training dataset for semiconductor RCA.
    """
    # 1. Define the building blocks for the synthetic data
    defect_types = ["Center", "Donut", "Edge-Ring", "Edge-Loc", "Local", "Random", "Scratch"]
    patterns = [
        "circular defects in center region",
        "ring-shaped pattern in middle region",
        "defects concentrated at wafer edge",
        "scattered local defects",
        "randomly distributed defects",
        "linear scratch patterns"
    ]
    root_causes = [
        "temperature gradient in deposition",
        "uneven coating at edge",
        "particle contamination",
        "mechanical handling damage",
        "chamber pressure variation",
        "chemical purity issues"
    ]
    actions = [
        "check temperature uniformity",
        "verify edge process parameters",
        "clean contamination sources",
        "inspect handling equipment",
        "calibrate pressure sensors",
        "test chemical quality"
    ]

    # 2. Define the templates with placeholders
    templates = [
        {
            'instruction': "Classify defect pattern: '{pattern_desc}'",
            'context': "WM-811K patterns: Center, Donut, Edge-Ring, Edge-Loc, Local, Random, Scratch, Near-Full",
            'response': "Based on the description '{pattern_desc}', this is a {defect_type} pattern. Root cause: {root_cause}. Recommended action: {action}."
        },
        {
            'instruction': "Yield dropped from {yield1}% to {yield2}%. Defect pattern: {pattern}. Provide RCA.",
            'context': "Analyze root cause for yield loss in semiconductor manufacturing.",
            'response': "RCA: Yield drop from {yield1}% to {yield2}% with {pattern} pattern. Root cause: {root_cause}. Recommended Actions: {action}. Expected recovery: {recovery}%."
        }
    ]

    all_examples = []

    # 3. Generation Loop
    while len(all_examples) < num_examples:
        template = random.choice(templates)
        
        # Create a dynamic dictionary of values to fill the template
        # We use random.randint to ensure the numbers vary across examples
        filler_data = {
            'pattern_desc': random.choice(patterns),
            'defect_type': random.choice(defect_types),
            'root_cause': random.choice(root_causes),
            'action': random.choice(actions),
            'yield1': random.randint(94, 99),
            'yield2': random.randint(75, 93),
            'pattern': random.choice(defect_types),
            'recovery': random.randint(5, 12)
        }
        
        # Inject the data into the strings
        try:
            example = {
                'instruction': template['instruction'].format_map(filler_data),
                'context': template['context'].format_map(filler_data),
                'response': template['response'].format_map(filler_data)
            }
            all_examples.append(example)
        except KeyError as e:
            logger.error(f"Missing key in filler_data: {e}")
            continue

    # 4. Save to JSONL (JSON Lines) format
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            # We use ensure_ascii=False to handle any special characters correctly
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    logger.info(f"Successfully created {len(all_examples)} examples at {output_file}")
    return all_examples


# =============================================================================
# DPO Dataset
# =============================================================================

def generate_dpo_preference_pairs() -> List[Dict[str, Any]]:
    """
    Generate DPO preference pairs.
    Each pair has a 'chosen' (good) and 'rejected' (bad) response.
    The model learns from these pairs.
    """
    pairs = []
    
    # Pair 1: Good RCA vs Bad RCA
    pairs.append({
        'prompt': "Defect pattern is Edge-Ring. Chamber pressure variance is ±5 Torr. What is the root cause?",
        'chosen': """**Root Cause Analysis**

**Defect Pattern**: Edge-Ring
**Key Finding**: Chamber pressure variance (±5 Torr)

**Analysis**:
- Edge-Ring defects are caused by edge-specific processing issues
- Pressure fluctuations cause uneven film deposition
- Effect is strongest at wafer edges where gas dynamics differ

**Root Cause**: Chamber pressure instability at edges

**Recommended Actions**:
1. Calibrate pressure transducer
2. Inspect edge ring for particles
3. Adjust edge-specific recipes

**Expected Impact**: 10-12% yield recovery""",
        'rejected': """The defect is Edge-Ring. This is caused by pressure. 

Just check the pressure. Maybe it's too high or too low. Or it could be the edge. 

I think the issue is pressure variation. The solution is to fix it."""
    })
    
    # Pair 2: Detailed analysis vs Vague
    pairs.append({
        'prompt': "SHAP analysis shows Feature 47 (deposition temperature) +0.85, Feature 203 (film thickness) -0.42. Interpret these results.",
        'chosen': """**SHAP Value Interpretation**

| Feature | SHAP Value | Interpretation |
|---------|------------|----------------|
| Feature 47 (Deposition Temperature) | +0.85 | **Primary driver** - Strong positive correlation with yield loss |
| Feature 203 (Film Thickness) | -0.42 | Protective effect - thicker films improve yield |

**Key Insights**:
1. Temperature is the #1 yield driver - a small increase dramatically increases defect probability
2. Film thickness has a protective relationship - target thicker films
3. These two features are likely independent

**Recommendations**:
- Focus on temperature uniformity first
- Establish temperature upper limit threshold
- Film thickness can be secondary optimization""",
        'rejected': """Feature 47 is positive so it causes defects. Feature 203 is negative so it's good. Temperature is important."""
    })
    
    # Pair 3: Actionable vs Generic
    pairs.append({
        'prompt': "Yield is 65%, target 85%. Main defects are Local and Random. Process params: temp 600-650°C, pressure 10-50 Torr. Suggest optimization.",
        'chosen': """**Optimization Strategy**

**Current**: 65% yield, Local/Random defects
**Target**: 85% yield

**Root Cause**: Local/Random = particle contamination

**Recommended Experiments**:

| Parameter | Current | Suggested | Rationale |
|-----------|---------|-----------|-----------|
| Temperature | 620°C | 605°C | Reduce thermal gradients |
| Pressure | 25 Torr | 30 Torr | Better uniformity |
| Clean time | 30s | 60s | Reduce particles |

**Expected**: +15-20% yield in 2 cycles

**Next Steps**:
1. Run DOE at 605°C/30Torr vs 620°C/25Torr
2. Monitor particle counts
3. Measure defect density""",
        'rejected': """Try different temperature and pressure. Maybe clean more. Could get to 85%."""
    })
    
    return pairs


def create_dpo_dataset(
    num_pairs: int = 50,
    output_file: str = "data/processed/llm/dpo_train.jsonl"
) -> List[Dict]:
    """
    Create DPO training dataset.
    
    Args:
        num_pairs: Number of preference pairs
        output_file: Output file path
        
    Returns:
        List of DPO pairs
    """
    pairs = generate_dpo_preference_pairs()
    
    # Add more variations
    while len(pairs) < num_pairs:
        pairs.extend(generate_dpo_preference_pairs())
    
    pairs = pairs[:num_pairs]
    
    # Save to file - use absolute path
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.path.dirname(__file__), "..", "..", output_file)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    logger.info(f"Created DPO dataset with {len(pairs)} pairs at {output_file}")
    return pairs


# =============================================================================
# Feedback Dataset (for GRPO)
# =============================================================================

def create_feedback_entry(
    prompt: str,
    response: str,
    rating: float,  # 1-5 stars or 0-1
    feedback_type: str = "quality"  # quality, accuracy, helpfulness
) -> Dict[str, Any]:
    """
    Create a feedback entry from user labeling.
    
    Args:
        prompt: The original prompt
        response: The model response
        rating: User rating (1-5 or 0-1)
        feedback_type: Type of feedback
        
    Returns:
        Feedback entry
    """
    return {
        'prompt': prompt,
        'response': response,
        'rating': rating,
        'feedback_type': feedback_type,
    }


def save_feedback(
    feedback: List[Dict[str, Any]],
    output_file: str = "data/processed/llm/feedback.jsonl"
):
    """Save user feedback for GRPO training."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:
        for entry in feedback:
            f.write(json.dumps(entry) + '\n')
    logger.info(f"Saved {len(feedback)} feedback entries to {output_file}")


def load_feedback(
    feedback_file: str = "data/processed/llm/feedback.jsonl"
) -> List[Dict[str, Any]]:
    """Load user feedback for GRPO training."""
    feedback = []
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            for line in f:
                feedback.append(json.loads(line))
    return feedback


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create directories
    os.makedirs("data/processed/llm", exist_ok=True)
    
    # Generate SFT dataset
    print("Generating SFT dataset...")
    sft_data = create_sft_dataset(num_examples=100)
    print(f"Created {len(sft_data)} SFT examples")
    
    # Generate DPO dataset
    print("\nGenerating DPO dataset...")
    dpo_data = create_dpo_dataset(num_pairs=50)
    print(f"Created {len(dpo_data)} DPO pairs")
    
    print("\n✅ Dataset generation complete!")