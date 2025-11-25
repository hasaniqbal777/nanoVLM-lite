#!/usr/bin/env python3
"""
Baseline evaluation for nanoVLM on A-OKVQA.
Tests enhanced prompts to improve VLM understanding.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Import base evaluator
sys.path.insert(0, str(Path(__file__).parent))
from base_evaluator import BaseEvaluator


class BaselineEvaluator(BaseEvaluator):
    """Evaluator with improved prompt engineering for better VLM understanding."""
    
    def get_eval_name(self) -> str:
        """Return evaluation name for display."""
        return "BASELINE"
    
    def format_mcq_prompt(self, question: str, choices: list) -> str:
        """Format question with prompt engineering for better VLM understanding."""
        prompt = f"Look at the image carefully and answer this question:\n\n{question}\n\nChoose the best answer from these options:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nRespond with only the letter (A, B, C, or D) of the correct answer."
        return prompt
    
    def format_open_answer_prompt(self, question: str) -> str:
        """Format open-answer prompt with prompt engineering."""
        return f"Look at the image carefully.\n\nQuestion: {question}\n\nProvide a concise, specific answer based on what you see in the image:"
    
    def extract_answer(self, text: str) -> str:
        """Extract the answer letter from model output."""
        text = text.strip().upper()
        
        import re
        
        # Pattern 1: "The correct answer is X" or "The answer is X"
        answer_match = re.search(r'(?:CORRECT\s+)?ANSWER\s+IS\s+([A-D])', text)
        if answer_match:
            return answer_match.group(1)
        
        # Pattern 2: "Answer: X" or "Answer:X"
        answer_match = re.search(r'ANSWER\s*:\s*([A-D])', text)
        if answer_match:
            return answer_match.group(1)
        
        # Pattern 3: Letter followed by period at end (e.g., "D.")
        answer_match = re.search(r'\b([A-D])\.\s*$', text)
        if answer_match:
            return answer_match.group(1)
        
        # Pattern 4: Standalone letter at end
        words = text.split()
        if words:
            last_word = words[-1].strip('.,!?:;')
            if len(last_word) == 1 and last_word in ['A', 'B', 'C', 'D']:
                return last_word
        
        # Pattern 5: Look for first isolated letter (word boundary on both sides)
        answer_match = re.search(r'\b([A-D])\b', text)
        if answer_match:
            return answer_match.group(1)
        
        # Default to A if no valid answer found
        return 'A'


# Save results helper
def save_results(results, output_path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for nanoVLM on A-OKVQA")
    parser.add_argument("--model-path", type=str, default="lusxvr/nanoVLM",
                        help="Path to model or HuggingFace model ID")
    parser.add_argument("--split", type=str, default="validation", 
                        choices=["train", "validation", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (None = all)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["mcq", "oa", "open-answer", "both"],
                        help="Evaluation mode: mcq, oa (open-answer), or both")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Target resolution for vision encoder (e.g., 384, 256). If not provided, uses model default.")
    parser.add_argument("--output", type=str, default="results/baseline_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Create evaluator with optional resolution
    evaluator = BaselineEvaluator(args.model_path, args.device, resolution=args.resolution)
    
    # Profile model
    profile_dict = evaluator.profile_model()
    
    # Normalize mode
    mode = args.mode.lower()
    if mode in ["oa", "open-answer"]:
        mode = "open-answer"
    
    # Run evaluations based on mode
    print("\n" + "="*60)
    print(f"RUNNING BASELINE EVALUATION ({mode.upper()})")
    print("="*60)
    
    results = {**profile_dict}
    
    if mode in ["mcq", "both"]:
        mcq_results = evaluator.evaluate_mcq(split=args.split, max_samples=args.max_samples)
        results.update(mcq_results)
    
    if mode in ["open-answer", "both"]:
        oa_results = evaluator.evaluate_open_answer(split=args.split, max_samples=args.max_samples)
        results.update(oa_results)
    
    results["model_path"] = args.model_path
    results["split"] = args.split
    results["mode"] = mode
    results["resolution"] = evaluator.model.cfg.vit_img_size
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Mode: {mode}")
    print(f"Resolution: {results['resolution']}×{results['resolution']}")
    print(f"Parameters: {results['total_params']:,}")
    print(f"Estimated TFLOPs: {results['estimated_tflops_per_forward']:.4f}")
    
    if mode in ["mcq", "both"]:
        print()
        print("MCQ Evaluation:")
        print(f"  Accuracy: {results['mcq_accuracy']*100:.2f}% ({results['mcq_correct']}/{results['mcq_total']})")
        print(f"  Avg Latency: {results['mcq_avg_latency']*1000:.2f} ms/sample")
    
    if mode in ["open-answer", "both"]:
        print()
        print("Open-Answer Evaluation:")
        print(f"  Accuracy: {results['oa_accuracy']*100:.2f}% ({results['oa_correct']}/{results['oa_total']})")
        print(f"  Avg Latency: {results['oa_avg_latency']*1000:.2f} ms/sample")
    
    print("="*60)


if __name__ == "__main__":
    main()
