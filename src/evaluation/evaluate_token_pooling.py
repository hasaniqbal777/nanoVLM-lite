#!/usr/bin/env python3
"""
Token Pooling Evaluator for nanoVLM on A-OKVQA.

Applies post-hoc token pooling to an existing pretrained model WITHOUT retraining.
"""

import sys
from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn

# Add nanoVLM to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "models" / "nanoVLM"))
sys.path.insert(0, str(Path(__file__).parent))

from base_evaluator import BaseEvaluator
from models.modality_projector import ModalityProjector


class TokenPooledEvaluator(BaseEvaluator):
    """
    Evaluator with post-hoc token pooling applied to the modality projector.
    
    Applies pooling AFTER vision encoding but BEFORE language model processing,
    reducing the number of visual tokens passed to the LM.
    """
    
    def __init__(
        self,
        model_path: str = "lusxvr/nanoVLM",
        device: str = "auto",
        resolution: int = None,
        seed: int = 1234,
        pooling_type: str = "none",
        pool_kernel: int = 2,
        pool_stride: int = 2,
        pool_grid: int = 8
    ):
        """
        Initialize evaluator with token pooling.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to use (auto, cuda, mps, cpu)
            resolution: Optional target resolution for vision encoder
            seed: Random seed for reproducibility
            pooling_type: One of "none", "avg", "max", "adaptive"
            pool_kernel: Kernel size for avg/max pooling
            pool_stride: Stride for avg/max pooling
            pool_grid: Target grid size for adaptive pooling (e.g., 8 for 8×8 = 64 tokens)
        """
        # Initialize base evaluator (loads model)
        super().__init__(model_path, device, resolution, seed)
        
        # Store pooling config
        self.pooling_type = pooling_type
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pool_grid = pool_grid
        
        # Apply token pooling if requested
        if pooling_type != "none":
            self._apply_token_pooling()
    
    def _apply_token_pooling(self):
        """
        Apply token pooling by replacing the modality projector.
        
        This modifies the model's MP (Modality Projector) to include pooling
        between the vision encoder output and the language model input.
        """
        print(f"\n{'='*60}")
        print(f"APPLYING TOKEN POOLING")
        print(f"{'='*60}")
        print(f"Pooling type: {self.pooling_type.upper()}")
        
        # Get original token count
        original_tokens = self.model.cfg.mp_image_token_length
        
        # Calculate new token count based on pooling type
        if self.pooling_type == "adaptive":
            new_token_count = self.pool_grid * self.pool_grid
            print(f"Target grid: {self.pool_grid}×{self.pool_grid}")
        elif self.pooling_type in ["avg", "max"]:
            # Calculate output size from pooling
            input_grid = int(original_tokens ** 0.5)
            output_grid = (input_grid - self.pool_kernel) // self.pool_stride + 1
            new_token_count = output_grid * output_grid
            print(f"Kernel: {self.pool_kernel}, Stride: {self.pool_stride}")
            print(f"Grid: {input_grid}×{input_grid} → {output_grid}×{output_grid}")
        else:
            new_token_count = original_tokens
        
        print(f"Tokens: {original_tokens} → {new_token_count}")
        reduction_pct = (1 - new_token_count/original_tokens)*100
        print(f"Reduction: {reduction_pct:.1f}%")
        
        # Update config with pooling parameters
        self.model.cfg.mp_token_pooling = self.pooling_type
        self.model.cfg.mp_pool_kernel = self.pool_kernel
        self.model.cfg.mp_pool_stride = self.pool_stride
        self.model.cfg.mp_target_grid = self.pool_grid
        self.model.cfg.mp_image_token_length = new_token_count
        
        # Create new modality projector with pooling
        new_projector = ModalityProjector(self.model.cfg)
        
        # Copy weights from original projector (only the linear projection)
        # The pooling layer is randomly initialized (but deterministic operations like avg/max don't use weights)
        new_projector.proj.weight.data.copy_(self.model.MP.proj.weight.data)
        if hasattr(self.model.MP.proj, 'bias') and self.model.MP.proj.bias is not None:
            new_projector.proj.bias.data.copy_(self.model.MP.proj.bias.data)
        
        # Replace modality projector in model
        self.model.MP = new_projector.to(self.device)
        
        # Estimate FLOP impact
        # Vision encoder FLOPs stay the same (still processes full image)
        # Modality projector FLOPs scale with token count
        # Language model FLOPs scale with input token count (prefill + generation)
        
        print(f"\nExpected FLOP impact:")
        print(f"  Vision encoder: No change (still processes full image)")
        print(f"  Modality projector: {reduction_pct:.1f}% reduction")
        print(f"  Language model (prefill): ~{reduction_pct:.1f}% reduction")
        print(f"  Language model (generation): Small reduction (cached context)")
        print(f"  Overall: ~{reduction_pct * 0.3:.1f}-{reduction_pct * 0.5:.1f}% reduction")
        print(f"    (depends on generation length)")
        print(f"✓ Token pooling applied successfully")
        print(f"{'='*60}")
    
    def format_mcq_prompt(self, question: str, choices: list) -> str:
        """Format MCQ prompt."""
        prompt = f"Look at the image carefully and answer this question:\n\n{question}\n\nChoose the best answer from these options:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nRespond with only the letter (A, B, C, or D) of the correct answer."
        return prompt

    def format_open_answer_prompt(self, question: str) -> str:
        """Format open-answer prompt."""
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
        
        # Pattern 3: Letter followed by period at end
        answer_match = re.search(r'\b([A-D])\.\s*$', text)
        if answer_match:
            return answer_match.group(1)
        
        # Pattern 4: Standalone letter at end
        words = text.split()
        if words:
            last_word = words[-1].strip('.,!?:;')
            if len(last_word) == 1 and last_word in ['A', 'B', 'C', 'D']:
                return last_word
        
        # Pattern 5: Look for first isolated letter
        answer_match = re.search(r'\b([A-D])\b', text)
        if answer_match:
            return answer_match.group(1)
        
        # Default to A if no valid answer found
        return 'A'
    
    def get_eval_name(self) -> str:
        """Get evaluation name for logging."""
        if self.pooling_type == "none":
            return "Baseline (No Pooling)"
        elif self.pooling_type == "adaptive":
            return f"Adaptive Pooling ({self.pool_grid}×{self.pool_grid})"
        else:
            return f"{self.pooling_type.upper()} Pooling (k={self.pool_kernel}, s={self.pool_stride})"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate nanoVLM with token pooling"
    )
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="lusxvr/nanoVLM",
                        help="Path to model or HuggingFace model ID")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Target resolution for vision encoder")
    
    # Pooling arguments
    parser.add_argument("--pooling", type=str, default="none",
                        choices=["none", "avg", "max", "adaptive"],
                        help="Token pooling type")
    parser.add_argument("--kernel", type=int, default=2,
                        help="Kernel size for avg/max pooling")
    parser.add_argument("--stride", type=int, default=2,
                        help="Stride for avg/max pooling")
    parser.add_argument("--grid", type=int, default=8,
                        help="Target grid size for adaptive pooling")
    
    # Evaluation arguments
    parser.add_argument("--split", type=str, default="validation",
                        choices=["train", "validation", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--mode", type=str, default="mcq",
                        choices=["mcq", "oa", "both"],
                        help="Evaluation mode")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Set default output file
    if args.output is None:
        if args.pooling == "none":
            args.output = "results/baseline_results.json"
        elif args.pooling == "adaptive":
            args.output = f"results/token_pooling_{args.pooling}_grid{args.grid}_results.json"
        else:
            args.output = f"results/token_pooling_{args.pooling}_k{args.kernel}_s{args.stride}_results.json"
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Create evaluator with token pooling
    evaluator = TokenPooledEvaluator(
        model_path=args.model_path,
        device=args.device,
        resolution=args.resolution,
        pooling_type=args.pooling,
        pool_kernel=args.kernel,
        pool_stride=args.stride,
        pool_grid=args.grid
    )
    
    # Profile model
    profile_dict = evaluator.profile_model()
    
    # Run evaluations
    print("\n" + "="*60)
    print(f"RUNNING EVALUATION - {evaluator.get_eval_name()}")
    print("="*60)
    
    results = {**profile_dict}
    
    if args.mode in ["mcq", "both"]:
        mcq_results = evaluator.evaluate_mcq(
            split=args.split,
            max_samples=args.max_samples
        )
        results.update(mcq_results)
    
    if args.mode in ["oa", "both"]:
        oa_results = evaluator.evaluate_open_answer(
            split=args.split,
            max_samples=args.max_samples
        )
        results.update(oa_results)
    
    # Add config to results
    results["model_path"] = args.model_path
    results["split"] = args.split
    results["mode"] = args.mode
    results["resolution"] = evaluator.model.cfg.vit_img_size
    results["pooling_config"] = {
        "type": args.pooling,
        "kernel": args.kernel if args.pooling in ["avg", "max"] else None,
        "stride": args.stride if args.pooling in ["avg", "max"] else None,
        "grid": args.grid if args.pooling == "adaptive" else None,
        "original_tokens": 64,  # Default from config
        "final_tokens": evaluator.model.cfg.mp_image_token_length
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Configuration: {evaluator.get_eval_name()}")
    
    print(f"\nToken Pooling:")
    print(f"  Original tokens: {results['pooling_config']['original_tokens']}")
    print(f"  Final tokens: {results['pooling_config']['final_tokens']}")
    token_reduction = (1 - results['pooling_config']['final_tokens'] / results['pooling_config']['original_tokens']) * 100
    print(f"  Reduction: {token_reduction:.1f}%")
    
    print(f"\nModel Profile:")
    print(f"  Total params: {results.get('total_params', 0):,}")
    if 'estimated_tflops_per_forward' in results:
        print(f"  Total FLOPs: {results['estimated_tflops_per_forward']:.4f} TFLOPs")
    if 'vision_tflops' in results:
        print(f"    Vision: {results['vision_tflops']:.4f} TFLOPs")
    if 'language_tflops' in results:
        print(f"    Language: {results['language_tflops']:.4f} TFLOPs")
    if 'visual_tokens' in results:
        print(f"  Visual tokens: {results['visual_tokens']}")
    
    if args.mode in ["mcq", "both"]:
        print(f"\nMCQ Evaluation:")
        print(f"  Accuracy: {results['mcq_accuracy']*100:.2f}% "
              f"({results['mcq_correct']}/{results['mcq_total']})")
        print(f"  Avg Latency: {results['mcq_avg_latency']*1000:.2f} ms/sample")
    
    if args.mode in ["oa", "both"]:
        print(f"\nOpen-Answer Evaluation:")
        print(f"  Accuracy: {results['oa_accuracy']*100:.2f}% "
              f"({results['oa_correct']}/{results['oa_total']})")
        print(f"  Avg Latency: {results['oa_avg_latency']*1000:.2f} ms/sample")
    
    print("="*60)


if __name__ == "__main__":
    main()
