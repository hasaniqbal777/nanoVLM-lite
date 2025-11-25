#!/usr/bin/env python3
"""
Baseline evaluation script for nanoVLM on A-OKVQA.
Measures accuracy, parameters, and latency.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# Add nanoVLM to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "models" / "nanoVLM"))

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string


class BaselineEvaluator:
    def __init__(self, model_path: str = "lusxvr/nanoVLM", device: str = "auto"):
        """Initialize evaluator with nanoVLM model."""
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")
        
        print(f"Loading model from {model_path}...")
        self.model = VisionLanguageModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Get processors
        self.tokenizer = get_tokenizer(
            self.model.cfg.lm_tokenizer, 
            self.model.cfg.vlm_extra_tokens, 
            self.model.cfg.lm_chat_template
        )
        
        resize_to_max_side_len = getattr(self.model.cfg, "resize_to_max_side_len", False)
        self.image_processor = get_image_processor(
            self.model.cfg.max_img_size, 
            self.model.cfg.vit_img_size, 
            resize_to_max_side_len
        )
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def format_mcq_prompt(self, question: str, choices: list) -> str:
        """Format question and choices into a prompt matching nanoVLM's training format."""
        # Format similar to their benchmark formatting
        prompt = f"{question}\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer with the letter."
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """Extract the answer letter from model output."""
        text = text.strip().upper()
        
        # First try to find pattern like "Answer: X" or "Answer:X"
        import re
        answer_match = re.search(r'ANSWER\s*:\s*([A-D])', text)
        if answer_match:
            return answer_match.group(1)
        
        # Try to find standalone letter at end
        words = text.split()
        if words:
            last_word = words[-1].strip('.,!?:;')
            if len(last_word) == 1 and last_word in ['A', 'B', 'C', 'D']:
                return last_word
        
        # Look for first letter that's not part of "ANSWER"
        # Skip the word "ANSWER" if present
        text_without_answer = text.replace('ANSWER', '').replace(':', '')
        for char in text_without_answer:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        # Default to A if no valid answer found
        return 'A'
    
    def evaluate_mcq(self, split: str = "validation", max_samples: int = None):
        """Run MCQ evaluation on the A-OKVQA dataset."""
        print(f"\n{'='*60}")
        print("MCQ EVALUATION")
        print(f"{'='*60}")
        print(f"Loading A-OKVQA {split} split...")
        dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        correct = 0
        total = 0
        latencies = []
        
        print(f"\nEvaluating on {len(dataset)} samples...")
        with torch.no_grad():
            for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
                try:
                    # Process image
                    image = example["image"]
                    if not isinstance(image, Image.Image):
                        continue
                    
                    image = image.convert("RGB")
                    processed_image, splitted_image_ratio = self.image_processor(image)
                    
                    # Remove global image token if tokenizer doesn't support it
                    if not hasattr(self.tokenizer, "global_image_token") and \
                       splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1:
                        processed_image = processed_image[1:]
                    
                    # Create prompt
                    question = example["question"]
                    choices = example["choices"]
                    prompt = self.format_mcq_prompt(question, choices)
                    
                    # Create input with image string
                    image_string = get_image_string(
                        self.tokenizer, 
                        [splitted_image_ratio], 
                        self.model.cfg.mp_image_token_length
                    )
                    
                    messages = [{"role": "user", "content": image_string + prompt}]
                    encoded_prompt = self.tokenizer.apply_chat_template(
                        [messages], 
                        tokenize=True, 
                        add_generation_prompt=True
                    )
                    tokens = torch.tensor(encoded_prompt).to(self.device)
                    img_tensor = processed_image.to(self.device)
                    
                    # Generate answer
                    start_time = time.time()
                    generated = self.model.generate(tokens, img_tensor, max_new_tokens=10, greedy=True)
                    end_time = time.time()
                    
                    latencies.append(end_time - start_time)
                    
                    # Decode and extract answer
                    output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                    predicted_letter = self.extract_answer(output)
                    
                    # Get correct answer letter
                    correct_idx = example["correct_choice_idx"]
                    correct_letter = chr(65 + correct_idx)
                    
                    if predicted_letter == correct_letter:
                        correct += 1
                    total += 1
                    
                    # Print first few for debugging
                    if idx < 3:
                        print(f"\n--- Example {idx+1} ---")
                        print(f"Q: {question}")
                        print(f"Choices: {choices}")
                        print(f"Model output: {output}")
                        print(f"Predicted: {predicted_letter}, Correct: {correct_letter}")
                
                except Exception as e:
                    print(f"\nError processing example {idx}: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "mcq_accuracy": accuracy,
            "mcq_correct": correct,
            "mcq_total": total,
            "mcq_avg_latency": avg_latency
        }
    
    def evaluate_open_answer(self, split: str = "validation", max_samples: int = None):
        """Run open-answer evaluation on the A-OKVQA dataset."""
        print(f"\n{'='*60}")
        print("OPEN-ANSWER EVALUATION")
        print(f"{'='*60}")
        print(f"Loading A-OKVQA {split} split...")
        dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        correct = 0
        total = 0
        latencies = []
        
        print(f"\nEvaluating on {len(dataset)} samples...")
        with torch.no_grad():
            for idx, example in enumerate(tqdm(dataset, desc="Evaluating (Open-Answer)")):
                try:
                    # Process image
                    image = example["image"]
                    if not isinstance(image, Image.Image):
                        continue
                    
                    image = image.convert("RGB")
                    processed_image, splitted_image_ratio = self.image_processor(image)
                    
                    # Remove global image token if tokenizer doesn't support it
                    if not hasattr(self.tokenizer, "global_image_token") and \
                       splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1:
                        processed_image = processed_image[1:]
                    
                    # Create open-ended prompt (no choices shown)
                    question = example["question"]
                    prompt = f"{question}\nProvide a short answer."
                    
                    # Create input with image string
                    image_string = get_image_string(
                        self.tokenizer, 
                        [splitted_image_ratio], 
                        self.model.cfg.mp_image_token_length
                    )
                    
                    messages = [{"role": "user", "content": image_string + prompt}]
                    encoded_prompt = self.tokenizer.apply_chat_template(
                        [messages], 
                        tokenize=True, 
                        add_generation_prompt=True
                    )
                    tokens = torch.tensor(encoded_prompt).to(self.device)
                    img_tensor = processed_image.to(self.device)
                    
                    # Generate answer
                    start_time = time.time()
                    generated = self.model.generate(tokens, img_tensor, max_new_tokens=20, greedy=True)
                    end_time = time.time()
                    
                    latencies.append(end_time - start_time)
                    
                    # Decode answer
                    output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                    predicted_answer = output.strip().lower()
                    
                    # Get correct answer(s) - A-OKVQA has multiple valid answers
                    correct_answers = [ans.lower() for ans in example["direct_answers"]]
                    correct_choice = example["choices"][example["correct_choice_idx"]].lower()
                    
                    # Check if prediction matches any correct answer or the correct choice
                    is_correct = any(ans in predicted_answer or predicted_answer in ans 
                                    for ans in correct_answers + [correct_choice])
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    # Print first few for debugging
                    if idx < 3:
                        print(f"\n--- Example {idx+1} ---")
                        print(f"Q: {question}")
                        print(f"Model output: {output}")
                        print(f"Correct answers: {correct_answers[:3]}")
                        print(f"Match: {'✓' if is_correct else '✗'}")
                
                except Exception as e:
                    print(f"\nError processing example {idx}: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "oa_accuracy": accuracy,
            "oa_correct": correct,
            "oa_total": total,
            "oa_avg_latency": avg_latency
        }
    
    def profile_model(self):
        """Profile model parameters and estimate FLOPs."""
        print("\nProfiling model...")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Estimate FLOPs (rough approximation)
        # For transformers: FLOPs ≈ 2 * params * sequence_length
        # This is a simplified estimate
        seq_len = 100  # typical sequence length
        estimated_flops = 2 * total_params * seq_len
        estimated_tflops = estimated_flops / 1e12
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "estimated_tflops_per_forward": estimated_tflops
        }
    
    def save_results(self, results, output_path):
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for nanoVLM on A-OKVQA")
    parser.add_argument("--model-path", type=str, default="lusxvr/nanoVLM",
                        help="Path to model or HuggingFace model ID (lusxvr/nanoVLM or lusxvr/nanoVLM-450M)")
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
    parser.add_argument("--output", type=str, default="results/baseline_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = BaselineEvaluator(args.model_path, args.device)
    
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
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Mode: {mode}")
    print(f"Parameters: {results['total_params']:,}")
    print(f"Estimated TFLOPs: {results['estimated_tflops_per_forward']:.4f}")
    
    if mode in ["mcq", "both"]:
        print()
        print("MCQ Evaluation:")
        print(f"  Accuracy: {results['mcq_accuracy']*100:.2f}% ({results['mcq_correct']}/{results['mcq_total']})")
        print(f"  Avg Latency: {results['mcq_avg_latency']*1000:.2f} ms/sample")
        print(f"  65% Threshold: {results['mcq_accuracy']*100*0.65:.2f}%")
    
    if mode in ["open-answer", "both"]:
        print()
        print("Open-Answer Evaluation:")
        print(f"  Accuracy: {results['oa_accuracy']*100:.2f}% ({results['oa_correct']}/{results['oa_total']})")
        print(f"  Avg Latency: {results['oa_avg_latency']*1000:.2f} ms/sample")
        print(f"  65% Threshold: {results['oa_accuracy']*100*0.65:.2f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()
