#!/usr/bin/env python3
"""
Base Evaluator for nanoVLM on A-OKVQA.
Provides core evaluation loop - extend with custom prompt formatting.
"""

from data.processors import get_tokenizer, get_image_processor, get_image_string
from models.vision_language_model import VisionLanguageModel
import sys
import time
import random
from pathlib import Path
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image


def seed_everything(seed=1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# Add nanoVLM to path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "models" / "nanoVLM"))


class BaseEvaluator(ABC):
    """Base class for nanoVLM evaluation. Subclass and implement prompt formatting methods."""

    def __init__(self, model_path: str = "lusxvr/nanoVLM", device: str = "auto", resolution: int = None, seed: int = 1234):
        """Initialize evaluator with nanoVLM model.

        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to use (auto, cuda, mps, cpu)
            resolution: Optional target resolution for vision encoder. If provided,
                       updates model config to process images at this resolution.
            seed: Random seed for reproducibility
        """
        # Set random seeds for reproducibility
        seed_everything(seed)

        self.device = self._get_device(device)
        print(f"Using device: {self.device}")

        print(f"Loading model from {model_path}...")
        self.model = VisionLanguageModel.from_pretrained(
            model_path).to(self.device)
        self.model.eval()

        # Update resolution if provided
        if resolution is not None:
            original_resolution = self.model.cfg.vit_img_size
            if resolution != original_resolution:
                print(
                    f"Updating split size from {original_resolution}×{original_resolution} to {resolution}×{resolution}")

                # Update the size of each image split/crop (not the max image size)
                self.model.cfg.vit_img_size = resolution
                # Keep max_img_size at 2048 for dynamic resize (Global-and-Split strategy)
                # This ensures images are still resized to max 2048px before splitting
                # Note: resize_to_max_side_len should remain True (default in config)

                # Interpolate positional embeddings to match new resolution
                self._interpolate_positional_embeddings(
                    original_resolution, resolution)

                # Update image token count for new resolution
                # Each split produces: (resolution/patch_size)^2 patches
                # After 4×4 pixel-shuffle compression: patches / 16 tokens per split
                patch_size = self.model.cfg.vit_patch_size
                num_patches = (resolution // patch_size) ** 2
                scale_factor = self.model.cfg.mp_pixel_shuffle_factor
                new_token_count = num_patches // (scale_factor ** 2)
                self.model.cfg.mp_image_token_length = new_token_count

                print(
                    f"Each {resolution}×{resolution} split now produces {new_token_count} visual tokens")
                print(
                    f"(from {num_patches} patches with {scale_factor}×{scale_factor} compression)")

        # Get processors
        self.tokenizer = get_tokenizer(
            self.model.cfg.lm_tokenizer,
            self.model.cfg.vlm_extra_tokens,
            self.model.cfg.lm_chat_template
        )

        resize_to_max_side_len = getattr(
            self.model.cfg, "resize_to_max_side_len", False)
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

    def _interpolate_positional_embeddings(self, old_resolution: int, new_resolution: int):
        """
        Interpolate positional embeddings to match new resolution.
        Required when changing image resolution to maintain model performance.
        """
        import torch.nn as nn

        patch_size = self.model.cfg.vit_patch_size
        old_num_patches = (old_resolution // patch_size) ** 2
        new_num_patches = (new_resolution // patch_size) ** 2

        if old_num_patches == new_num_patches:
            return

        print(
            f"Interpolating positional embeddings: {old_num_patches} → {new_num_patches} patches")

        # Get current position embeddings
        pos_embed = self.model.vision_encoder.patch_embedding.position_embedding.data

        # Handle CLS token if present
        has_cls = self.model.vision_encoder.patch_embedding.cls_flag
        if has_cls:
            cls_pos_embed = pos_embed[:, :1, :]
            pos_embed = pos_embed[:, 1:, :]

        # Reshape to 2D grid
        old_grid_size = int(old_num_patches ** 0.5)
        new_grid_size = int(new_num_patches ** 0.5)
        embed_dim = pos_embed.shape[2]

        pos_embed = pos_embed.reshape(
            1, old_grid_size, old_grid_size, embed_dim)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, embed_dim, H, W]

        # Interpolate using bilinear
        pos_embed = torch.nn.functional.interpolate(
            pos_embed,
            size=(new_grid_size, new_grid_size),
            mode='bilinear',
            align_corners=False
        )

        # Reshape back
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [1, H, W, embed_dim]
        pos_embed = pos_embed.reshape(1, new_num_patches, embed_dim)

        # Add CLS token back if needed
        if has_cls:
            pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)

        # Update the model's positional embeddings
        self.model.vision_encoder.patch_embedding.position_embedding = nn.Parameter(
            pos_embed)
        print(
            f"Updated positional embeddings to {new_grid_size}×{new_grid_size} grid")

    @abstractmethod
    def format_mcq_prompt(self, question: str, choices: list) -> str:
        """Format MCQ prompt. Must be implemented by subclass."""
        pass

    @abstractmethod
    def format_open_answer_prompt(self, question: str) -> str:
        """Format open-answer prompt. Must be implemented by subclass."""
        pass

    @abstractmethod
    def extract_answer(self, text: str) -> str:
        """Extract answer letter from model output. Must be implemented by subclass."""
        pass

    def evaluate_mcq(self, split: str = "validation", max_samples: int = None):
        """Run MCQ evaluation on the A-OKVQA dataset."""
        print(f"\n{'='*60}")
        print(f"MCQ EVALUATION ({self.get_eval_name()})")
        print(f"{'='*60}")
        print(f"Loading A-OKVQA {split} split...")
        dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        correct = 0
        total = 0
        latencies = []

        print(f"Evaluating on {len(dataset)} samples...")
        with torch.no_grad():
            for idx, example in enumerate(tqdm(dataset, desc="MCQ Evaluation")):
                try:
                    # Process image
                    image = example["image"]
                    if not isinstance(image, Image.Image):
                        continue

                    image = image.convert("RGB")
                    processed_image, splitted_image_ratio = self.image_processor(
                        image)

                    # Remove global image token if tokenizer doesn't support it
                    if not hasattr(self.tokenizer, "global_image_token") and \
                       splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1:
                        processed_image = processed_image[1:]

                    # Create prompt using subclass implementation
                    question = example["question"]
                    choices = example["choices"]
                    prompt = self.format_mcq_prompt(question, choices)

                    # Create input with image string
                    image_string = get_image_string(
                        self.tokenizer,
                        [splitted_image_ratio],
                        self.model.cfg.mp_image_token_length
                    )

                    messages = [
                        {"role": "user", "content": image_string + prompt}]
                    encoded_prompt = self.tokenizer.apply_chat_template(
                        [messages],
                        tokenize=True,
                        add_generation_prompt=True
                    )
                    tokens = torch.tensor(encoded_prompt).to(self.device)
                    img_tensor = processed_image.to(self.device)

                    # Generate answer
                    start_time = time.time()
                    generated = self.model.generate(
                        tokens, img_tensor, max_new_tokens=10, greedy=True)
                    end_time = time.time()

                    latencies.append(end_time - start_time)

                    # Decode and extract answer using subclass implementation
                    output = self.tokenizer.batch_decode(
                        generated, skip_special_tokens=True)[0]
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
                        print(
                            f"Predicted: {predicted_letter}, Correct: {correct_letter}")
                        print(
                            f"Match: {'✓' if predicted_letter == correct_letter else '✗'}")

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
        print(f"OPEN-ANSWER EVALUATION ({self.get_eval_name()})")
        print(f"{'='*60}")
        print(f"Loading A-OKVQA {split} split...")
        dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        correct = 0
        total = 0
        latencies = []

        print(f"Evaluating on {len(dataset)} samples...")
        with torch.no_grad():
            for idx, example in enumerate(tqdm(dataset, desc="Open-Answer Evaluation")):
                try:
                    # Process image
                    image = example["image"]
                    if not isinstance(image, Image.Image):
                        continue

                    image = image.convert("RGB")
                    processed_image, splitted_image_ratio = self.image_processor(
                        image)

                    # Remove global image token if tokenizer doesn't support it
                    if not hasattr(self.tokenizer, "global_image_token") and \
                       splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1:
                        processed_image = processed_image[1:]

                    # Create prompt using subclass implementation
                    question = example["question"]
                    prompt = self.format_open_answer_prompt(question)

                    # Create input with image string
                    image_string = get_image_string(
                        self.tokenizer,
                        [splitted_image_ratio],
                        self.model.cfg.mp_image_token_length
                    )

                    messages = [
                        {"role": "user", "content": image_string + prompt}]
                    encoded_prompt = self.tokenizer.apply_chat_template(
                        [messages],
                        tokenize=True,
                        add_generation_prompt=True
                    )
                    tokens = torch.tensor(encoded_prompt).to(self.device)
                    img_tensor = processed_image.to(self.device)

                    # Generate answer
                    start_time = time.time()
                    generated = self.model.generate(
                        tokens, img_tensor, max_new_tokens=20, greedy=True)
                    end_time = time.time()

                    latencies.append(end_time - start_time)

                    # Decode answer
                    output = self.tokenizer.batch_decode(
                        generated, skip_special_tokens=True)[0]
                    predicted_answer = output.strip().lower()

                    # Get correct answer(s) - A-OKVQA has multiple valid answers
                    direct_answers = example["direct_answers"]
                    if isinstance(direct_answers, str):
                        import ast
                        try:
                            direct_answers = ast.literal_eval(direct_answers)
                        except:
                            direct_answers = [direct_answers]

                    correct_answers = [ans.strip().lower()
                                       for ans in direct_answers]
                    correct_choice = example["choices"][example["correct_choice_idx"]].lower(
                    )

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
                        print(f"Choices: {example['choices']}")
                        print(f"Model output: {output}")
                        print(
                            f"Predicted: {predicted_answer}, Correct: {correct_answers[:3]}")
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

    def profile_model(self, test_image_path: str = "test/test_image.webp", test_query_path: str = "test/test_query.txt"):
        """Profile model parameters and FLOPs using both manual calculation and actual profiling.

        Args:
            test_image_path: Path to test image for profiling
            test_query_path: Path to test query file for profiling
        """
        print("\nProfiling model...")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        results = {
            "total_params": total_params,
            "trainable_params": trainable_params,
        }

        # Try to use FLOPCounter for detailed profiling
        try:
            # Import FLOPCounter
            sys.path.insert(0, str(Path(__file__).parent))
            from flop_counter import FLOPCounter

            # Create a FLOPCounter instance using the already-loaded model
            flop_counter = FLOPCounter.__new__(FLOPCounter)
            flop_counter.model = self.model
            flop_counter.cfg = self.model.cfg
            flop_counter.device = self.device
            flop_counter.tokenizer = self.tokenizer
            flop_counter.image_processor = self.image_processor

            # 1. Manual ViT FLOP calculation
            print("\n1. Calculating vision encoder FLOPs (manual)...")
            if Path(test_image_path).exists():
                manual_vit_flops = flop_counter.calculate_vit_flops_manual(
                    image_path=test_image_path)
            else:
                print(
                    f"Warning: Test image not found at {test_image_path}, using n_images=1")
                manual_vit_flops = flop_counter.calculate_vit_flops_manual(
                    n_images=1)

            results["vit_manual_tflops"] = manual_vit_flops['vit_total_tflops']
            results["vit_manual_gflops"] = manual_vit_flops['vit_total_gflops']
            print(
                f"Vision encoder (manual): {manual_vit_flops['vit_total_tflops']:.4f} TFLOPs")

            # 2. Automatic ViT FLOP profiling
            if Path(test_image_path).exists():
                print("\n2. Profiling vision encoder FLOPs (automatic)...")
                auto_vit_flops = flop_counter.calculate_vit_flops_automatic(
                    test_image_path)
                results["vit_auto_tflops"] = auto_vit_flops['vit_total_tflops']
                results["vit_auto_gflops"] = auto_vit_flops['vit_total_gflops']
                print(
                    f"Vision encoder (automatic): {auto_vit_flops['vit_total_tflops']:.4f} TFLOPs")

            # 3. Generation FLOP profiling
            if Path(test_image_path).exists() and Path(test_query_path).exists():
                print("\n3. Profiling generation FLOPs...")
                with open(test_query_path, 'r') as f:
                    question = f.read().strip()

                gen_flops = flop_counter.calculate_generation_flops(
                    test_image_path,
                    question,
                    n_tokens=3,
                    warmup_steps=0
                )
                results["generation_tflops"] = gen_flops['generation_total_tflops']
                results["generation_gflops"] = gen_flops['generation_total_gflops']
                results["generation_flops_per_token"] = gen_flops['generation_flops_per_token']
                print(
                    f"Generation (3 tokens): {gen_flops['generation_total_tflops']:.4f} TFLOPs")

            # Use vision encoder TFLOPs as the main estimate
            results["estimated_tflops_per_forward"] = results.get(
                "vit_manual_tflops", 0)

        except Exception as e:
            print(f"\nWarning: Detailed FLOP profiling failed ({e})")
            print("Using rough approximation...")
            # Fallback to rough approximation
            seq_len = 100
            estimated_flops = 2 * total_params * seq_len
            estimated_tflops = estimated_flops / 1e12
            results["estimated_tflops_per_forward"] = estimated_tflops

        return results

    def get_eval_name(self) -> str:
        """Return evaluation name for display. Override in subclass."""
        return self.__class__.__name__
