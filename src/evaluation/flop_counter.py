#!/usr/bin/env python3
"""
FLOP Counter for nanoVLM
Measures FLOPs for vision encoder and language generation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Add nanoVLM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "nanoVLM"))

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string


class FLOPCounter:
    def __init__(self, model_path: str = "lusxvr/nanoVLM", device: str = "auto"):
        """Initialize FLOP counter with nanoVLM model."""
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")
        
        print(f"Loading model from {model_path}...")
        self.model = VisionLanguageModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        self.cfg = self.model.cfg
        
        # Get processors
        self.tokenizer = get_tokenizer(
            self.cfg.lm_tokenizer, 
            self.cfg.vlm_extra_tokens, 
            self.cfg.lm_chat_template
        )
        
        resize_to_max_side_len = getattr(self.cfg, "resize_to_max_side_len", False)
        self.image_processor = get_image_processor(
            self.cfg.max_img_size, 
            self.cfg.vit_img_size, 
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
    
    def calculate_vit_flops_manual(self, image_path: str = None, n_images: int = None) -> dict:
        """
        Calculate ViT FLOPs manually based on architecture.
        
        Args:
            image_path: Path to image to determine actual split (optional)
            n_images: Number of images processed (overrides image_path)
            
        Returns:
            Dictionary with detailed FLOP counts
        """
        cfg = self.cfg
        
        # Determine number of images
        n_h, n_w = None, None
        if n_images is None and image_path is not None:
            image = Image.open(image_path).convert("RGB")
            _, splitted_image_ratio = self.image_processor(image)
            n_h, n_w = splitted_image_ratio
            n_images = n_h * n_w
        elif n_images is None:
            n_images = 1
        
        # ViT configuration
        T = (cfg.vit_img_size // cfg.vit_patch_size) ** 2  # Number of patches
        D = cfg.vit_hidden_dim
        H = cfg.vit_n_heads
        d = D // H
        I = cfg.vit_inter_dim
        L = cfg.vit_n_blocks
        
        # 2 flops per MAC (Multiply-Accumulate)
        # Patch embedding: Conv2d(3, D, kernel_size=patch_size, stride=patch_size)
        patch_conv = 2 * (3 * cfg.vit_patch_size * cfg.vit_patch_size) * D * T
        
        # Per-layer computations
        # QKV projection: 3 separate linear layers (Q, K, V)
        proj_qkv = 6 * T * D * D
        
        # Output projection
        proj_out = 2 * T * D * D
        
        # Attention: Q@K^T and softmax(Q@K^T)@V
        attn = 4 * H * (T ** 2) * d
        
        # MLP: two linear layers
        mlp = 4 * T * D * I
        
        # Total per layer
        per_layer = proj_qkv + proj_out + attn + mlp
        
        # Total for ViT
        vit_per_image = patch_conv + L * per_layer
        vit_total = vit_per_image * n_images
        
        results = {
            'N_img': n_images,
            'T_patches': T,
            'D_hidden': D,
            'H_heads': H,
            'd_per_head': d,
            'I_intermediate': I,
            'L_layers': L,
            'patch_conv_flops': patch_conv,
            'proj_qkv_flops': proj_qkv,
            'proj_out_flops': proj_out,
            'attn_flops': attn,
            'mlp_flops': mlp,
            'per_layer_flops': per_layer,
            'vit_per_image_flops': vit_per_image,
            'vit_total_flops': vit_total,
            'vit_total_gflops': vit_total / 1e9,
            'vit_total_tflops': vit_total / 1e12
        }
        
        if n_h is not None and n_w is not None:
            results['n_h'] = n_h
            results['n_w'] = n_w
        
        return results
    
    def calculate_vit_flops_automatic(self, image_path: str) -> dict:
        """
        Calculate ViT FLOPs automatically using profiling.
        
        Args:
            image_path: Path to test image
            
        Returns:
            Dictionary with FLOP counts
        """
        from torch.utils.flop_counter import FlopCounterMode
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        processed_image, splitted_image_ratio = self.image_processor(image)
        
        n_h, n_w = splitted_image_ratio
        N_img = n_h * n_w
        
        # Move to device
        img_tensor = processed_image.to(self.device)
        
        # Count FLOPs for vision encoder
        flop_counter = FlopCounterMode(display=False)
        
        with torch.no_grad():
            with flop_counter:
                _ = self.model.vision_encoder(img_tensor)
        
        total_flops = flop_counter.get_total_flops()
        
        return {
            'n_h': n_h,
            'n_w': n_w,
            'N_img': int(N_img),
            'vit_total_flops': total_flops,
            'vit_total_gflops': total_flops / 1e9,
            'vit_total_tflops': total_flops / 1e12
        }
    
    def calculate_generation_flops(
        self, 
        image_path: str, 
        question: str, 
        n_tokens: int = 3,
        warmup_steps: int = 0
    ) -> dict:
        """
        Calculate FLOPs for generation with optional warmup.
        
        Args:
            image_path: Path to test image
            question: Question text
            n_tokens: Number of tokens to generate
            warmup_steps: Number of warmup generation steps
            
        Returns:
            Dictionary with FLOP counts and timing
        """
        from torch.utils.flop_counter import FlopCounterMode
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        processed_image, splitted_image_ratio = self.image_processor(image)
        
        # Remove global image token if tokenizer doesn't support it
        if not hasattr(self.tokenizer, "global_image_token") and \
           splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1:
            processed_image = processed_image[1:]
        
        # Create prompt
        image_string = get_image_string(
            self.tokenizer, 
            [splitted_image_ratio], 
            self.cfg.mp_image_token_length
        )
        
        messages = [{"role": "user", "content": image_string + question}]
        encoded_prompt = self.tokenizer.apply_chat_template(
            [messages], 
            tokenize=True, 
            add_generation_prompt=True
        )
        tokens = torch.tensor(encoded_prompt).to(self.device)
        img_tensor = processed_image.to(self.device)
        
        S_prefill = tokens.size(1)
        
        results = {
            'S_prefill_tokens': S_prefill,
            'n_tokens': n_tokens,
            'warmup_steps': warmup_steps
        }
        
        # Warmup if requested
        if warmup_steps > 0:
            print(f"\nRunning {warmup_steps} warmup steps...")
            with torch.no_grad():
                for _ in range(warmup_steps):
                    _ = self.model.generate(tokens, img_tensor, max_new_tokens=1, greedy=True)
        
        # Measure generation FLOPs
        print(f"\nMeasuring FLOPs for {n_tokens} token generation...")
        flop_counter = FlopCounterMode(display=False)
        
        with torch.no_grad():
            start_time = time.time()
            with flop_counter:
                _ = self.model.generate(tokens, img_tensor, max_new_tokens=n_tokens, greedy=True)
            end_time = time.time()
        
        total_flops = flop_counter.get_total_flops()
        latency = end_time - start_time
        
        results.update({
            'generation_total_flops': total_flops,
            'generation_total_gflops': total_flops / 1e9,
            'generation_total_tflops': total_flops / 1e12,
            'generation_flops_per_token': total_flops / n_tokens,
            'latency_seconds': latency,
            'latency_per_token_ms': (latency / n_tokens) * 1000
        })
        
        return results


def print_results(title: str, results: dict):
    """Pretty print results."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    for key, value in results.items():
        if isinstance(value, float):
            if 'flops' in key.lower() and value > 1e6:
                print(f"{key}: {value:,.0f}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="FLOP counter for nanoVLM")
    parser.add_argument("--model-path", type=str, default="lusxvr/nanoVLM",
                        help="Path to model or HuggingFace model ID")
    parser.add_argument("--image", type=str, default="test/test_image.webp",
                        help="Path to test image")
    parser.add_argument("--question", type=str, default="What is the name of the dog's toy?",
                        help="Question to ask about the image")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--gen-tokens", type=int, default=3,
                        help="Number of tokens to generate for FLOP measurement")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup steps before measuring")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    # Create counter
    counter = FLOPCounter(args.model_path, args.device)
    
    # 1. Manual ViT FLOPs
    print("\n" + "="*60)
    print("CALCULATING FLOPS")
    print("="*60)
    
    manual_results = counter.calculate_vit_flops_manual(image_path=args.image)
    print_results("1. Vision Encoder FLOPs (Manual Calculation)", manual_results)
    
    # 2. Automatic ViT FLOPs
    auto_results = counter.calculate_vit_flops_automatic(args.image)
    print_results("2. Vision Encoder FLOPs (Automatic Profiling)", auto_results)
    
    # 3. Generation FLOPs without warmup
    gen_no_warmup = counter.calculate_generation_flops(
        args.image, 
        args.question, 
        n_tokens=args.gen_tokens,
        warmup_steps=0
    )
    print_results(f"3. Generation FLOPs ({args.gen_tokens} tokens, NO warmup)", gen_no_warmup)
    
    # 4. Generation FLOPs with warmup
    gen_with_warmup = counter.calculate_generation_flops(
        args.image, 
        args.question, 
        n_tokens=args.gen_tokens,
        warmup_steps=args.warmup
    )
    print_results(f"4. Generation FLOPs ({args.gen_tokens} tokens, WITH {args.warmup} warmup steps)", gen_with_warmup)
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"Manual ViT TFLOPs:    {manual_results['vit_total_tflops']:.4f}")
    print(f"Automatic ViT TFLOPs: {auto_results['vit_total_tflops']:.4f}")
    print(f"Difference:           {abs(manual_results['vit_total_tflops'] - auto_results['vit_total_tflops']):.4f}")
    print()
    print(f"Generation (no warmup):   {gen_no_warmup['generation_total_tflops']:.4f} TFLOPs")
    print(f"Generation (with warmup): {gen_with_warmup['generation_total_tflops']:.4f} TFLOPs")
    print(f"Warmup overhead:          {gen_no_warmup['latency_seconds'] - gen_with_warmup['latency_seconds']:.3f}s")
    print("="*60)
    
    # Save results to JSON if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "model_path": args.model_path,
            "image": args.image,
            "question": args.question,
            "device": str(counter.device),
            "manual_vit_flops": manual_results,
            "automatic_vit_flops": auto_results,
            "generation_no_warmup": gen_no_warmup,
            "generation_with_warmup": gen_with_warmup,
            "summary": {
                "manual_vit_tflops": manual_results['vit_total_tflops'],
                "automatic_vit_tflops": auto_results['vit_total_tflops'],
                "vit_difference_tflops": abs(manual_results['vit_total_tflops'] - auto_results['vit_total_tflops']),
                "generation_no_warmup_tflops": gen_no_warmup['generation_total_tflops'],
                "generation_with_warmup_tflops": gen_with_warmup['generation_total_tflops'],
                "warmup_overhead_seconds": gen_no_warmup['latency_seconds'] - gen_with_warmup['latency_seconds']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
