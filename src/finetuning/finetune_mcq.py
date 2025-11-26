#!/usr/bin/env python3
"""
Fine-tune nanoVLM on A-OKVQA dataset for multiple-choice question answering.

This script trains the model to predict the correct answer letter (A, B, C, or D)
given an image and a multiple-choice question with four options.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Add nanoVLM to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "models" / "nanoVLM"))

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

# Add local imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import build_mc_prompt, LETTER_TO_IDX, IDX_TO_LETTER


def seed_everything(seed=1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_auto():
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AOKVQAMCQDataset(Dataset):
    """Dataset wrapper for A-OKVQA multiple-choice questions."""
    
    def __init__(self, split="train"):
        """
        Initialize the dataset.
        
        Args:
            split: Dataset split to load ('train' or 'validation')
        """
        self.ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing:
                - image: PIL Image
                - question: Question text
                - choices: List of 4 answer choices
                - gt_idx: Index of correct answer (0-3)
                - qid: Question ID
        """
        ex = self.ds[int(idx)]
        return {
            "image": ex["image"],
            "question": ex["question"],
            "choices": list(ex["choices"]),
            "gt_idx": int(ex["correct_choice_idx"]),
            "qid": str(ex["question_id"])
        }


def train(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    seed_everything(args.seed)
    device = device_auto()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading model: {args.model_id}")
    
    # Load pretrained model
    model = VisionLanguageModel.from_pretrained(args.model_id).to(device)
    model.train()

    # Freeze components if requested
    if args.freeze_vision:
        if hasattr(model, "vision_encoder"):
            for p in model.vision_encoder.parameters():
                p.requires_grad = False
            print("[freeze] vision_encoder frozen")
        else:
            print("[warning] model.vision_encoder not found, skipping freeze")
            
    if args.freeze_proj:
        if hasattr(model, "MP"):
            for p in model.MP.parameters():
                p.requires_grad = False
            print("[freeze] modality projector (MP) frozen")
        else:
            print("[warning] model.MP not found, skipping freeze")
    
    # Get tokenizer and image processor
    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer, 
        model.cfg.vlm_extra_tokens, 
        model.cfg.lm_chat_template
    )
    
    resize_to_max_side_len = getattr(model.cfg, "resize_to_max_side_len", False)
    imgproc = get_image_processor(
        model.cfg.max_img_size, 
        model.cfg.vit_img_size,
        resize_to_max_side_len
    )

    # Load datasets
    print("Loading datasets...")
    ds_train = AOKVQAMCQDataset(split="train")
    ds_val = AOKVQAMCQDataset(split="validation")
    print(f"Train size: {len(ds_train)} | Val size: {len(ds_val)}")
    
    # Setup optimizer with only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({100*num_trainable/num_total:.2f}%)")
    
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=args.lr, 
        betas=(0.9, 0.95), 
        weight_decay=args.weight_decay
    )
    
    # Setup learning rate scheduler
    total_steps = math.ceil(len(ds_train) * args.epochs / max(1, args.grad_accum))
    warmup_steps = int(args.warmup_ratio * total_steps)
    print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Training state
    global_step = 0
    best_val_acc = -1.0
    best_model_path = None

    def step_on_example(sample: Dict[str, Any]) -> torch.Tensor:
        """
        Perform a forward pass on a single example.
        
        Args:
            sample: Dictionary with image, question, choices, and ground truth
            
        Returns:
            Loss tensor
        """
        img: Image.Image = sample["image"].convert("RGB")
        q, choices = sample["question"], sample["choices"]
        gt_letter = IDX_TO_LETTER[sample["gt_idx"]]
        prompt = build_mc_prompt(q, choices)

        # Process image
        proc_img, _ = imgproc(img)
        n_imgs = proc_img.shape[0]
        num_img_tokens = model.cfg.mp_image_token_length * n_imgs
        image_tokens = tokenizer.image_token * num_img_tokens

        # Create messages for user prompt and full sequence with answer
        messages_user = [{"role": "user", "content": image_tokens + prompt}]
        messages_full = [
            {"role": "user", "content": image_tokens + prompt},
            {"role": "assistant", "content": gt_letter}
        ]

        # Tokenize both sequences
        ids_user = tokenizer.apply_chat_template(
            messages_user, tokenize=True, add_generation_prompt=False
        )
        ids_full = tokenizer.apply_chat_template(
            messages_full, tokenize=True, add_generation_prompt=False
        )

        # Find where the answer token actually appears in ids_full
        # Search from the end backwards to avoid matching answer letters in the question
        ans_ids = tokenizer(gt_letter, add_special_tokens=False).input_ids
        start = None
        for i in range(len(ids_full) - len(ans_ids), -1, -1):
            if ids_full[i:i+len(ans_ids)] == ans_ids:
                start = i
                break
        
        if start is None:
            raise ValueError(f"Could not find answer '{gt_letter}' in tokenized sequence")
        
        input_ids = torch.tensor(ids_full, dtype=torch.long, device=device).unsqueeze(0)
        labels = torch.full_like(input_ids, fill_value=-100)
        
        # CRITICAL: In language modeling, logits[i] predicts token at position i+1
        # So to predict token at position `start`, we need to set labels at position `start-1`
        for k in range(len(ans_ids)):
            labels[0, start - 1 + k] = ans_ids[k]

        # Forward pass
        # Note: forward() returns (hidden_states, loss) when targets=None
        # We need to apply the LM head to get logits
        hidden_states, _ = model(input_ids, proc_img.to(device), targets=None)
        logits = model.decoder.head(hidden_states)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss

    @torch.no_grad()
    def evaluate_mcq(split_ds, max_samples=None) -> float:
        """
        Evaluate the model on multiple-choice questions.
        
        Args:
            split_ds: Dataset to evaluate on
            max_samples: Maximum number of samples to evaluate (None for all)
            
        Returns:
            Accuracy percentage
        """
        model.eval()
        correct = 0
        total = min(len(split_ds), max_samples) if max_samples else len(split_ds)
        
        for i, ex in enumerate(tqdm(split_ds, total=total, desc="Eval-MCQ", leave=False)):
            if max_samples and i >= max_samples:
                break
            img: Image.Image = ex["image"].convert("RGB")
            q, choices = ex["question"], list(ex["choices"])
            gt_idx = int(ex["gt_idx"])

            # Process image
            proc_img, _ = imgproc(img)
            n_imgs = proc_img.shape[0]
            num_img_tokens = model.cfg.mp_image_token_length * n_imgs
            image_tokens = tokenizer.image_token * num_img_tokens

            # Create prompt
            prompt = build_mc_prompt(q, choices)
            messages = [{"role": "user", "content": image_tokens + prompt}]
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

            # Get logits for next token
            # Note: forward() returns (hidden_states, loss) when targets=None
            # We need to apply the LM head to get logits
            hidden_states, _ = model(input_ids, proc_img.to(device), targets=None)
            logits = model.decoder.head(hidden_states)
            next_logits = logits[0, -1, :]

            # Get probabilities for each answer choice
            letter_ids = [
                tokenizer(x, add_special_tokens=False).input_ids[0] 
                for x in "ABCD"
            ]
            pred_idx = int(next_logits[letter_ids].softmax(dim=-1).argmax().item())
            correct += int(pred_idx == gt_idx)

        model.train()
        return 100.0 * correct / total
    
    # Skip baseline evaluation - user will evaluate separately
    print("\n" + "=" * 80)
    print("Skipping baseline evaluation - starting training immediately")
    print("=" * 80 + "\n")
    eval_samples = args.eval_samples if hasattr(args, 'eval_samples') and args.eval_samples > 0 else None
    baseline_acc = None

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    accum = args.grad_accum
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(total_steps), desc="Training", dynamic_ncols=True)
    
    # Create shuffled indices for training data
    train_indices = list(range(len(ds_train)))
    random.shuffle(train_indices)
    train_iter_idx = 0

    for step in pbar:
        # Accumulate gradients over multiple examples
        loss_accum = 0.0
        
        for _ in range(accum):
            if train_iter_idx >= len(train_indices):
                # Reshuffle when we've gone through all data
                random.shuffle(train_indices)
                train_iter_idx = 0
            sample = ds_train[train_indices[train_iter_idx]]
            train_iter_idx += 1

            loss = step_on_example(sample) / accum
            loss.backward()
            loss_accum += loss.item()

        # Clip gradients
        clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 
            args.grad_clip
        )
        
        # Update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        
        pbar.set_postfix(
            loss=f"{loss_accum:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

        # Periodic evaluation
        if (global_step % args.eval_every) == 0:
            val_acc = evaluate_mcq(ds_val, max_samples=eval_samples)
            print(f"\n[Eval] step={global_step} | val MCQ acc = {val_acc:.2f}%")
            if eval_samples:
                print(f"  (evaluated on {eval_samples} samples)")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = out_dir / f"best_mcq_{val_acc:.2f}.pt"
                print(f"[Checkpoint] Saving to {ckpt_path}")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "step": global_step,
                    "args": vars(args)
                }, ckpt_path)
                best_model_path = ckpt_path

    # Final evaluation
    final_acc = evaluate_mcq(ds_val, max_samples=eval_samples)
    print(f"\n[Final] val MCQ acc = {final_acc:.2f}%  (best={best_val_acc:.2f}%)")
    if eval_samples:
        print(f"  (evaluated on {eval_samples} samples)")

    # Save final model in HuggingFace format
    full_out = out_dir / "mcq_finetuned"
    model.save_pretrained(str(full_out))
    print(f"Saved HF checkpoint to: {full_out}")
    
    # Save training summary
    summary = {
        "baseline_acc": baseline_acc,
        "final_val_acc": final_acc,
        "best_val_acc": best_val_acc,
        "improvement": final_acc - baseline_acc,
        "total_steps": global_step,
        "args": vars(args)
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Baseline accuracy:     {baseline_acc:.2f}%")
    print(f"Final accuracy:        {final_acc:.2f}%")
    print(f"Best accuracy:         {best_val_acc:.2f}%")
    print(f"Improvement:           {final_acc - baseline_acc:+.2f}%")
    print("=" * 80)
    
    print("\nTraining complete!")
    if best_model_path:
        print(f"Best model saved to: {best_model_path}")


def main():
    """Parse arguments and start training."""
    ap = argparse.ArgumentParser(description="Fine-tune nanoVLM on A-OKVQA")
    
    # Model arguments
    ap.add_argument(
        "--model_id", 
        default="lusxvr/nanoVLM", 
        help="Pretrained HF model id or local folder"
    )
    ap.add_argument(
        "--output_dir", 
        default="checkpoints/mcq_finetuning",
        help="Directory to save checkpoints"
    )
    
    # Training arguments
    ap.add_argument(
        "--epochs", 
        type=int, 
        default=2,
        help="Number of training epochs"
    )
    ap.add_argument(
        "--lr", 
        type=float, 
        default=2e-4,
        help="Learning rate"
    )
    ap.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Weight decay"
    )
    ap.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=0.10,
        help="Warmup ratio of total steps"
    )
    ap.add_argument(
        "--grad_accum", 
        type=int, 
        default=32,
        help="Gradient accumulation steps (effective batch size = grad_accum)"
    )
    ap.add_argument(
        "--grad_clip", 
        type=float, 
        default=1.0,
        help="Gradient clipping value"
    )
    ap.add_argument(
        "--label_smoothing", 
        type=float, 
        default=0.0,
        help="Label smoothing factor"
    )
    
    # Evaluation arguments
    ap.add_argument(
        "--eval_every",
        type=int, 
        default=500,
        help="Evaluate every N steps"
    )
    ap.add_argument(
        "--eval_samples",
        type=int,
        default=-1,
        help="Number of validation samples to evaluate (-1 for all, useful for quick testing)"
    )
    
    # Freezing arguments
    ap.add_argument(
        "--freeze_vision", 
        action="store_true",
        default=True,
        help="Freeze vision encoder weights (default: True)"
    )
    ap.add_argument(
        "--freeze_proj", 
        action="store_true",
        default=False,
        help="Freeze modality projector weights (default: False)"
    )
    
    # Other arguments
    ap.add_argument(
        "--seed", 
        type=int, 
        default=1234,
        help="Random seed"
    )
    
    args = ap.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("Fine-tuning Configuration")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k:20s}: {v}")
    print("=" * 80)
    
    train(args)


if __name__ == "__main__":
    main()
