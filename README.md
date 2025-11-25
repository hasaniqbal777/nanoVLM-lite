# nanoVLM Optimization for A-OKVQA

**Goal**: Reduce inference TFLOPs as much as possible while maintaining ≥65% MCQ accuracy on A-OKVQA.

## Setup

### 1. Clone nanoVLM Repository
```bash
./scripts/1-setup.sh
```

This clones the nanoVLM model code from https://github.com/huggingface/nanoVLM to `models/nanoVLM/`.

### 2. Install Dependencies
```bash
uv sync
```

## Baseline Evaluation

Run baseline evaluation to measure accuracy and performance:

```bash
# Quick test on 10 samples
./scripts/2-baseline.sh 10

# Full evaluation on 100 samples (both MCQ and OA)
./scripts/2-baseline.sh 100 both

# MCQ only evaluation
./scripts/2-baseline.sh 100 mcq

# Open-answer only evaluation
./scripts/2-baseline.sh 100 oa

# With custom output file
./scripts/2-baseline.sh 1 both --output results/baseline_1.json

# With test split
./scripts/2-baseline.sh 100 both --split test

# Full validation set (no max samples specified)
./scripts/2-baseline.sh --output results/baseline_full.json

# Or explicitly use 0 for all samples
./scripts/2-baseline.sh 0 both --output results/baseline_full.json

# Or use Python directly
uv run python src/evaluation/baseline.py --max-samples 100 --mode both
```

**Results** (100 samples):
- Model: `lusxvr/nanoVLM` (460M parameters)
- Accuracy: TBD%
- Latency: ~950ms/sample
- Target (65%): TBD%

## FLOP Measurement

Measure FLOPs for vision encoder and generation:

```bash
# Run FLOP counter with test image
./scripts/3-flop-counter.sh

# With custom image and query
./scripts/3-flop-counter.sh test/test_image.webp test/test_query.txt

# With custom output path
./scripts/3-flop-counter.sh test/test_image.webp test/test_query.txt results/my_flops.json
```

**FLOP Counter Deliverables**:
- Manual ViT FLOPs calculation (hand-computed)
- Automatic ViT FLOPs profiling (PyTorch profiler)
- Generation FLOPs for 3 tokens (with and without warmup)
- Results saved to `results/baseline_flops.json`

## Project Structure

```
.
├── scripts/
│   ├── common.sh           # Shared shell functions
│   ├── 1-setup.sh          # Clone nanoVLM repository
│   ├── 2-baseline.sh       # Run baseline evaluation
│   └── 3-flop-counter.sh   # Measure FLOPs
├── src/
│   └── evaluation/
│       ├── baseline.py     # Baseline evaluation script
│       └── flop_counter.py # FLOP measurement script
├── models/
│   └── nanoVLM/            # Cloned nanoVLM repository
├── test/
│   ├── test_image.webp     # Test image
│   └── test_query.txt      # Test query
├── results/                # Evaluation results
└── pyproject.toml          # Dependencies (uv)
```

## Resources

- **Model**: https://huggingface.co/lusxvr/nanoVLM
- **Repository**: https://github.com/huggingface/nanoVLM
- **Dataset**: https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA

## Next Steps

After baseline evaluation:
1. Implement optimization strategies (quantization, pruning, etc.)
2. Evaluate optimized models
3. Select best model meeting 65% accuracy threshold
