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

## Model Evaluation

Run model evaluation to measure accuracy and performance:

```bash
# Default: 100 samples, both modes, 512 resolution
./scripts/2-evaluate.sh

# With specific sample count
./scripts/2-evaluate.sh --max-samples 50

# MCQ only evaluation
./scripts/2-evaluate.sh --max-samples 100 --mode mcq

# Open-answer only evaluation
./scripts/2-evaluate.sh --max-samples 100 --mode oa

# With resolution reduction (384x384)
./scripts/2-evaluate.sh --resolution 384

# Combined: 100 samples, 256 resolution
./scripts/2-evaluate.sh --max-samples 100 --resolution 256

# With custom output file
./scripts/2-evaluate.sh --max-samples 10 --output results/custom.json

# With test split
./scripts/2-evaluate.sh --split test

# Or use Python directly
uv run python src/evaluation/evaluate.py \
    --max-samples 100 \
    --mode both \
    --resolution 384 \
    --output results/results_res384.json
```

**Output**: Results saved to `results/results_res{RESOLUTION}.json` (includes checkpoint for FLOP measurement)

**Results** (100 samples, 512×512 resolution):
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

# Using checkpoint from evaluation
uv run python src/evaluation/flop_counter.py \
    --checkpoint checkpoints/res384_for_flops.pt \
    --image test/test_image.webp \
    --question "What is the name of the dog's toy?" \
    --output results/flops_res384.json
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
│   ├── common.sh                # Shared shell functions
│   ├── 1-setup.sh               # Clone nanoVLM repository
│   ├── 2-evaluate.sh            # Run model evaluation
│   ├── 3-flop-counter.sh        # Measure FLOPs
│   └── 4-finetune-mcq.sh        # Fine-tune on MCQ task
├── src/
│   ├── evaluation/
│   │   ├── base_evaluator.py   # Base evaluator with FLOP profiling
│   │   ├── evaluate.py          # Model evaluation script
│   │   └── flop_counter.py      # FLOP measurement script
│   └── optimization/
│       └── resolution_reduction.py  # Resolution reduction strategy
├── models/
│   └── nanoVLM/                 # Cloned nanoVLM repository
├── test/
│   ├── test_image.webp          # Test image
│   └── test_query.txt           # Test query
├── checkpoints/                 # Model checkpoints for FLOP calculation
├── results/                     # Evaluation results
└── pyproject.toml               # Dependencies (uv)
```

## Resources

- **Model**: https://huggingface.co/lusxvr/nanoVLM
- **Repository**: https://github.com/huggingface/nanoVLM
- **Dataset**: https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA

## Optimization Strategies

### 1. Resolution Reduction ✅
- **Status**: Implemented
- **Method**: Reduce input image resolution (512→384→256→192)
- **FLOP Reduction**: Quadratic with resolution (e.g., 384→512 = 44%)
- **Implementation**: Automatic positional embedding interpolation
- **Use**: `./scripts/2-evaluate.sh --resolution 384`

### 2. Token Pooling/Dropping ✅
- **Status**: Implemented
- **Method**: Reduce number of image tokens processed by language model
- **Strategies**:
  - **Average Pooling**: Combine adjacent tokens (2x2 → 1, 4x reduction)
  - **Max Pooling**: Take maximum activation from patch
  - **Token Dropping**: Drop least informative tokens based on L2 norm
- **FLOP Reduction**: Linear with token reduction (50% tokens = ~50% LM FLOPs)
- **Use**: 
  ```bash
  # 2x2 pooling (4x token reduction)
  python src/optimization/token_pooling.py --pool-factor 2 --pool-method avg
  
  # Keep 50% of tokens (drop 50%)
  python src/optimization/token_pooling.py --keep-ratio 0.5 --drop-method norm
  ```

### 3. Other Strategies (Future)
- Quantization (INT8, INT4)
- Pruning (structured/unstructured)
- Knowledge distillation
- Layer reduction

## Next Steps

1. Run baseline evaluation at multiple resolutions
2. Measure FLOP reduction vs accuracy trade-off
3. Select optimal resolution meeting ≥65% accuracy threshold
4. Implement additional optimization strategies if needed
