# nanoVLM-Lite

**Reducing inference FLOPs for a lightweight, efficient vision-language model**

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
│   ├── 4-finetune-mcq.sh        # Fine-tune on MCQ task
│   └── 5-token-pooling.sh       # Evaluate token pooling/dropping
├── src/
│   ├── evaluation/
│   │   ├── base_evaluator.py           # Base evaluator with FLOP profiling
│   │   ├── evaluate.py                 # Model evaluation script
│   │   ├── evaluate_token_pooling.py   # Token pooling evaluation
│   │   └── flop_counter.py             # FLOP measurement script
│   └── optimization/
│       ├── token_pooling.py            # Token pooling/dropping modules
│       ├── vlm_with_pooling.py         # VLM with token pooling integration
│       └── test_token_pooling.py       # Token pooling test suite
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

### 2. Token Pooling ✅
- **Status**: Implemented
- **Method**: Apply token pooling to pretrained model WITHOUT retraining
- **Approach**: Replace modality projector with pooling-enabled version
- **Strategies**:
  - **Average Pooling**: Spatial mean aggregation (kernel/stride configurable)
  - **Max Pooling**: Maximum activation per patch (kernel/stride configurable)
  - **Adaptive Pooling**: Target grid size (e.g., 8×8 = 64 tokens)
- **FLOP Reduction**: Linear with token reduction (64→16 tokens ≈ 75% LM FLOPs)
- **Implementation**: `TokenPooledEvaluator(BaseEvaluator)` with post-hoc pooling
- **Evaluation**:
  ```bash
  # Baseline (no pooling)
  ./scripts/5-token-pooling.sh --pooling none
  
  # Average pooling with 2×2 kernel, stride 2
  ./scripts/5-token-pooling.sh --pooling avg --kernel 2 --stride 2
  
  # Max pooling with 4×4 kernel, stride 4  
  ./scripts/5-token-pooling.sh --pooling max --kernel 4 --stride 4
  
  # Adaptive pooling to 8×8 grid (64 tokens)
  ./scripts/5-token-pooling.sh --pooling adaptive --grid 8
  
  # With more samples
  ./scripts/5-token-pooling.sh --max-samples 500 --pooling avg
  
  # Or use Python directly
  uv run python src/evaluation/evaluate_token_pooling.py \
      --pooling avg --kernel 2 --stride 2 --max-samples 100
  ```

### 3. Other Strategies (Future)
- Quantization (INT8, INT4)
- Knowledge distillation

## Next Steps

1. Run baseline evaluation at multiple resolutions
2. Measure FLOP reduction vs accuracy trade-off
3. Select optimal resolution meeting ≥65% accuracy threshold
4. Implement additional optimization strategies if needed
