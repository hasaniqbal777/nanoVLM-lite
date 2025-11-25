# Test Data

## Test Image
- **File**: `test_image.webp`
- **Source**: https://petscare-assets-prod.s3.amazonaws.com/media/original_images/chihuahua-puppy-grass-squeaky-bone-toy-58273.webp
- **Description**: Chihuahua puppy with a squeaky bone toy on grass

## Test Query
- **File**: `test_query.txt`
- **Question**: "What is the name of the dog's toy?"
- **Expected Answer**: bone (or squeaky bone, bone toy, etc.)

## Usage

Test the model with this example:
```bash
uv run python -c "
from PIL import Image
from pathlib import Path
import sys
sys.path.insert(0, 'models/nanoVLM')

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

# Load model
model = VisionLanguageModel.from_pretrained('lusxvr/nanoVLM')
model.eval()

# Load image
image = Image.open('test/test_image.webp').convert('RGB')

# Process and generate
# ... (see baseline.py for full example)
"
```
