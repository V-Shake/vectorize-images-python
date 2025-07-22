# Image Vectorization Project

## Setup Instructions

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test the Installation

```bash
# Run the examples
python examples.py

# Or test with your own image
python main.py --input "path/to/your/image.jpg" --output "vectorized.png" --colors 6
```

## Quick Start

### Using as a Library

```python
from image_vectorizer import ImageVectorizer

vectorizer = ImageVectorizer()
result = vectorizer.vectorize_image(
    input_path="input.jpg",
    output_path="output.png",
    n_colors=8,
    smoothing=2
)
```

### Using Command Line

```bash
# Single image
python main.py --input photo.jpg --output vectorized.png --colors 6

# Batch processing
python main.py --input_dir input_folder/ --output_dir output_folder/ --colors 8
```

## Troubleshooting

### Import Errors
If you see import errors for cv2, numpy, etc., make sure you've installed the requirements:
```bash
pip install -r requirements.txt
```

### File Not Found Errors
- Make sure your input image paths are correct
- Use absolute paths if relative paths don't work
- Check that the image format is supported (jpg, png, bmp, tiff)

### Memory Issues
- Try reducing the number of colors (n_colors parameter)
- Use smaller images for testing
- Increase smoothing to reduce detail
