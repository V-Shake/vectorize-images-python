"""
🎨 Style Gallery Generator

This script creates multiple artistic styles from the example room image.
Perfect for discovering which style you like best!

What it creates:
- Basic vectorization
- 🎨 Minimalist (3 colors)
- 🎌 Thick Manga Style
- ✏️ Thin Clean Lines  
- 🔍 Black & White Manga (12 grey levels, thick lines)
- ⚫ Simple B&W (2 grey levels, thick lines)
- 🌊 No Edges (Smooth)

Just run this file to generate a complete style gallery!
"""

import os
import sys
from image_vectorizer import ImageVectorizer
import logging
import numpy as np
from matplotlib.image import imsave

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_grayscale_manga(image_array, n_colors=4):
    """Convert a color image to black, white, and grey tones only.
    
    Args:
        image_array: The input image array
        n_colors: Number of grey levels to use (2-20)
    """
    # Convert to grayscale
    if len(image_array.shape) == 3:
        # Convert RGB to grayscale using proper weights
        gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = image_array
    
    # Ensure n_colors is reasonable
    n_colors = max(2, min(20, n_colors))
    
    # Create grey levels based on n_colors
    # Generate evenly spaced grey levels from black (0) to white (255)
    levels = np.linspace(0, 255, n_colors).astype(np.uint8)
    
    # Create thresholds between the levels
    if n_colors > 1:
        thresholds = []
        for i in range(n_colors - 1):
            threshold = (float(levels[i]) + float(levels[i + 1])) / 2.0
            thresholds.append(threshold)
        thresholds = np.array(thresholds)
    else:
        thresholds = np.array([127.5])  # Single threshold for 2 colors
    
    # Quantize the grayscale image to the specified number of levels
    indices = np.digitize(gray, thresholds)
    quantized = levels[indices]
    
    # Convert back to RGB (same value for all channels)
    if len(image_array.shape) == 3:
        result = np.stack([quantized] * 3, axis=-1).astype(np.uint8)
    else:
        result = quantized.astype(np.uint8)
    
    return result


def example_basic_usage():
    """Basic example of image vectorization."""
    print("🎨 Basic Image Vectorization Example")
    print("=" * 50)
    
    vectorizer = ImageVectorizer()
    
    # Use the example image that's already there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "examples", "room-original.png")
    output_path = os.path.join(script_dir, "examples", "room_basic.png")
    
    if os.path.exists(input_path):
        try:
            result = vectorizer.vectorize_image(
                input_path=input_path,
                output_path=output_path,
                n_colors=8,
                smoothing=2.0,
                add_edges=True
            )
            print(f"✅ Basic vectorization completed: {output_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"⚠️  Input image not found: {input_path}")


def example_different_styles():
    """Demonstrate different vectorization styles."""
    print("\n🎨 Different Vectorization Styles")
    print("=" * 50)
    
    vectorizer = ImageVectorizer()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "examples", "room-original.png")
    
    if not os.path.exists(input_path):
        print(f"⚠️  Input image not found: {input_path}")
        return
    
    # Create examples directory
    output_dir = os.path.join(script_dir, "examples", "style_gallery")
    os.makedirs(output_dir, exist_ok=True)
    
    styles = [
        {
            "name": "🎨 Minimalist (3 colors)",
            "params": {"n_colors": 3, "smoothing": 3.0, "add_edges": True, "edge_threshold1": 30, "edge_threshold2": 100},
            "output": os.path.join(output_dir, "minimalist.png")
        },
        {
            "name": "🎌 Thick Manga Style",
            "params": {"n_colors": 8, "smoothing": 2.0, "add_edges": True, "edge_threshold1": 30, "edge_threshold2": 100},
            "output": os.path.join(output_dir, "manga_thick.png")
        },
        {
            "name": "✏️ Thin Clean Lines",
            "params": {"n_colors": 12, "smoothing": 1.0, "add_edges": True, "edge_threshold1": 80, "edge_threshold2": 200},
            "output": os.path.join(output_dir, "thin_lines.png")
        },
        {
            "name": "🔍 Black & White Manga (12 greys)",
            "params": {"n_colors": 12, "smoothing": 2.0, "add_edges": True, "edge_threshold1": 30, "edge_threshold2": 100},
            "output": os.path.join(output_dir, "black_white_manga.png")
        },
        {
            "name": "⚫ Simple B&W (2 greys)",
            "params": {"n_colors": 2, "smoothing": 1.0, "add_edges": True, "edge_threshold1": 80, "edge_threshold2": 200},
            "output": os.path.join(output_dir, "black_white_simple.png")
        },
        {
            "name": "🌊 No Edges (Smooth)",
            "params": {"n_colors": 15, "smoothing": 2.0, "add_edges": False},
            "output": os.path.join(output_dir, "no_edges.png")
        }
    ]
    
    for style in styles:
        try:
            print(f"Creating {style['name']}...")
            
            if "Black & White" in style['name'] or "B&W" in style['name']:
                # Special processing for black & white manga styles
                result = vectorizer.vectorize_image(
                    input_path=input_path,
                    output_path=None,  # Don't save yet
                    **style['params']
                )
                # Convert to black/white/grey using the same number of colors
                bw_result = convert_to_grayscale_manga(result, style['params']['n_colors'])
                # Save the converted result
                imsave(style['output'], bw_result)
            else:
                # Normal processing for other styles
                vectorizer.vectorize_image(
                    input_path=input_path,
                    output_path=style['output'],
                    **style['params']
                )
            
            print(f"✅ {style['name']} completed: {style['output']}")
        except Exception as e:
            print(f"❌ Error creating {style['name']}: {e}")


def main():
    """Run all examples."""
    print("🚀 Image Vectorizer Examples")
    print("=" * 70)
    
    # Run the working examples
    example_basic_usage()
    example_different_styles()
    
    print("\n🎉 Style gallery completed!")
    print("\n📁 Generated files:")
    print("- examples/room_basic.png")
    print("- examples/style_gallery/minimalist.png")
    print("- examples/style_gallery/manga_thick.png")
    print("- examples/style_gallery/thin_lines.png")
    print("- examples/style_gallery/black_white_manga.png (12 grey levels)")
    print("- examples/style_gallery/black_white_simple.png (2 grey levels)")
    print("- examples/style_gallery/no_edges.png")
    print("\n💡 Check the 'examples/style_gallery/' folder to see all the different styles!")


if __name__ == "__main__":
    main()
