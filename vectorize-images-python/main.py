"""
Command Line Interface for Image Vectorizer

This script provides a command-line interface for vectorizing images
using the ImageVectorizer class.
"""

import argparse
import sys
import os
from image_vectorizer import ImageVectorizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Vectorize images using k-means clustering and edge detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Vectorize a single image
  python main.py --input photo.jpg --output vectorized.png --colors 6

  # Batch process all images in a folder
  python main.py --input_dir input_folder/ --output_dir output_folder/ --colors 8

  # Advanced options
  python main.py --input photo.jpg --output result.png --colors 5 --smoothing 1.5 --no-edges
        """
    )
    
    # Input/Output arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', 
                      help='Path to input image file')
    group.add_argument('--input_dir', '-id',
                      help='Directory containing input images for batch processing')
    
    parser.add_argument('--output', '-o',
                       help='Path to output image file (required for single image)')
    parser.add_argument('--output_dir', '-od',
                       help='Output directory for batch processing (required for batch mode)')
    
    # Processing parameters
    parser.add_argument('--colors', '-c', type=int, default=8,
                       help='Number of color clusters (2-50, default: 8)')
    parser.add_argument('--smoothing', '-s', type=float, default=2.0,
                       help='Gaussian smoothing factor (0-10, default: 2.0)')
    parser.add_argument('--no-edges', action='store_true',
                       help='Disable edge detection (edges are enabled by default)')
    parser.add_argument('--black-white', action='store_true',
                       help='Convert to black & white manga style')
    parser.add_argument('--edge-threshold1', type=int, default=180,
                       help='Lower Canny threshold (default: 180)')
    parser.add_argument('--edge-threshold2', type=int, default=280,
                       help='Upper Canny threshold (default: 280)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    
    if not (2 <= args.colors <= 50):
        parser.error("--colors must be between 2 and 50")
    
    if not (0 <= args.smoothing <= 10):
        parser.error("--smoothing must be between 0 and 10")
    
    # Create vectorizer instance
    vectorizer = ImageVectorizer()
    
    try:
        if args.input:
            # Single image processing
            logger.info(f"Processing single image: {args.input}")
            
            result = vectorizer.vectorize_image(
                input_path=args.input,
                output_path=args.output,
                n_colors=args.colors,
                smoothing=args.smoothing,
                add_edges=not args.no_edges,
                edge_threshold1=args.edge_threshold1,
                edge_threshold2=args.edge_threshold2,
                black_white=args.black_white
            )
            
            print(f"✅ Successfully processed: {args.input} -> {args.output}")
            
        else:
            # Batch processing
            logger.info(f"Batch processing directory: {args.input_dir}")
            
            processed_count = vectorizer.batch_process(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                n_colors=args.colors,
                smoothing=args.smoothing,
                add_edges=not args.no_edges,
                edge_threshold1=args.edge_threshold1,
                edge_threshold2=args.edge_threshold2,
                black_white=args.black_white
            )
            
            print(f"✅ Successfully processed {processed_count} images in {args.output_dir}")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Parameter error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception("Detailed error information:")
        sys.exit(1)


if __name__ == "__main__":
    main()
