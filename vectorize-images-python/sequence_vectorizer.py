"""
🎬 Image Sequence Vectorizer

This module processes entire image sequences (like animations) with consistent colors.
Unlike processing images individually, this analyzes ALL images together first to create
a unified color palette, ensuring visual consistency across the entire sequence.

Perfect for:
- Animation frames
- Video sequences  
- Image series that should have consistent styling

Usage:
    python sequence_vectorizer.py
"""

import os
import cv2
import numpy as np
from matplotlib.image import imread, imsave
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import gaussian_filter
from image_vectorizer import _convert_to_grayscale_manga, DEFAULT_COLORS, DEFAULT_SMOOTHING, DEFAULT_ADD_EDGES, DEFAULT_EDGE_THRESHOLD1, DEFAULT_EDGE_THRESHOLD2, DEFAULT_BLACK_WHITE
import logging
from typing import Optional, List
import glob
from pathlib import Path

# ================== SEQUENCE CONFIGURATION ==================
# Input/Output paths
INPUT_SEQUENCE_FOLDER = "examples_image_sequence/room-animation"
OUTPUT_SEQUENCE_FOLDER = "examples_image_sequence/stylized_output"

# Processing settings (inherits from image_vectorizer.py defaults)
SEQUENCE_COLORS = 6                     # Colors for entire sequence (try 4-12)
SEQUENCE_SMOOTHING = DEFAULT_SMOOTHING  # Smoothing factor
SEQUENCE_ADD_EDGES = DEFAULT_ADD_EDGES  # Add black edges
SEQUENCE_EDGE_THRESHOLD1 = 30           # Thick manga-style edges
SEQUENCE_EDGE_THRESHOLD2 = 100          # Thick manga-style edges  
SEQUENCE_BLACK_WHITE = False            # ⚫ SET TO True FOR B&W MODE!

# Performance settings
SAMPLING_RATE = 50  # Take every Nth pixel for analysis (lower = faster, higher = more accurate)
# ============================================================

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequenceVectorizer:
    """
    Vectorize image sequences with consistent color palettes.
    
    This class analyzes all images in a sequence together to create a unified
    color palette, ensuring visual consistency across all frames.
    """
    
    def __init__(self):
        """Initialize the SequenceVectorizer."""
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.global_kmeans = None
        
    def collect_color_samples(self, 
                            input_folder: str, 
                            sampling_rate: int = SAMPLING_RATE,
                            smoothing: float = SEQUENCE_SMOOTHING) -> np.ndarray:
        """
        Collect color samples from all images to create a unified palette.
        
        Parameters:
        -----------
        input_folder : str
            Path to folder containing image sequence
        sampling_rate : int
            Take every Nth pixel (lower = faster processing)
        smoothing : float
            Gaussian smoothing factor
            
        Returns:
        --------
        np.ndarray
            Combined pixel samples from all images
        """
        all_pixels = []
        image_files = self._get_image_files(input_folder)
        
        logger.info(f"🚧 Collecting color samples from {len(image_files)} images...")
        
        for i, filepath in enumerate(image_files):
            try:
                # Read and preprocess image
                image = imread(filepath)
                
                # Handle different image formats
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image] * 3, axis=-1)
                elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA -> RGB
                    image = image[:, :, :3]
                    
                # Convert to uint8 type with range 0~255
                image = (image * 255).astype(np.uint8)
                
                # Apply smoothing
                if smoothing > 0:
                    image = gaussian_filter(image, (smoothing, smoothing, 0))
                
                # Sample pixels for faster processing
                pixels = image.reshape(-1, 3)
                sampled_pixels = pixels[::sampling_rate]
                all_pixels.append(sampled_pixels)
                
                if (i + 1) % 5 == 0:  # Progress update every 5 images
                    logger.info(f"   📸 Processed {i + 1}/{len(image_files)} images for sampling...")
                    
            except Exception as e:
                logger.warning(f"   ⚠️  Skipped {filepath}: {e}")
                continue
                
        if not all_pixels:
            raise ValueError("No valid images found for color sampling")
            
        combined_pixels = np.concatenate(all_pixels, axis=0)
        logger.info(f"✅ Collected {len(combined_pixels):,} color samples from sequence")
        
        return combined_pixels
    
    def create_global_palette(self, 
                            color_samples: np.ndarray, 
                            n_colors: int = SEQUENCE_COLORS) -> MiniBatchKMeans:
        """
        Create a global color palette from collected samples.
        
        Parameters:
        -----------
        color_samples : np.ndarray
            Pixel samples from all images
        n_colors : int
            Number of colors in the palette
            
        Returns:
        --------
        MiniBatchKMeans
            Trained color clustering model
        """
        logger.info(f"🎨 Creating global color palette with {n_colors} colors...")
        
        # Create and train k-means model
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=253, n_init=10)
        kmeans.fit(color_samples)
        
        logger.info("✅ Global color palette created successfully")
        return kmeans
    
    def process_sequence(self,
                        input_folder: str,
                        output_folder: str,
                        n_colors: int = SEQUENCE_COLORS,
                        smoothing: float = SEQUENCE_SMOOTHING,
                        add_edges: bool = SEQUENCE_ADD_EDGES,
                        edge_threshold1: int = SEQUENCE_EDGE_THRESHOLD1,
                        edge_threshold2: int = SEQUENCE_EDGE_THRESHOLD2,
                        black_white: bool = SEQUENCE_BLACK_WHITE,
                        sampling_rate: int = SAMPLING_RATE) -> int:
        """
        Process entire image sequence with consistent color palette.
        
        Parameters:
        -----------
        input_folder : str
            Path to input image sequence folder
        output_folder : str
            Path to output folder
        n_colors : int
            Number of colors in the unified palette
        smoothing : float
            Gaussian smoothing factor
        add_edges : bool
            Whether to add black edge outlines
        edge_threshold1 : int
            Lower Canny edge threshold
        edge_threshold2 : int
            Upper Canny edge threshold
        black_white : bool
            Convert to black & white manga style
        sampling_rate : int
            Pixel sampling rate for color analysis
            
        Returns:
        --------
        int
            Number of processed images
        """
        # Validate input
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Step 1: Collect color samples from all images
        color_samples = self.collect_color_samples(input_folder, sampling_rate, smoothing)
        
        # Step 2: Create global color palette
        self.global_kmeans = self.create_global_palette(color_samples, n_colors)
        
        # Step 3: Process each image with the unified palette
        image_files = self._get_image_files(input_folder)
        processed_count = 0
        
        logger.info(f"🎬 Processing {len(image_files)} images with unified palette...")
        
        for i, filepath in enumerate(image_files):
            try:
                # Generate output filename
                filename = os.path.basename(filepath)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"stylized_{filename}")
                
                # Process single image
                self._process_single_image(
                    filepath, output_path, smoothing, add_edges, 
                    edge_threshold1, edge_threshold2, black_white
                )
                
                processed_count += 1
                
                if (i + 1) % 5 == 0:  # Progress update
                    logger.info(f"   🎨 Processed {i + 1}/{len(image_files)} images...")
                    
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to process {filepath}: {e}")
                continue
                
        logger.info(f"✅ Sequence processing complete! {processed_count} images saved to: {output_folder}")
        return processed_count
    
    def _process_single_image(self, 
                            input_path: str, 
                            output_path: str,
                            smoothing: float,
                            add_edges: bool,
                            edge_threshold1: int,
                            edge_threshold2: int,
                            black_white: bool):
        """Process a single image with the global color palette."""
        # Read and preprocess image
        image = imread(input_path)
        
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA -> RGB
            image = image[:, :, :3]
            
        image = (image * 255).astype(np.uint8)
        
        # Apply smoothing
        if smoothing > 0:
            image = gaussian_filter(image, (smoothing, smoothing, 0))
        
        # Apply global color palette
        pixels = image.reshape(-1, 3)
        labels = self.global_kmeans.predict(pixels)
        clustered_img = self.global_kmeans.cluster_centers_.astype('uint8')[labels].reshape(image.shape)
        
        # Add edges if requested
        if add_edges:
            gray = cv2.cvtColor(clustered_img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, threshold1=edge_threshold1, threshold2=edge_threshold2)
            edge_mask = edges > 0
            clustered_img[edge_mask] = [0, 0, 0]
        
        # Convert to black & white if requested
        if black_white:
            n_colors = len(self.global_kmeans.cluster_centers_)
            clustered_img = _convert_to_grayscale_manga(clustered_img, n_colors)
        
        # Save result
        imsave(output_path, clustered_img)
    
    def _get_image_files(self, folder: str) -> List[str]:
        """Get sorted list of image files from folder."""
        image_files = []
        
        for ext in self.supported_formats:
            pattern = os.path.join(folder, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(folder, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
        
        return sorted(image_files)


def main():
    """Main function to process the example image sequence."""
    print("🎬 Image Sequence Vectorizer")
    print("=" * 50)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up paths
    input_path = os.path.join(script_dir, INPUT_SEQUENCE_FOLDER)
    output_path = os.path.join(script_dir, OUTPUT_SEQUENCE_FOLDER)
    
    print(f"📁 Input folder: {INPUT_SEQUENCE_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_SEQUENCE_FOLDER}")
    print(f"🎨 Colors: {SEQUENCE_COLORS}")
    print(f"📏 Smoothing: {SEQUENCE_SMOOTHING}")
    print(f"🖼️  Edges: {SEQUENCE_ADD_EDGES}")
    print(f"⚫ Black & White: {SEQUENCE_BLACK_WHITE}")
    
    if not os.path.exists(input_path):
        print(f"❌ Input folder not found: {input_path}")
        print("💡 Make sure you have images in the examples_image_sequence/room-animation/ folder")
        return
    
    try:
        # Create and run sequence vectorizer
        vectorizer = SequenceVectorizer()
        processed_count = vectorizer.process_sequence(
            input_folder=input_path,
            output_folder=output_path,
            n_colors=SEQUENCE_COLORS,
            smoothing=SEQUENCE_SMOOTHING,
            add_edges=SEQUENCE_ADD_EDGES,
            edge_threshold1=SEQUENCE_EDGE_THRESHOLD1,
            edge_threshold2=SEQUENCE_EDGE_THRESHOLD2,
            black_white=SEQUENCE_BLACK_WHITE,
            sampling_rate=SAMPLING_RATE
        )
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Processed {processed_count} images with consistent styling")
        print(f"📂 Results saved in: {OUTPUT_SEQUENCE_FOLDER}")
        print(f"\n💡 All images now have the same color palette for visual consistency!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Detailed error information:")


if __name__ == "__main__":
    main()
