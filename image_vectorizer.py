"""
Image Vectorizer Module

This module provides functionality to convert raster images into stylized,
vector-like artwork using k-means clustering, Gaussian smoothing, and edge detection.
"""

import cv2
import numpy as np
from matplotlib.image import imread, imsave
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import gaussian_filter
import os
from typing import Optional, Tuple
import logging

# ================== EASY CONFIGURATION ==================
# Change these values to customize your output:
DEFAULT_COLORS = 8         # Increase this for more colors (2-50)
DEFAULT_SMOOTHING = 2.0    # Smoothing factor (0-10)
DEFAULT_ADD_EDGES = True   # True/False for black edges

# BLACK & WHITE MODE 🔲
DEFAULT_BLACK_WHITE = False  # True = Convert to black/white/grey, False = Keep colors
# When True, uses DEFAULT_COLORS to determine grey levels (2-20 works best)

# Edge detection settings (for manga/anime style outlines):
DEFAULT_EDGE_THRESHOLD1 = 100   # Lower = more edge details (10-200)
DEFAULT_EDGE_THRESHOLD2 = 250  # Upper = edge sensitivity (50-300)
# 🎨 STYLES TO TRY:
# Thick manga lines:  30, 100  
# Thin clean lines:   80, 200
# Many details:       20, 80
# Minimal outlines:   100, 250 ← CURRENT
# Black & White manga: Set DEFAULT_BLACK_WHITE = True + adjust DEFAULT_COLORS
# ========================================================

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _convert_to_grayscale_manga(image_array, n_colors=4):
    """Convert a color image to black, white, and grey tones only.
    
    Args:
        image_array: The input image array
        n_colors: Number of grey levels to use (2-20)
    
    Returns:
        np.ndarray: Black/white/grey converted image
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


class ImageVectorizer:
    """
    A class for vectorizing images using machine learning techniques.
    
    This class applies k-means clustering to reduce colors, Gaussian smoothing
    to reduce noise, and Canny edge detection to add clean outlines.
    """
    
    def __init__(self):
        """Initialize the ImageVectorizer."""
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def vectorize_image(self, 
                       input_path: str,
                       output_path: Optional[str] = None,
                       n_colors: int = DEFAULT_COLORS,
                       smoothing: float = DEFAULT_SMOOTHING,
                       add_edges: bool = DEFAULT_ADD_EDGES,
                       edge_threshold1: int = DEFAULT_EDGE_THRESHOLD1,
                       edge_threshold2: int = DEFAULT_EDGE_THRESHOLD2,
                       black_white: bool = DEFAULT_BLACK_WHITE) -> np.ndarray:
        """
        Vectorize/stylize an image using k-means clustering and edge detection.
        
        Parameters:
        -----------
        input_path : str
            Path to the input image
        output_path : str, optional
            Path to save the output image. If None, doesn't save.
        n_colors : int, default=8
            Number of color clusters (2-50)
        smoothing : float, default=2.0
            Gaussian smoothing factor (0-10)
        add_edges : bool, default=True
            Whether to add black edges using Canny detection
        edge_threshold1 : int, default=180
            Lower threshold for Canny edge detection
        edge_threshold2 : int, default=280
            Upper threshold for Canny edge detection
        black_white : bool, default=False
            Convert to black/white/grey tones (manga style)
        
        Returns:
        --------
        np.ndarray
            Processed image as numpy array
        
        Raises:
        -------
        FileNotFoundError
            If input image doesn't exist
        ValueError
            If parameters are out of valid range
        """
        
        # Validate parameters
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        if not (2 <= n_colors <= 50):
            raise ValueError("n_colors must be between 2 and 50")
        
        if not (0 <= smoothing <= 10):
            raise ValueError("smoothing must be between 0 and 10")
        
        logger.info(f"Processing image: {input_path}")
        logger.info(f"Parameters - Colors: {n_colors}, Smoothing: {smoothing}, Edges: {add_edges}")
        
        try:
            # Read image as float values (0~1)
            image = imread(input_path)
            
            # Handle different image formats
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:  # RGBA -> RGB
                image = image[:, :, :3]
            
            # Convert to uint8 type with range 0~255
            image = (image * 255).astype(np.uint8)
            
            # Apply Gaussian smoothing to reduce noise
            if smoothing > 0:
                smoothed_img = gaussian_filter(image, (smoothing, smoothing, 0))
            else:
                smoothed_img = image.copy()
            
            # Reshape image for k-means clustering
            pixels = smoothed_img.reshape(-1, 3)
            
            # Apply k-means clustering to reduce colors
            logger.info("Applying k-means clustering...")
            kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=253, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # Create clustered image with reduced colors
            clustered_img = kmeans.cluster_centers_.astype('uint8')[labels].reshape(image.shape)
            
            # Add black edges using Canny edge detection
            if add_edges:
                logger.info("Adding edge detection...")
                gray = cv2.cvtColor(clustered_img, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, threshold1=edge_threshold1, threshold2=edge_threshold2)
                edge_mask = edges > 0
                clustered_img[edge_mask] = [0, 0, 0]  # Set edges to black
            
            # Convert to black & white/grey if requested
            if black_white:
                logger.info(f"Converting to black & white manga style with {n_colors} grey levels...")
                clustered_img = _convert_to_grayscale_manga(clustered_img, n_colors)
            
            # Save the result if output path is provided
            if output_path:
                # Ensure output directory exists
                output_dir = os.path.dirname(output_path)
                if output_dir:  # Only create directory if path has a directory component
                    os.makedirs(output_dir, exist_ok=True)
                imsave(output_path, clustered_img)
                logger.info(f"Saved vectorized image to: {output_path}")
            
            logger.info("Image processing completed successfully!")
            return clustered_img
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    def batch_process(self,
                     input_dir: str,
                     output_dir: str,
                     n_colors: int = DEFAULT_COLORS,
                     smoothing: float = DEFAULT_SMOOTHING,
                     add_edges: bool = DEFAULT_ADD_EDGES,
                     edge_threshold1: int = DEFAULT_EDGE_THRESHOLD1,
                     edge_threshold2: int = DEFAULT_EDGE_THRESHOLD2,
                     black_white: bool = DEFAULT_BLACK_WHITE) -> int:
        """
        Process all images in a directory.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing input images
        output_dir : str
            Directory to save processed images
        **kwargs : 
            Parameters passed to vectorize_image method
        
        Returns:
        --------
        int
            Number of images processed successfully
        """
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        image_files = []
        
        # Find all image files
        for file in os.listdir(input_dir):
            if os.path.splitext(file.lower())[1] in self.supported_formats:
                image_files.append(file)
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        for filename in image_files:
            try:
                input_path = os.path.join(input_dir, filename)
                # Create output filename with '_vectorized' suffix
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_vectorized{ext}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Process the image
                self.vectorize_image(
                    input_path=input_path,
                    output_path=output_path,
                    n_colors=n_colors,
                    smoothing=smoothing,
                    add_edges=add_edges,
                    edge_threshold1=edge_threshold1,
                    edge_threshold2=edge_threshold2,
                    black_white=black_white
                )
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                continue
        
        logger.info(f"Batch processing completed. {processed_count}/{len(image_files)} images processed successfully.")
        return processed_count


def segment_image(img_path: str, 
                 n_colors: int = 25, 
                 smoothing: float = 3.0, 
                 add_edges: bool = True) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    
    This is the original function translated from Chinese to English.
    For new code, use the ImageVectorizer class instead.
    """
    vectorizer = ImageVectorizer()
    return vectorizer.vectorize_image(
        input_path=img_path,
        n_colors=n_colors,
        smoothing=smoothing,
        add_edges=add_edges
    )


def simple_vectorize(input_image: str, output_image: str = "vectorized.png"):
    """
    SUPER SIMPLE function - just vectorize an image with default settings.
    
    Usage:
        simple_vectorize("my_photo.jpg")                    # Creates vectorized.png
        simple_vectorize("my_photo.jpg", "my_result.png")   # Creates my_result.png
    
    To change colors/settings, edit the DEFAULT_COLORS value at the top of this file.
    """
    vectorizer = ImageVectorizer()
    return vectorizer.vectorize_image(input_image, output_image)


if __name__ == "__main__":
    # Example usage
    vectorizer = ImageVectorizer()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process the example image with current settings
    input_path = os.path.join(script_dir, "examples", "room-original.png")
    output_path = os.path.join(script_dir, "examples", "room-current-style.png")

    if os.path.exists(input_path):
        result = vectorizer.vectorize_image(
            input_path=input_path,
            output_path=output_path
        )
        print("✅ Image processed successfully!")
        print(f"📁 Output saved to: {output_path}")
        print(f"🎨 Used settings: Colors={DEFAULT_COLORS}, Edges={DEFAULT_EDGE_THRESHOLD1},{DEFAULT_EDGE_THRESHOLD2}")
    else:
        print(f"❌ Input image not found: {input_path}")
        print("🔍 Looking for: examples/room-rendering.png")
