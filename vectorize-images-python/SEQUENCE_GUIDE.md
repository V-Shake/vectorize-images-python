"""
🎬 SEQUENCE VECTORIZER GUIDE
How to process image sequences with consistent colors

Perfect for animations, video frames, or any series of images that should look consistent!

=== QUICK START ===

1. Put your image sequence in: examples_image_sequence/room-animation/
2. Run: python sequence_vectorizer.py
3. Find results in: examples_image_sequence/stylized_output/

=== CUSTOMIZATION ===

Edit sequence_vectorizer.py at the top:

🎨 SEQUENCE_COLORS = 6          # Number of colors (4-12 works great)
📏 SEQUENCE_SMOOTHING = 2.0     # Smoothing (0-10)
🖼️  SEQUENCE_ADD_EDGES = True    # Black edges on/off
⚫ SEQUENCE_BLACK_WHITE = False  # ← Change to True for B&W mode!

For black & white sequences:
- Set SEQUENCE_BLACK_WHITE = True
- Set SEQUENCE_COLORS = 6 (try 4-10 for different grey levels)
- Run: python sequence_vectorizer.py

=== DIFFERENCE FROM NORMAL BATCH PROCESSING ===

Normal batch processing (main.py --input_dir):
❌ Each image gets its own color palette → inconsistent colors
❌ Frame 1 might be red/blue, Frame 2 might be green/yellow
❌ Causes flickering in animations

Sequence processing (sequence_vectorizer.py):  
✅ All images share the same color palette → consistent colors
✅ Frame 1 red stays red in Frame 2, Frame 3, etc.
✅ Perfect for smooth animations

=== YOUR WORKFLOW ===

For single images:          python image_vectorizer.py
For mixed batch:            python main.py --input_dir folder/ --output_dir out/
For consistent sequences:   python sequence_vectorizer.py

=== EXAMPLES ===

Color sequence (6 colors):
    SEQUENCE_COLORS = 6
    SEQUENCE_BLACK_WHITE = False

Black & white manga (8 greys):
    SEQUENCE_COLORS = 8  
    SEQUENCE_BLACK_WHITE = True
    SEQUENCE_EDGE_THRESHOLD1 = 30  # Thick lines
    SEQUENCE_EDGE_THRESHOLD2 = 100

Clean minimal (3 colors):
    SEQUENCE_COLORS = 3
    SEQUENCE_BLACK_WHITE = False
    SEQUENCE_SMOOTHING = 3.0

=== PERFORMANCE TIPS ===

Slow? Increase SAMPLING_RATE from 50 to 100 (faster but less accurate color analysis)
Want perfect colors? Decrease SAMPLING_RATE to 20 (slower but more accurate)

=== TECHNICAL NOTES ===

The sequence vectorizer:
1. Samples colors from ALL images first
2. Creates one unified color palette via k-means
3. Applies this same palette to every image
4. Ensures perfect visual consistency across the sequence

This is much better than processing each frame individually!
"""
