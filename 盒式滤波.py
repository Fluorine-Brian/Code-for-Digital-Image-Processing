import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os
from scipy.ndimage import convolve # Import convolve for efficient filtering

def to_grayscale(image):
    """
    Converts an image to 8-bit grayscale if it's not already.
    Handles 2D (grayscale), 3D (RGB), and 4D (RGBA) images.
    """
    if len(image.shape) == 2: # Already 2D grayscale (H, W)
        gray_image = image
    elif len(image.shape) == 3:
        if image.shape[2] == 1: # 3D grayscale (H, W, 1)
            gray_image = image.squeeze(axis=2) # Remove the channel dimension
        elif image.shape[2] == 3: # RGB image (H, W, 3)
            # Convert RGB to grayscale using luminance method
            gray_image = (0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2])
        elif image.shape[2] == 4: # RGBA image (H, W, 4)
            # Discard alpha channel and convert RGB part to grayscale
            gray_image = (0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2])
        else:
            raise ValueError(f"Unsupported 3D image format with {image.shape[2]} channels.")
    else:
        raise ValueError(f"Unsupported image format with {len(image.shape)} dimensions. Expected 2D or 3D.")

    # Ensure the output is uint8 and scaled correctly
    if gray_image.dtype != np.uint8:
        if np.issubdtype(gray_image.dtype, np.floating):
            gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
        elif np.issubdtype(gray_image.dtype, np.integer):
            if np.max(gray_image) > 255:
                gray_image = (gray_image / np.max(gray_image) * 255).astype(np.uint8)
            else:
                gray_image = gray_image.astype(np.uint8)
    return gray_image

def apply_box_filter(image, kernel_size):
    """
    Applies a box (average) filter to the image using zero padding.

    Args:
        image (np.array): 8-bit grayscale input image.
        kernel_size (int): Size of the square box filter (e.g., 3 for 3x3).

    Returns:
        np.array: Filtered image (uint8).
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be 8-bit (uint8) for filtering.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")

    # Create a box filter kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size * kernel_size)

    # Convert image to float for convolution to avoid overflow
    image_float = image.astype(float)

    # Apply convolution with zero padding (mode='constant', cval=0)
    # This replicates the dark borders mentioned in the textbook
    filtered_image_float = convolve(image_float, kernel, mode='constant', cval=0.0)

    # Clip values to [0, 255] and convert back to uint8
    filtered_image = np.clip(filtered_image_float, 0, 255).astype(np.uint8)

    return filtered_image

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image" # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

    # Image filename to process
    image_filename = "Fig0333(a)(test_pattern_blurring_orig).tif"
    image_path = os.path.join(input_dir, image_filename)
    base_name = os.path.splitext(image_filename)[0]

    print(f"\nProcessing image: {image_filename}")

    # 1. Read and convert image to 8-bit grayscale
    try:
        original_image_raw = imageio.imread(image_path)
        original_image = to_grayscale(original_image_raw)
        print(f"Successfully loaded and converted '{image_filename}' to 8-bit grayscale.")
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        print("Please ensure the image file is in the 'original_image' directory or provide the full path.")
        print("Creating a random 8-bit grayscale image for demonstration.")
        original_image = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    except ValueError as e:
        print(f"Error processing '{image_filename}': {e}")
        print("Creating a random 8-bit grayscale image for demonstration.")
        original_image = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)

    # Define kernel sizes as per textbook example
    kernel_sizes = [3, 11, 21]
    filtered_images = {}

    # 2. Apply box filters with different kernel sizes
    for k_size in kernel_sizes:
        print(f"Applying {k_size}x{k_size} box filter...")
        filtered_img = apply_box_filter(original_image, k_size)
        filtered_images[k_size] = filtered_img
        print(f"Finished {k_size}x{k_size} filter.")

    # --- Visualization ---
    plt.figure(figsize=(12, 12)) # Adjust figure size for 2x2 layout
    plt.suptitle(f'Box Filtering Results for {image_filename}', fontsize=16)

    # Subplot 1: Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Image')
    plt.axis('off')

    # Subplot 2: Filtered with 3x3 kernel
    plt.subplot(2, 2, 2)
    plt.imshow(filtered_images[3], cmap='gray', vmin=0, vmax=255)
    plt.title('b) Box Filter (3x3)')
    plt.axis('off')

    # Subplot 3: Filtered with 11x11 kernel
    plt.subplot(2, 2, 3)
    plt.imshow(filtered_images[11], cmap='gray', vmin=0, vmax=255)
    plt.title('c) Box Filter (11x11)')
    plt.axis('off')

    # Subplot 4: Filtered with 21x21 kernel
    plt.subplot(2, 2, 4)
    plt.imshow(filtered_images[21], cmap='gray', vmin=0, vmax=255)
    plt.title('d) Box Filter (21x21)')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_box_filter_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close() # Close the plot to free memory

    # Save individual filtered images
    for k_size, img in filtered_images.items():
        filtered_image_path = os.path.join(output_dir, f"{base_name}_box_filter_{k_size}x{k_size}.tif")
        imageio.imwrite(filtered_image_path, img)
        print(f"Box filtered image ({k_size}x{k_size}) saved to: {filtered_image_path}")

    print("\nImage processing complete.")

