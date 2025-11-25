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

def apply_sobel_gradient(image):
    """
    Applies Sobel operators to compute the gradient magnitude of an image.
    This corresponds to Fig 3.51(b) in the textbook.

    Args:
        image (np.array): 8-bit grayscale input image.

    Returns:
        np.array: Gradient magnitude image (uint8).
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be 8-bit (uint8) for Sobel gradient calculation.")

    # Define Sobel kernels for horizontal (Gx) and vertical (Gy) gradients
    # These correspond to Fig 3.50(d) and (e) in the textbook
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=float)

    # Convert image to float for convolution to handle intermediate values
    image_float = image.astype(float)

    # Compute gradients in x and y directions
    # mode='nearest' padding is suitable for edge detection
    Gx = convolve(image_float, sobel_x, mode='nearest')
    Gy = convolve(image_float, sobel_y, mode='nearest')

    # Compute gradient magnitude: M = sqrt(Gx^2 + Gy^2)
    # This is typically done before scaling to 0-255
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)

    # Scale the gradient magnitude to the full 0-255 range for display
    # This ensures the edges are clearly visible as white against a black background
    # as shown in Fig 3.51(b)
    max_val = np.max(gradient_magnitude)
    if max_val > 0:
        scaled_gradient_magnitude = (gradient_magnitude / max_val * 255)
    else: # Handle case where image is uniform (no edges)
        scaled_gradient_magnitude = np.zeros_like(gradient_magnitude)

    # Clip values to [0, 255] and convert back to uint8
    final_gradient_image = np.clip(scaled_gradient_magnitude, 0, 255).astype(np.uint8)

    return final_gradient_image

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image" # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

    # Image filename to process
    image_filename = "Fig0342(a)(contact_lens_original).tif"
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

    # 2. Apply Sobel gradient operator
    print("Applying Sobel gradient operator...")
    gradient_image = apply_sobel_gradient(original_image)
    print("Finished applying Sobel gradient operator.")

    # --- Visualization ---
    plt.figure(figsize=(12, 6)) # Adjust figure size for 1x2 layout
    plt.suptitle(f'Gradient Edge Enhancement for {image_filename}', fontsize=16)

    # Subplot 1: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Contact Lens Image')
    plt.axis('off')

    # Subplot 2: Sobel Gradient Magnitude Image
    plt.subplot(1, 2, 2)
    plt.imshow(gradient_image, cmap='gray', vmin=0, vmax=255)
    plt.title('b) Sobel Gradient Magnitude')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_gradient_enhancement_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close() # Close the plot to free memory

    # Save the gradient magnitude image separately
    gradient_output_path = os.path.join(output_dir, f"{base_name}_sobel_gradient.tif")
    imageio.imwrite(gradient_output_path, gradient_image)
    print(f"Sobel gradient image saved to: {os.path.join(output_dir, f'{base_name}_sobel_gradient.tif')}")

    print("\nImage processing complete.")

