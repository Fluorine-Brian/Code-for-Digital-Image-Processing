import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os


def to_grayscale(image):
    """
    Converts an image to 8-bit grayscale if it's not already.
    Handles 2D (grayscale), 3D (RGB), and 4D (RGBA) images.
    """
    if len(image.shape) == 2:  # Already 2D grayscale (H, W)
        gray_image = image
    elif len(image.shape) == 3:
        if image.shape[2] == 1:  # 3D grayscale (H, W, 1)
            gray_image = image.squeeze(axis=2)  # Remove the channel dimension
        elif image.shape[2] == 3:  # RGB image (H, W, 3)
            # Convert RGB to grayscale using luminance method
            gray_image = (0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2])
        elif image.shape[2] == 4:  # RGBA image (H, W, 4)
            # Discard alpha channel and convert RGB part to grayscale
            gray_image = (0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2])
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


def local_contrast_enhancement(image, kernel_size=3, k0=0.4, k1=0.02, E=4.0):
    """
    Performs local contrast enhancement based on local mean and standard deviation.
    This implements the method described in Example 3.10 (Equation 3.29).

    Args:
        image (np.array): 8-bit grayscale input image.
        kernel_size (int): Size of the square neighborhood (e.g., 3 for 3x3). Must be an odd number.
        k0 (float): Threshold constant for local mean (m_xy <= k0 * M).
        k1 (float): Threshold constant for local standard deviation (sigma_xy <= k1 * sigma).
        E (float): Enhancement factor for pixels meeting the criteria.

    Returns:
        np.array: Enhanced image (uint8).
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be 8-bit (uint8) for local contrast enhancement.")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    H, W = image.shape
    output_image = np.zeros_like(image, dtype=np.uint8)
    pad_size = kernel_size // 2

    # Pad the image to handle border regions
    padded_image = np.pad(image, pad_size, mode='edge')

    # Calculate global mean (M) and global standard deviation (sigma)
    M = np.mean(image)
    sigma = np.std(image)

    print(f"Global Mean (M): {M:.2f}, Global Std Dev (sigma): {sigma:.2f}")
    print(f"Thresholds: k0*M = {k0 * M:.2f}, k1*sigma = {k1 * sigma:.2f}")

    # Convert image to float for calculations to avoid overflow
    image_float = image.astype(float)
    padded_image_float = padded_image.astype(float)

    for r in range(H):
        for c in range(W):
            # Extract the local neighborhood from the padded image
            neighborhood = padded_image_float[r: r + kernel_size, c: c + kernel_size]

            # Calculate local mean (m_xy) and local standard deviation (sigma_xy)
            m_xy = np.mean(neighborhood)
            sigma_xy = np.std(neighborhood)

            # Get the original pixel value
            f_xy = image_float[r, c]

            # Apply the enhancement condition (Equation 3.29)
            if m_xy <= k0 * M and sigma_xy <= k1 * sigma:
                g_xy = E * f_xy
            else:
                g_xy = f_xy

            # Clip the result to [0, 255] and store
            output_image[r, c] = np.clip(g_xy, 0, 255).astype(np.uint8)

    return output_image


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Image filename to process
    # IMPORTANT: Ensure this is a .tif file as specified by the user
    image_filename = "Fig0326(a)(embedded_square_noisy_512).tif"
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

    # 2. Perform Local Contrast Enhancement (Example 3.10 method)
    # Textbook parameters: kernel_size=3, k0=0.4, k1=0.02, E=4.0
    kernel_size = 3
    k0 = 0.4
    k1 = 0.02
    E = 4.0
    print(
        f"Performing local contrast enhancement with kernel_size={kernel_size}, k0={k0}, k1={k1}, E={E}. This may take a moment...")
    enhanced_image = local_contrast_enhancement(original_image, kernel_size=kernel_size, k0=k0, k1=k1, E=E)
    print("Performed local contrast enhancement.")

    # --- Visualization ---
    plt.figure(figsize=(12, 6))  # Adjust figure size for 1x2 layout
    plt.suptitle(f'Local Contrast Enhancement (Example 3.10) for {image_filename}', fontsize=16)

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Image')
    plt.axis('off')

    # Enhanced Image
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'b) Enhanced Image (k0={k0}, k1={k1}, E={E})')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_local_enhancement_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()  # Close the plot to free memory

    # Save the enhanced image separately
    enhanced_image_path = os.path.join(output_dir, f"{base_name}_enhanced.tif")
    imageio.imwrite(enhanced_image_path, enhanced_image)
    print(f"Enhanced image saved to: {os.path.join(output_dir, f'{base_name}_enhanced.tif')}")

    print("\nImage processing complete.")

