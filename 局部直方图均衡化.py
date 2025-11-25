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


def histogram_equalization_global(image):
    """
    Performs global histogram equalization on an 8-bit grayscale image.
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be 8-bit (uint8) for histogram equalization.")

    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()

    cdf_min = cdf.min()
    cdf_max = cdf.max()

    if cdf_max == cdf_min:
        equalized_image = np.full_like(image, 127, dtype=np.uint8)
    else:
        transform_func = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
        transform_func = transform_func.astype('uint8')
        equalized_image = transform_func[image]

    return equalized_image


def local_histogram_equalization(image, kernel_size=3):
    """
    Performs local histogram equalization on an 8-bit grayscale image.
    Each pixel's value is adjusted based on the histogram of its local neighborhood.

    Args:
        image (np.array): 8-bit grayscale input image.
        kernel_size (int): Size of the square neighborhood (e.g., 3 for 3x3).
                           Must be an odd number.

    Returns:
        np.array: Locally equalized image (uint8).
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be 8-bit (uint8) for local histogram equalization.")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    H, W = image.shape
    output_image = np.zeros_like(image, dtype=np.uint8)
    pad_size = kernel_size // 2

    # Pad the image to handle border regions
    # 'edge' mode repeats the edge pixels
    padded_image = np.pad(image, pad_size, mode='edge')

    for r in range(H):
        for c in range(W):
            # Extract the local neighborhood from the padded image
            neighborhood = padded_image[r: r + kernel_size, c: c + kernel_size]

            # Calculate local histogram and CDF for this neighborhood
            local_hist, _ = np.histogram(neighborhood.flatten(), bins=256, range=[0, 256])
            local_cdf = local_hist.cumsum()

            # Normalize local CDF to [0, 255]
            cdf_min = local_cdf.min()
            cdf_max = local_cdf.max()

            if cdf_max == cdf_min:
                # If all pixels in the neighborhood are the same,
                # the output pixel value remains unchanged (or set to mid-gray)
                output_image[r, c] = image[r, c]
            else:
                # Apply the equalization formula to the current pixel's value
                pixel_value = image[r, c]
                s = (local_cdf[pixel_value] - cdf_min) * 255 / (cdf_max - cdf_min)
                output_image[r, c] = np.clip(s, 0, 255).astype(np.uint8)

    return output_image


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Image filename to process
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

    # 2. Perform Global Histogram Equalization
    global_equalized_image = histogram_equalization_global(original_image)
    print("Performed global histogram equalization.")

    # 3. Perform Local Histogram Equalization (with 3x3 neighborhood as per textbook)
    kernel_size = 3
    print(
        f"Performing local histogram equalization with {kernel_size}x{kernel_size} neighborhood. This may take a moment...")
    local_equalized_image = local_histogram_equalization(original_image, kernel_size=kernel_size)
    print("Performed local histogram equalization.")

    # --- Visualization ---
    plt.figure(figsize=(18, 6))  # Adjust figure size for 1x3 layout
    plt.suptitle(f'Histogram Equalization Comparison for {image_filename}', fontsize=16)

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Image')
    plt.axis('off')

    # Global Histogram Equalization Result
    plt.subplot(1, 3, 2)
    plt.imshow(global_equalized_image, cmap='gray', vmin=0, vmax=255)
    plt.title('b) Global Histogram Equalization')
    plt.axis('off')

    # Local Histogram Equalization Result
    plt.subplot(1, 3, 3)
    plt.imshow(local_equalized_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'c) Local Histogram Equalization ({kernel_size}x{kernel_size} neighborhood)')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_hist_eq_comparison.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()  # Close the plot to free memory

    # Save individual processed images
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_global_equalized.tif"), global_equalized_image)
    print(f"Global equalized image saved to: {os.path.join(output_dir, f'{base_name}_global_equalized.tif')}")

    imageio.imwrite(os.path.join(output_dir, f"{base_name}_local_equalized.tif"), local_equalized_image)
    print(f"Local equalized image saved to: {os.path.join(output_dir, f'{base_name}_local_equalized.tif')}")

    print("\nImage processing complete.")

