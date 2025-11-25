import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, median_filter  # Import specific filters


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


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Image filename to process
    image_filename = "Fig0335(a)(ckt_board_saltpep_prob_pt05).tif"
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

    # 2. Apply Gaussian Lowpass Filter (as per Fig 3.43(b))
    # Kernel size 19x19, standard deviation sigma=3
    # scipy.ndimage.gaussian_filter takes sigma directly.
    # The size of the filter is implicitly determined by sigma, or can be specified by 'size' parameter.
    # For sigma=3, a size of 19x19 (approx 6*sigma + 1) is reasonable.
    gaussian_sigma = 3
    print(f"Applying Gaussian filter with sigma={gaussian_sigma}...")
    gaussian_filtered_image = gaussian_filter(original_image, sigma=gaussian_sigma)
    gaussian_filtered_image = np.clip(gaussian_filtered_image, 0, 255).astype(np.uint8)
    print("Finished Gaussian filtering.")

    # 3. Apply Median Filter (as per Fig 3.43(c))
    # Kernel size 7x7
    median_kernel_size = 7
    print(f"Applying Median filter with kernel size {median_kernel_size}x{median_kernel_size}...")
    median_filtered_image = median_filter(original_image, size=median_kernel_size)
    print("Finished Median filtering.")

    # --- Visualization ---
    plt.figure(figsize=(18, 6))  # Adjust figure size for 1x3 layout
    plt.suptitle(f'Noise Filtering Comparison for {image_filename}', fontsize=16)

    # Subplot 1: Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Image (Salt-and-Pepper Noise)')
    plt.axis('off')

    # Subplot 2: Gaussian Filtered Image
    plt.subplot(1, 3, 2)
    plt.imshow(gaussian_filtered_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'b) Gaussian Filter ($\\sigma={gaussian_sigma}$)')
    plt.axis('off')

    # Subplot 3: Median Filtered Image
    plt.subplot(1, 3, 3)
    plt.imshow(median_filtered_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'c) Median Filter ({median_kernel_size}x{median_kernel_size})')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_noise_filtering_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()  # Close the plot to free memory

    # Save individual filtered images
    gaussian_output_path = os.path.join(output_dir, f"{base_name}_gaussian_filtered.tif")
    imageio.imwrite(gaussian_output_path, gaussian_filtered_image)
    print(f"Gaussian filtered image saved to: {gaussian_output_path}")

    median_output_path = os.path.join(output_dir, f"{base_name}_median_filtered.tif")
    imageio.imwrite(median_output_path, median_filtered_image)
    print(f"Median filtered image saved to: {median_output_path}")

    print("\nImage processing complete.")

