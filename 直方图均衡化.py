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


def calculate_histogram(image, bins=256):
    """
    Calculates the histogram of an 8-bit grayscale image.
    """
    hist, _ = np.histogram(image.flatten(), bins, [0, bins])
    return hist


def histogram_equalization(image):
    """
    Performs histogram equalization on an 8-bit grayscale image.
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be 8-bit (uint8) for histogram equalization.")

    # Calculate histogram
    hist = calculate_histogram(image)

    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Normalize CDF to [0, 255]
    # Handle cases where cdf_min is 0 (e.g., empty image or all pixels are 0)
    cdf_min = cdf.min()
    cdf_max = cdf.max()

    if cdf_max == cdf_min:  # Avoid division by zero if all pixels are same value
        equalized_image = np.full_like(image, 127, dtype=np.uint8)  # Set to mid-gray
    else:
        cdf_normalized = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
        # Map original pixel values to new values using the normalized CDF
        equalized_image = cdf_normalized[image]

    return equalized_image.astype('uint8')


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original images are located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # List of image filenames to process
    image_filenames = [
        "Fig0320(1)(top_left).tif",
        "Fig0320(2)(2nd_from_top).tif",
        "Fig0320(3)(third_from_top).tif",
        "Fig0320(4)(bottom_left).tif"
    ]

    for filename in image_filenames:
        image_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]  # Get filename without extension

        print(f"\nProcessing image: {filename}")

        # 1. Read and convert image to 8-bit grayscale
        try:
            original_image_raw = imageio.imread(image_path)
            original_image = to_grayscale(original_image_raw)
            print(f"Successfully loaded and converted '{filename}' to 8-bit grayscale.")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            print("Please ensure the image file is in the 'original_image' directory or provide the full path.")
            print(f"Skipping '{filename}'.")
            continue  # Skip to the next image
        except ValueError as e:
            print(f"Error processing '{filename}': {e}")
            print(f"Skipping '{filename}'.")
            continue

        # 2. Perform histogram equalization
        equalized_image = histogram_equalization(original_image)
        print(f"Performed histogram equalization on '{filename}'.")

        # 3. Calculate histograms
        original_hist = calculate_histogram(original_image)
        equalized_hist = calculate_histogram(equalized_image)
        print(f"Calculated histograms for '{filename}'.")

        # --- Visualization ---
        plt.figure(figsize=(12, 10))
        plt.suptitle(f'Histogram Equalization for {filename}', fontsize=16)

        # Original Image
        plt.subplot(2, 2, 1)
        plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
        plt.title('Original Image')
        plt.axis('off')

        # Original Histogram
        plt.subplot(2, 2, 2)
        plt.plot(original_hist, color='black')
        plt.title('Original Histogram')
        plt.xlabel('Gray Level')
        plt.ylabel('Pixel Count')
        plt.xlim([0, 255])
        plt.grid(True, linestyle='--', alpha=0.6)

        # Equalized Image
        plt.subplot(2, 2, 3)
        plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
        plt.title('Equalized Image')
        plt.axis('off')

        # Equalized Histogram
        plt.subplot(2, 2, 4)
        plt.plot(equalized_hist, color='black')
        plt.title('Equalized Histogram')
        plt.xlabel('Gray Level')
        plt.ylabel('Pixel Count')
        plt.xlim([0, 255])
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

        # --- Save Visualization and Individual Images ---
        # Save the combined visualization figure
        combined_output_path = os.path.join(output_dir, f"{base_name}_equalization_results.png")
        plt.savefig(combined_output_path, bbox_inches='tight')
        print(f"Combined visualization saved to: {combined_output_path}")
        plt.close()  # Close the plot to free memory

        # Save the equalized image separately
        equalized_image_path = os.path.join(output_dir, f"{base_name}_equalized.tif")
        imageio.imwrite(equalized_image_path, equalized_image)
        print(f"Equalized image saved to: {equalized_image_path}")

    print("\nAll images processed.")

