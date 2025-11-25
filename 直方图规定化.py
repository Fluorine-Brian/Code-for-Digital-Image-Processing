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
    Returns the equalized image and the transformation function T(r).
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be 8-bit (uint8) for histogram equalization.")

    hist = calculate_histogram(image)
    cdf = hist.cumsum()

    cdf_min = cdf.min()
    cdf_max = cdf.max()

    if cdf_max == cdf_min:
        equalized_image = np.full_like(image, 127, dtype=np.uint8)
        transform_func = np.full(256, 127, dtype=np.uint8)
    else:
        transform_func = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
        transform_func = transform_func.astype('uint8')
        equalized_image = transform_func[image]

    return equalized_image, transform_func


def histogram_specification(input_image, target_histogram):
    """
    Performs histogram specification (matching) on an 8-bit grayscale image.
    Matches the input image's histogram to the target_histogram.
    Returns the specified image and the mapping function G_inv(s).
    """
    if input_image.dtype != np.uint8:
        raise ValueError("Input image must be 8-bit (uint8) for histogram specification.")
    if target_histogram.shape[0] != 256:
        raise ValueError("Target histogram must have 256 bins.")

    # 1. Calculate CDF of the input image
    input_hist = calculate_histogram(input_image)
    input_cdf = input_hist.cumsum()
    input_cdf_normalized = input_cdf / input_cdf[-1]  # Normalize to [0, 1]

    # 2. Calculate CDF of the target histogram
    target_cdf = target_histogram.cumsum()
    target_cdf_normalized = target_cdf / target_cdf[-1]  # Normalize to [0, 1]

    # 3. Create a lookup table (mapping function G_inv(s))
    # For each gray level 'r' in the input image, find 'z' such that T(r) approx G(z)
    # where T(r) is input_cdf_normalized[r] and G(z) is target_cdf_normalized[z]
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find the index 'j' in target_cdf_normalized where the value is >= input_cdf_normalized[i]
        # This effectively finds z such that G(z) >= T(r)
        j = np.searchsorted(target_cdf_normalized, input_cdf_normalized[i], side='left')
        # Ensure j stays within bounds [0, 255]
        lookup_table[i] = min(j, 255)

    # 4. Apply the lookup table to the input image
    specified_image = lookup_table[input_image]

    return specified_image.astype('uint8'), lookup_table


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original images are located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Image filename to process
    image_filename = "Fig0324.png"  # Assuming Fig0324.png is in original_image folder
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
        original_image = np.random.randint(0, 256, size=(400, 400), dtype=np.uint8)
    except ValueError as e:
        print(f"Error processing '{image_filename}': {e}")
        print("Creating a random 8-bit grayscale image for demonstration.")
        original_image = np.random.randint(0, 256, size=(400, 400), dtype=np.uint8)

    # 2. Perform Histogram Equalization
    equalized_image, equalization_transform_func = histogram_equalization(original_image)
    print("Performed histogram equalization.")

    # 3. Define a Target Histogram for Specification (matching Fig 3.25(a))
    # This target histogram has a narrow peak in the dark region.
    target_histogram = np.zeros(256, dtype=float)
    # Approximate the shape from Fig 3.25(a) - a sharp peak at low gray levels
    # Using an exponential decay for demonstration
    for i in range(256):
        target_histogram[i] = np.exp(-i / 10.0)  # Adjust 10.0 for desired decay rate
    target_histogram[0:10] *= 5  # Make the very first bins significantly higher
    target_histogram = target_histogram / target_histogram.sum() * original_image.size  # Scale to total pixels

    # 4. Perform Histogram Specification
    specified_image, specification_lookup_table = histogram_specification(original_image, target_histogram)
    print("Performed histogram specification.")

    # 5. Calculate Histograms for all images
    original_hist = calculate_histogram(original_image)
    equalized_hist = calculate_histogram(equalized_image)
    specified_hist = calculate_histogram(specified_image)
    print("Calculated all histograms.")

    # --- Visualization ---
    plt.figure(figsize=(18, 15))  # Adjust figure size for better display

    # Row 1: Original Image and its Histogram
    plt.subplot(3, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Image')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.plot(original_hist, color='black')
    plt.title('b) Original Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Pixel Count')
    plt.xlim([0, 255])
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(3, 3, 3)
    plt.plot(target_histogram, color='red')
    plt.title('Target Histogram $p_z(z)$')
    plt.xlabel('Gray Level')
    plt.ylabel('Pixel Count')
    plt.xlim([0, 255])
    plt.grid(True, linestyle='--', alpha=0.6)

    # Row 2: Equalized Image, Equalized Histogram, and Equalization Transform Function
    plt.subplot(3, 3, 4)
    plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
    plt.title('c) Equalized Image')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.plot(equalized_hist, color='black')
    plt.title('d) Equalized Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Pixel Count')
    plt.xlim([0, 255])
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(3, 3, 6)
    plt.plot(np.arange(256), equalization_transform_func, color='blue')
    plt.plot([0, 255], [0, 255], 'k--', alpha=0.7, label='Identity')  # Identity line
    plt.title('e) Equalization Transform $T(r)$')
    plt.xlabel('Input Gray Level $r$')
    plt.ylabel('Output Gray Level $s$')
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Row 3: Specified Image, Specified Histogram, and Specification Transform Function
    plt.subplot(3, 3, 7)
    plt.imshow(specified_image, cmap='gray', vmin=0, vmax=255)
    plt.title('f) Specified Image')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.plot(specified_hist, color='black')
    plt.title('g) Specified Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Pixel Count')
    plt.xlim([0, 255])
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(3, 3, 9)
    # Plot G(z) (target CDF) and G_inv(s) (specification lookup table)
    # For G(z), we use the normalized target CDF
    target_cdf_normalized_for_plot = target_histogram.cumsum() / target_histogram.cumsum()[-1] * 255
    plt.plot(np.arange(256), target_cdf_normalized_for_plot, color='green', label='$G(z)$ (Target CDF)')
    plt.plot(np.arange(256), specification_lookup_table, color='purple', linestyle='--', label='$G^{-1}(s)$ (Mapping)')
    plt.plot([0, 255], [0, 255], 'k--', alpha=0.7, label='Identity')  # Identity line
    plt.title('h) Specification Transforms')
    plt.xlabel('Input Gray Level')
    plt.ylabel('Output Gray Level')
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_hist_comparison_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()  # Close the plot to free memory

    # Save individual processed images
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_equalized.tif"), equalized_image)
    print(f"Equalized image saved to: {os.path.join(output_dir, f'{base_name}_equalized.tif')}")

    imageio.imwrite(os.path.join(output_dir, f"{base_name}_specified.tif"), specified_image)
    print(f"Specified image saved to: {os.path.join(output_dir, f'{base_name}_specified.tif')}")

    print("\nImage processing complete.")

