import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os
from scipy.ndimage import convolve  # Import convolve for efficient filtering


# --- Helper Functions ---

def normalize_to_uint8(img_float):
    """
    Linearly scale a float image to [0, 255] and convert to uint8.
    """
    img = img_float.astype(np.float64)
    min_val, max_val = img.min(), img.max()
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)
    img_norm = (img - min_val) / (max_val - min_val)
    return np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)

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


def get_laplacian_response(image, laplacian_kernel):
    """
    Applies a Laplacian filter to an image and returns the raw (float) response.
    """
    image_float = image.astype(np.float64)
    laplacian_response = convolve(image_float, laplacian_kernel, mode='nearest')
    return laplacian_response


def apply_sobel_gradient(image):
    """
    Applies Sobel operators to compute the gradient magnitude of an image.
    Returns the scaled gradient magnitude image (uint8).
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    image_float = image.astype(float)
    Gx = convolve(image_float, sobel_x, mode='nearest')
    Gy = convolve(image_float, sobel_y, mode='nearest')

    gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

    # Scale to 0-255 for display
    max_val = np.max(gradient_magnitude)
    if max_val > 0:
        scaled_gradient_magnitude = (gradient_magnitude / max_val * 255)
    else:
        scaled_gradient_magnitude = np.zeros_like(gradient_magnitude)

    return np.clip(scaled_gradient_magnitude, 0, 255).astype(np.uint8)


def apply_box_filter(image, kernel_size):
    """
    Applies a box (average) filter to the image.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size * kernel_size)
    image_float = image.astype(float)
    filtered_image_float = convolve(image_float, kernel, mode='nearest',
                                    cval=0.0)  # Use nearest for smoothing gradients
    return np.clip(filtered_image_float, 0, 255).astype(np.uint8)


def power_law_transform(image, gamma):
    """
    Applies a power-law (gamma) transformation to an 8-bit grayscale image.
    s = c * r^gamma, where c is typically 1 for 0-255 range.
    """
    # Normalize image to [0, 1] for power-law transformation
    normalized_image = image.astype(float) / 255.0

    # Apply power-law transformation
    transformed_image = normalized_image ** gamma

    # Scale back to [0, 255] and convert to uint8
    return np.clip(transformed_image * 255, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Image filename to process
    image_filename = "Fig0343(a)(skeleton_orig).tif"
    image_path = os.path.join(input_dir, image_filename)
    base_name = os.path.splitext(image_filename)[0]

    print(f"\nProcessing image: {image_filename}")

    # 1. Read and convert image to 8-bit grayscale
    original_image = imageio.imread(image_path)

    # --- Define Kernels and Parameters ---
    # Laplacian Kernel (4-connected, center -4, as per Fig 3.45(a))
    laplacian_kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float64)
    laplacian_c = -1  # Sharpening factor for Laplacian (g = f + c*Laplacian)

    # Averaging filter for smoothing gradient (5x5 as per textbook)
    avg_kernel_size = 5

    # Power-law transformation gamma (0.5 as per textbook)
    gamma_value = 0.5

    # --- Image Processing Steps (as per Fig 3.57) ---

    # Step 1: Laplacian Image (Fig 3.57(b))
    # Get raw Laplacian response (float, can be negative)
    laplacian_response = get_laplacian_response(original_image, laplacian_kernel)
    # For display, clip negative values to 0 and scale to 0-255
    laplacian_image_display = normalize_to_uint8(laplacian_response)
    print(np.mean(laplacian_image_display))
    # laplacian_image_display = normalize_to_uint8(laplacian_image_display)
    print("Generated Laplacian image.")

    # Step 2: Sharpened Image (Laplacian) (Fig 3.57(c))
    # g(x,y) = f(x,y) + c * Laplacian_response
    sharpened_laplacian_image_float = original_image.astype(float) + laplacian_c * laplacian_response
    sharpened_laplacian_image = np.clip(sharpened_laplacian_image_float, 0, 255).astype(np.uint8)
    print("Generated Laplacian sharpened image.")

    # Step 3: Sobel Gradient Magnitude (Fig 3.57(d))
    sobel_gradient_image = apply_sobel_gradient(original_image)
    print("Generated Sobel gradient magnitude image.")

    # Step 4: Smoothed Sobel Gradient (Fig 3.57(e))
    smoothed_sobel_gradient = apply_box_filter(sobel_gradient_image, avg_kernel_size)
    print(f"Smoothed Sobel gradient with {avg_kernel_size}x{avg_kernel_size} filter.")

    # Step 5: Product Image (Fig 3.57(f))
    # Convert images to float for multiplication
    sharpened_laplacian_float = sharpened_laplacian_image.astype(float)
    smoothed_sobel_float = smoothed_sobel_gradient.astype(float)

    # Normalize smoothed_sobel_float to [0, 1] before multiplication to use it as a weight
    # This is crucial for the product to make sense as a weighted sharpening
    smoothed_sobel_normalized = smoothed_sobel_float / 255.0

    # Product = Sharpened_Laplacian * Smoothed_Sobel_Normalized
    product_image_float = sharpened_laplacian_float * smoothed_sobel_normalized
    product_image = np.clip(product_image_float, 0, 255).astype(np.uint8)
    print("Generated product image.")

    # Step 6: Power-Law Transformation (Fig 3.57(g))
    power_law_image = power_law_transform(product_image, gamma_value)
    print(f"Applied power-law transformation with gamma={gamma_value}.")

    # Step 7: Final Sharpened Image (Fig 3.57(h))
    # Final_Image = Original_Image + Power_Law_Transformed_Image
    final_sharpened_image_float = original_image.astype(float) + power_law_image.astype(float)
    final_sharpened_image = np.clip(final_sharpened_image_float, 0, 255).astype(np.uint8)
    print("Generated final sharpened image.")

    # --- Visualization ---
    plt.figure(figsize=(16, 8))  # Adjust figure size for 2x4 layout
    plt.suptitle(f'Combined Spatial Enhancement for {image_filename}', fontsize=16)

    # Row 1
    plt.subplot(2, 4, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Image')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(laplacian_image_display, cmap='gray', vmin=0, vmax=255)
    plt.title('b) Laplacian Image (Display)')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(sharpened_laplacian_image, cmap='gray', vmin=0, vmax=255)
    plt.title('c) Sharpened (Laplacian)')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(sobel_gradient_image, cmap='gray', vmin=0, vmax=255)
    plt.title('d) Sobel Gradient Magnitude')
    plt.axis('off')

    # Row 2
    plt.subplot(2, 4, 5)
    plt.imshow(smoothed_sobel_gradient, cmap='gray', vmin=0, vmax=255)
    plt.title(f'e) Smoothed Sobel Gradient ({avg_kernel_size}x{avg_kernel_size})')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(product_image, cmap='gray', vmin=0, vmax=255)
    plt.title('f) Product Image')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(power_law_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'g) Power-Law Transformed ($\\gamma={gamma_value}$)')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(final_sharpened_image, cmap='gray', vmin=0, vmax=255)
    plt.title('h) Final Sharpened Image')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_combined_enhancement_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()  # Close the plot to free memory

    # Save key intermediate and final images
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_laplacian_display.tif"), laplacian_image_display)
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_sharpened_laplacian.tif"), sharpened_laplacian_image)
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_sobel_gradient.tif"), sobel_gradient_image)
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_smoothed_sobel.tif"), smoothed_sobel_gradient)
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_product_image.tif"), product_image)
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_power_law.tif"), power_law_image)
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_final_sharpened.tif"), final_sharpened_image)
    print(f"Intermediate and final images saved to: {output_dir}")

    print("\nImage processing complete.")

