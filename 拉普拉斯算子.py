import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os
from scipy.ndimage import convolve  # Import convolve for efficient filtering


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
    # If the image is float, clip to 0-255. If integer and >255, scale down.
    if gray_image.dtype != np.uint8:
        if np.issubdtype(gray_image.dtype, np.floating):
            gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
        elif np.issubdtype(gray_image.dtype, np.integer):
            if np.max(gray_image) > 255:
                gray_image = (gray_image / np.max(gray_image) * 255).astype(np.uint8)
            else:
                gray_image = gray_image.astype(np.uint8)
    return gray_image


def apply_laplacian_sharpening(image, laplacian_kernel, c=-1):
    """
    Applies a Laplacian filter to an image and then sharpens it.
    Sharpened_image = Original_image + c * Laplacian_image

    Args:
        image (np.array): 8-bit grayscale input image.
        laplacian_kernel (np.array): The Laplacian filter kernel.
        c (int/float): Scaling factor for the Laplacian image.
                       Typically -1 for sharpening when the kernel's center is negative.

    Returns:
        tuple: (laplacian_image_display, sharpened_image)
               laplacian_image_display: The Laplacian image, clipped to [0, 255] for display.
               sharpened_image: The sharpened image (uint8).
    """
    # Convert image to float for convolution to handle negative values
    image_float = image.astype(float)

    # Apply Laplacian filter
    # mode='nearest' padding is often used for sharpening to avoid dark borders
    laplacian_response = convolve(image_float, laplacian_kernel, mode='nearest', cval=0.0)

    # For display, clip Laplacian response to [0, 255]
    # Note: The textbook states negative values are clipped to 0, making most of it black.
    laplacian_image_display = np.clip(laplacian_response, 0, 255).astype(np.uint8)

    # Sharpen the image: g(x,y) = f(x,y) + c * [Laplacian_response]
    sharpened_image_float = image_float + c * laplacian_response

    # Clip the sharpened image to [0, 255] and convert back to uint8
    sharpened_image = np.clip(sharpened_image_float, 0, 255).astype(np.uint8)

    return laplacian_image_display, sharpened_image


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Image filename to process
    image_filename = "Fig0338(a)(blurry_moon).tif"
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

    # Define Laplacian kernels as per textbook Fig 3.45
    # Kernel (a) - 4-connected Laplacian
    laplacian_kernel_a = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=float)

    # Kernel (b) - 8-connected Laplacian
    laplacian_kernel_b = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ], dtype=float)

    # 2. Apply Laplacian sharpening with Kernel (a)
    print("Applying Laplacian sharpening with Kernel (a)...")
    laplacian_img_a_display, sharpened_img_a = apply_laplacian_sharpening(original_image, laplacian_kernel_a, c=-1)
    print("Finished sharpening with Kernel (a).")

    # 3. Apply Laplacian sharpening with Kernel (b)
    print("Applying Laplacian sharpening with Kernel (b)...")
    laplacian_img_b_display, sharpened_img_b = apply_laplacian_sharpening(original_image, laplacian_kernel_b, c=-1)
    print("Finished sharpening with Kernel (b).")

    # --- Visualization ---
    plt.figure(figsize=(15, 12))  # Adjust figure size for 2x2 layout
    plt.suptitle(f'Laplacian Sharpening for {image_filename}', fontsize=16)

    # Subplot 1: Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Blurry Moon Image')
    plt.axis('off')

    # Subplot 2: Laplacian Image (from Kernel a), clipped for display
    plt.subplot(2, 2, 2)
    plt.imshow(laplacian_img_a_display, cmap='gray', vmin=0, vmax=255)
    plt.title('b) Laplacian Image (Kernel a, clipped)')
    plt.axis('off')

    # Subplot 3: Sharpened Image with Kernel (a)
    plt.subplot(2, 2, 3)
    plt.imshow(sharpened_img_a, cmap='gray', vmin=0, vmax=255)
    plt.title('c) Sharpened Image (Kernel a, $c=-1$)')
    plt.axis('off')

    # Subplot 4: Sharpened Image with Kernel (b)
    plt.subplot(2, 2, 4)
    plt.imshow(sharpened_img_b, cmap='gray', vmin=0, vmax=255)
    plt.title('d) Sharpened Image (Kernel b, $c=-1$)')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_laplacian_sharpening_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()  # Close the plot to free memory

    # Save individual processed images
    imageio.imwrite(os.path.join(output_dir, f"{base_name}_laplacian_a_display.tif"), laplacian_img_a_display)
    print(
        f"Laplacian image (Kernel a, display) saved to: {os.path.join(output_dir, f'{base_name}_laplacian_a_display.tif')}")

    imageio.imwrite(os.path.join(output_dir, f"{base_name}_sharpened_a.tif"), sharpened_img_a)
    print(f"Sharpened image (Kernel a) saved to: {os.path.join(output_dir, f'{base_name}_sharpened_a.tif')}")

    imageio.imwrite(os.path.join(output_dir, f"{base_name}_laplacian_b_display.tif"), laplacian_img_b_display)
    print(
        f"Laplacian image (Kernel b, display) saved to: {os.path.join(output_dir, f'{base_name}_laplacian_b_display.tif')}")

    imageio.imwrite(os.path.join(output_dir, f"{base_name}_sharpened_b.tif"), sharpened_img_b)
    print(f"Sharpened image (Kernel b) saved to: {os.path.join(output_dir, f'{base_name}_sharpened_b.tif')}")

    print("\nImage processing complete.")

