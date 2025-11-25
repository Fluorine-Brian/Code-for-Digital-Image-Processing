import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os


# --- Helper Functions (reused from previous tasks) ---

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


def pad_image_for_dft(image):
    """
    Pads the image to a size suitable for DFT (e.g., 2*M x 2*N).
    The textbook mentions padding to double the image size.
    """
    M, N = image.shape
    P, Q = 2 * M, 2 * N  # Double the dimensions

    # Create a padded image with zeros
    padded_image = np.zeros((P, Q), dtype=image.dtype)
    padded_image[0:M, 0:N] = image

    return padded_image, (M, N)  # Return padded image and original shape


def calculate_power_percentage(centered_dft_spectrum, D0):
    """
    Calculates the percentage of total power contained within a circle of radius D0.

    Args:
        centered_dft_spectrum (np.array): The centered 2D DFT spectrum (complex values).
        D0 (float): Radius of the circle.

    Returns:
        float: Percentage of power.
    """
    P, Q = centered_dft_spectrum.shape

    # Power spectrum is |F(u,v)|^2
    power_spectrum = np.abs(centered_dft_spectrum) ** 2

    total_power = np.sum(power_spectrum)
    power_within_D0 = 0.0

    center_u, center_v = P / 2, Q / 2

    # Create a grid of distances from the center
    u_coords = np.arange(P) - P / 2
    v_coords = np.arange(Q) - Q / 2
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    D_uv = np.sqrt(U ** 2 + V ** 2)

    # Sum power where distance is within D0
    power_within_D0 = np.sum(power_spectrum[D_uv <= D0])

    if total_power > 0:
        return (power_within_D0 / total_power) * 100
    else:
        return 0.0


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Image filename to process
    image_filename = "Fig0440.tif"
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
        original_image = np.random.randint(0, 256, size=(688, 688), dtype=np.uint8)  # Use 688x688 for demo
    except ValueError as e:
        print(f"Error processing '{image_filename}': {e}")
        print("Creating a random 8-bit grayscale image for demonstration.")
        original_image = np.random.randint(0, 256, size=(688, 688), dtype=np.uint8)  # Use 688x688 for demo

    # Store original image shape
    original_M, original_N = original_image.shape

    # 2. Pad the image to double its size for DFT
    padded_image, _ = pad_image_for_dft(original_image)
    P, Q = padded_image.shape
    print(f"Image padded from {original_M}x{original_N} to {P}x{Q}.")

    # 3. Compute 2D DFT and center it
    dft_original = np.fft.fft2(padded_image.astype(float))
    centered_dft = np.fft.fftshift(dft_original)
    print("Computed and centered DFT.")

    # 4. Calculate Log Magnitude Spectrum for visualization
    # Add a small constant to avoid log(0)
    spectrum_log_magnitude = 20 * np.log(np.abs(centered_dft) + 1e-9)

    # Normalize for display to 0-255 range
    spectrum_display = (spectrum_log_magnitude - np.min(spectrum_log_magnitude)) / \
                       (np.max(spectrum_log_magnitude) - np.min(spectrum_log_magnitude)) * 255
    spectrum_display = spectrum_display.astype(np.uint8)

    # Define cutoff frequencies D0 (radii) as per textbook Fig 4.40(b)
    D0_values = [10, 30, 60, 160, 460]
    power_percentages = {}

    # 5. Calculate power percentages for each D0
    print("\nCalculating power percentages within D0 circles:")
    for D0 in D0_values:
        power_percent = calculate_power_percentage(centered_dft, D0)
        power_percentages[D0] = power_percent
        print(f"  Power within D0={D0} circle: {power_percent:.2f}%")

    # --- Visualization (Matching Fig 4.40(a) and (b)) ---
    plt.figure(figsize=(14, 7))  # Adjust figure size for 1x2 layout
    plt.suptitle(f'Figure 4.40: Test Pattern and its Frequency Spectrum for {image_filename}', fontsize=16)

    # Subplot 1: Original Image (Fig 4.40(a))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Test Pattern Image')
    plt.axis('off')

    # Subplot 2: Centered Frequency Spectrum with D0 Circles (Fig 4.40(b))
    plt.subplot(1, 2, 2)
    plt.imshow(spectrum_display, cmap='gray', vmin=0, vmax=255)
    plt.title('b) Frequency Spectrum with D0 Circles')
    plt.axis('off')

    # Draw circles for D0 values on the spectrum
    # The center of the spectrum is P/2, Q/2
    center_x, center_y = Q / 2, P / 2  # Note: imshow plots (y, x), so center_x is Q/2, center_y is P/2
    for D0 in D0_values:
        circle = plt.Circle((center_x, center_y), D0, color='red', fill=False, linewidth=1, alpha=0.7)
        plt.gca().add_patch(circle)

        # Add text for power percentage near the circle
        # Adjust text position to avoid overlap and be readable
        text_x = center_x + D0 * np.cos(np.deg2rad(45))  # Example: 45 degrees from center
        text_y = center_y + D0 * np.sin(np.deg2rad(45))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # --- Save Visualization ---
    combined_output_path = os.path.join(output_dir, f"{base_name}_fig4_40_reproduction.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Figure 4.40 reproduction saved to: {combined_output_path}")
    plt.close()  # Close the plot to free memory

    print("\nImage processing complete.")

