import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os

# --- Helper Functions ---

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

def pad_image_for_dft(image):
    """
    Pads the image to a size suitable for DFT (e.g., 2*M x 2*N).
    The textbook mentions padding to double the image size.
    """
    M, N = image.shape
    P, Q = 2 * M, 2 * N # Double the dimensions

    # Create a padded image with zeros
    padded_image = np.zeros((P, Q), dtype=image.dtype)
    padded_image[0:M, 0:N] = image

    return padded_image, (M, N) # Return padded image and original shape

def create_ideal_lowpass_filter(shape, D0):
    """
    Creates an Ideal Lowpass Filter (ILPF) in the frequency domain.
    H(u,v) = 1 if D(u,v) <= D0, else 0.

    Args:
        shape (tuple): (P, Q) dimensions of the frequency domain.
        D0 (float): Cutoff frequency (radius of the circle).

    Returns:
        np.array: The ILPF filter mask.
    """
    P, Q = shape
    H = np.zeros((P, Q), dtype=float)

    # Center of the frequency rectangle
    center_u, center_v = P / 2, Q / 2

    # Create a grid of distances from the center
    for u in range(P):
        for v in range(Q):
            D_uv = np.sqrt((u - center_u)**2 + (v - center_v)**2)
            if D_uv <= D0:
                H[u, v] = 1
    return H

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
    power_spectrum = np.abs(centered_dft_spectrum)**2

    total_power = np.sum(power_spectrum)
    power_within_D0 = 0.0

    center_u, center_v = P / 2, Q / 2

    for u in range(P):
        for v in range(Q):
            D_uv = np.sqrt((u - center_u)**2 + (v - center_v)**2)
            if D_uv <= D0:
                power_within_D0 += power_spectrum[u, v]

    if total_power > 0:
        return (power_within_D0 / total_power) * 100
    else:
        return 0.0

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image" # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

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
        original_image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
    except ValueError as e:
        print(f"Error processing '{image_filename}': {e}")
        print("Creating a random 8-bit grayscale image for demonstration.")
        original_image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)

    # Store original image shape for cropping later
    original_M, original_N = original_image.shape

    # 2. Pad the image for DFT
    padded_image, _ = pad_image_for_dft(original_image)
    P, Q = padded_image.shape
    print(f"Image padded from {original_M}x{original_N} to {P}x{Q}.")

    # 3. Compute 2D DFT and center it
    dft_original = np.fft.fft2(padded_image.astype(float))
    centered_dft = np.fft.fftshift(dft_original)
    print("Computed and centered DFT.")

    # For visualization of spectrum (log magnitude)
    # Add a small constant to avoid log(0)
    spectrum_display = 20 * np.log(np.abs(centered_dft) + 1e-9)
    # Normalize for display
    spectrum_display = (spectrum_display - np.min(spectrum_display)) / (np.max(spectrum_display) - np.min(spectrum_display)) * 255
    spectrum_display = spectrum_display.astype(np.uint8)


    # Define cutoff frequencies D0 as per textbook (radii 10, 30, 60, 160, 460)
    # Note: The textbook image is 688x688, padded to 1376x1376.
    # If your image is smaller, these D0 values might be too large or too small.
    # For a 512x512 image padded to 1024x1024, max D is sqrt((512)^2 + (512)^2) approx 724.
    # Let's use relative D0 values or adjust based on actual image size.
    # For demonstration, we'll use the textbook's D0 values directly, assuming they are relative
    # or that the demonstration image is large enough.
    # If original_image is 688x688, then P,Q = 1376. Max D0 is sqrt(688^2+688^2) approx 973.
    # D0 values: 10, 30, 60, 160, 460.
    # The largest D0 (460) is less than half of the max possible D0 (approx 973/2 = 486.5), so it's fine.
    D0_values = [10, 30, 60, 160, 460]

    filtered_images = {}
    power_percentages = {}

    # 4. Apply ILPF for each D0
    for D0 in D0_values:
        print(f"Applying Ideal Lowpass Filter with D0 = {D0}...")

        # Create ILPF mask
        H = create_ideal_lowpass_filter((P, Q), D0)

        # Apply filter in frequency domain
        filtered_dft = centered_dft * H

        # Compute inverse DFT and shift back
        idft_shifted = np.fft.ifftshift(filtered_dft)
        filtered_image_complex = np.fft.ifft2(idft_shifted)

        # Take the real part and crop to original size
        filtered_image = np.real(filtered_image_complex)[0:original_M, 0:original_N]

        # Scale to 0-255 and convert to uint8
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        filtered_images[D0] = filtered_image

        # Calculate power percentage
        power_percent = calculate_power_percentage(centered_dft, D0)
        power_percentages[D0] = power_percent
        print(f"  Power within D0={D0} circle: {power_percent:.2f}%")

    # --- Visualization ---
    # Create a figure with 2 rows and 3 columns for original, spectrum, and 4 filtered images
    # We'll put original and spectrum in the first row, then filtered images.
    plt.figure(figsize=(18, 12))
    plt.suptitle(f'Ideal Lowpass Filtering for {image_filename}', fontsize=16)

    # Subplot 1: Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Image')
    plt.axis('off')

    # Subplot 2: Centered Frequency Spectrum (Log Magnitude)
    plt.subplot(2, 3, 2)
    plt.imshow(spectrum_display, cmap='gray', vmin=0, vmax=255)
    plt.title('b) Centered Frequency Spectrum')
    plt.axis('off')

    # Subplot 3: Placeholder or another view of spectrum
    # For Fig 4.40(b), it shows the spectrum with circles. We can draw circles on it.
    plt.subplot(2, 3, 3)
    plt.imshow(spectrum_display, cmap='gray', vmin=0, vmax=255)
    plt.title('b) Spectrum with D0 Circles')
    plt.axis('off')
    # Draw circles for D0 values
    center_x, center_y = spectrum_display.shape[1] / 2, spectrum_display.shape[0] / 2
    for D0 in D0_values:
        circle = plt.Circle((center_x, center_y), D0, color='white', fill=False, linewidth=1, alpha=0.7)
        plt.gca().add_patch(circle)


    # Subplots 4-8: Filtered Images (adjusting layout for 5 images)
    # We need 5 images, so a 2x3 layout might be tight. Let's use 3x3 for better spacing.
    # Or, if we stick to 2x3, we can put 3 in the second row, and 2 in a third row.
    # Let's try 3x3 for better visual.
    plt.figure(figsize=(18, 18))
    plt.suptitle(f'Ideal Lowpass Filtering Results for {image_filename}', fontsize=16)

    plt.subplot(3, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Image')
    plt.axis('off')

    for i, D0 in enumerate(D0_values):
        plt.subplot(3, 3, i + 2) # Start from subplot 2
        plt.imshow(filtered_images[D0], cmap='gray', vmin=0, vmax=255)
        plt.title(f'ILPF (D0={D0})\nPower: {power_percentages[D0]:.2f}%')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_ilpf_results_combined.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close() # Close the plot to free memory

    # Save individual filtered images
    for D0, img in filtered_images.items():
        filtered_image_path = os.path.join(output_dir, f"{base_name}_ilpf_D0_{D0}.tif")
        imageio.imwrite(filtered_image_path, img)
        print(f"ILPF filtered image (D0={D0}) saved to: {filtered_image_path}")

    print("\nImage processing complete.")

