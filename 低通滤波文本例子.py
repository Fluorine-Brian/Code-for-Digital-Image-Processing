import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def pad_image_for_dft(image, padding_mode='reflect'):
    """
    Pads the image to a size suitable for DFT (e.g., 2*M x 2*N).
    Uses 'reflect' mode for padding to avoid dark borders.
    """
    M, N = image.shape
    P, Q = 2 * M, 2 * N  # Double the dimensions

    # Calculate padding amounts
    pad_h = P - M
    pad_w = Q - N

    # Pad the image using specified mode
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode=padding_mode)

    return padded_image, (M, N)  # Return padded image and original shape


def create_gaussian_lowpass_filter(shape, D0):
    """
    Creates a Gaussian Lowpass Filter (GLPF) in the frequency domain.
    H(u,v) = exp(-D(u,v)^2 / (2*D0^2)).

    Args:
        shape (tuple): (P, Q) dimensions of the frequency domain.
        D0 (float): Cutoff frequency (standard deviation of the Gaussian).

    Returns:
        np.array: The GLPF filter mask.
    """
    P, Q = shape
    H = np.zeros((P, Q), dtype=float)

    # Center of the frequency rectangle
    center_u, center_v = P / 2, Q / 2

    # Create a grid of distances from the center
    u_coords = np.arange(P) - center_u
    v_coords = np.arange(Q) - center_v
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    D_uv_squared = U ** 2 + V ** 2

    # Apply GLPF formula
    if D0 == 0:
        H = np.ones(shape, dtype=float)  # Pass all frequencies if D0 is 0
    else:
        H = np.exp(-D_uv_squared / (2 * D0 ** 2))

    return H


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Image filename to process
    image_filename = "Fig0419(a)(text_gaps_of_1_and_2_pixels).tif"
    image_path = os.path.join(input_dir, image_filename)
    base_name = os.path.splitext(image_filename)[0]

    print(f"\nProcessing image: {image_filename}")

    # 1. Read and convert image to 8-bit grayscale
    try:
        original_image = imageio.imread(image_path)
        print(f"Successfully loaded and converted '{image_filename}' to 8-bit grayscale.")
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        print("Please ensure the image file is in the 'original_image' directory or provide the full path.")
        print("Creating a random 8-bit grayscale image for demonstration.")
        original_image = np.random.randint(0, 256, size=(400, 600), dtype=np.uint8)  # Demo size
    except ValueError as e:
        print(f"Error processing '{image_filename}': {e}")
        print("Creating a random 8-bit grayscale image for demonstration.")
        original_image = np.random.randint(0, 256, size=(400, 600), dtype=np.uint8)  # Demo size

    # Store original image shape for cropping later
    original_M, original_N = original_image.shape

    # 2. Pad the image for DFT using 'reflect' mode
    padded_image, _ = pad_image_for_dft(original_image, padding_mode='reflect')
    P, Q = padded_image.shape
    print(f"Image padded from {original_M}x{original_N} to {P}x{Q} using 'reflect' mode.")

    # 3. Compute 2D DFT and center it
    dft_original = np.fft.fft2(padded_image.astype(float))
    centered_dft = np.fft.fftshift(dft_original)
    print("Computed and centered DFT.")

    # Define cutoff frequency D0 as per textbook (D0=120)
    D0_value = 120

    # 4. Apply GLPF with D0=120
    print(f"Applying Gaussian Lowpass Filter with D0 = {D0_value}...")

    # Create GLPF mask
    H = create_gaussian_lowpass_filter((P, Q), D0_value)

    # Apply filter in frequency domain
    filtered_dft = centered_dft * H

    # Compute inverse DFT and shift back
    idft_shifted = np.fft.ifftshift(filtered_dft)
    filtered_image_complex = np.fft.ifft2(idft_shifted)

    # Take the real part and crop to original size
    filtered_image = np.real(filtered_image_complex)[0:original_M, 0:original_N]

    # Scale to 0-255 and convert to uint8
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    print(f"Finished GLPF with D0={D0_value}.")

    # --- Visualization (Matching Fig 4.48(a) and (b)) ---
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Text Gap Repair using GLPF for {image_filename}', fontsize=16)

    # Subplot 1: Original Image (Fig 4.48(a))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Text with Gaps')
    plt.axis('off')

    # Subplot 2: Filtered Image (Fig 4.48(b))
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'b) GLPF Filtered (D0={D0_value})')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_glpf_text_repair_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()  # Close the plot to free memory

    # Save the filtered image
    filtered_image_path = os.path.join(output_dir, f"{base_name}_glpf_D0_{D0_value}.tif")
    imageio.imwrite(filtered_image_path, filtered_image)
    print(f"GLPF filtered image (D0={D0_value}) saved to: {filtered_image_path}")

    print("\nImage processing complete.")

