import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os


# --- Helper Functions ---

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


def create_butterworth_lowpass_filter(shape, D0, n):
    """
    Creates a Butterworth Lowpass Filter (BLPF) in the frequency domain.
    H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n)).

    Args:
        shape (tuple): (P, Q) dimensions of the frequency domain.
        D0 (float): Cutoff frequency (radius where H(u,v) is 0.5).
        n (float): Order of the Butterworth filter.

    Returns:
        np.array: The BLPF filter mask.
    """
    P, Q = shape
    H = np.zeros((P, Q), dtype=float)

    # Center of the frequency rectangle
    center_u, center_v = P / 2, Q / 2

    # Create a grid of distances from the center
    u_coords = np.arange(P) - center_u
    v_coords = np.arange(Q) - center_v
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    D_uv = np.sqrt(U ** 2 + V ** 2)

    # Apply BLPF formula
    # Avoid division by zero if D0 is 0 or D_uv is 0 and D0 is 0
    if D0 == 0:
        H = np.ones(shape, dtype=float)  # Pass all frequencies if D0 is 0
    else:
        # Handle D_uv = 0 separately to avoid division by zero in (D_uv/D0)
        # At D_uv = 0, H(0,0) should be 1.
        H = 1 / (1 + (D_uv / D0) ** (2 * n))
        # Ensure the center is exactly 1 if D0 is not 0
        H[int(center_u), int(center_v)] = 1.0

    return H


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "original_image"  # Folder where your original image is located
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Image filename to process (Fig 4.46(a) is the same as Fig 4.40(a))
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

    # Define cutoff frequencies D0 as per textbook Fig 4.40 (radii 10, 30, 60, 160, 460)
    # These D0 values are used directly for BLPF.
    D0_values = [10, 30, 60, 160, 460]
    # Define Butterworth filter order as per textbook (n=2.25)
    n_order = 2.25

    filtered_images = {}

    # 4. Apply BLPF for each D0
    for D0 in D0_values:
        print(f"Applying Butterworth Lowpass Filter with D0 = {D0}, n = {n_order}...")

        # Create BLPF mask
        H = create_butterworth_lowpass_filter((P, Q), D0, n_order)

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
        print(f"Finished BLPF with D0={D0}.")

    # --- Visualization (Matching Fig 4.46(a) to (f)) ---
    # We have 1 original + 5 filtered images = 6 images. A 2x3 layout is suitable.
    plt.figure(figsize=(18, 12))
    plt.suptitle(f'Butterworth Lowpass Filtering Results for {image_filename} (n={n_order})', fontsize=16)

    # Subplot 1: Original Image (Fig 4.46(a))
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Image')
    plt.axis('off')

    # Subplots 2-6: Filtered Images (Fig 4.46(b) to (f))
    # The textbook labels are b, c, d, e, f.
    # We will map:
    # subplot 2 -> D0=10 (b)
    # subplot 3 -> D0=30 (c)
    # subplot 4 -> D0=60 (d)
    # subplot 5 -> D0=160 (e)
    # subplot 6 -> D0=460 (f)

    subplot_labels = ['b', 'c', 'd', 'e', 'f']
    for i, D0 in enumerate(D0_values):
        plt.subplot(2, 3, i + 2)  # Start from subplot 2
        plt.imshow(filtered_images[D0], cmap='gray', vmin=0, vmax=255)
        plt.title(f'{subplot_labels[i]}) BLPF (D0={D0}, n={n_order})')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # --- Save Visualization and Individual Images ---
    # Save the combined visualization figure
    combined_output_path = os.path.join(output_dir, f"{base_name}_blpf_results_combined_n{n_order}.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()  # Close the plot to free memory

    # Save individual filtered images
    for D0, img in filtered_images.items():
        filtered_image_path = os.path.join(output_dir, f"{base_name}_blpf_D0_{D0}_n{n_order}.tif")
        imageio.imwrite(filtered_image_path, img)
        print(f"BLPF filtered image (D0={D0}, n={n_order}) saved to: {filtered_image_path}")

    print("\nImage processing complete.")

