import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os


def pad_image_for_dft(image, padding_mode='reflect'):
    """Pads the image to a size suitable for DFT (e.g., 2*M x 2*N)."""
    M, N = image.shape
    P, Q = 2 * M, 2 * N
    pad_h = P - M
    pad_w = Q - N
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode=padding_mode)
    return padded_image, (M, N)


def create_gaussian_lowpass_filter(shape, D0):
    """Creates a Gaussian Lowpass Filter (GLPF) in the frequency domain."""
    P, Q = shape
    H = np.zeros((P, Q), dtype=float)
    center_u, center_v = P / 2, Q / 2
    u_coords = np.arange(P) - center_u
    v_coords = np.arange(Q) - center_v
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    D_uv_squared = U ** 2 + V ** 2
    if D0 == 0:
        H = np.ones(shape, dtype=float)
    else:
        H = np.exp(-D_uv_squared / (2 * D0 ** 2))
    return H


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"  # Output directory for this task
    os.makedirs(output_dir, exist_ok=True)  # Corrected os.makedirs usage

    image_filename = "Fig0451(a)(satellite_original).tif"  # Image filename for this task
    image_path = os.path.join(input_dir, image_filename)
    base_name = os.path.splitext(image_filename)[0]

    print(f"Processing image: {image_filename}")

    original_image = imageio.imread(image_path)  # Read image (assuming grayscale)
    original_M, original_N = original_image.shape

    padded_image, _ = pad_image_for_dft(original_image, padding_mode='reflect')
    P, Q = padded_image.shape
    print(f"Image padded from {original_M}x{original_N} to {P}x{Q} using 'reflect' mode.")

    dft_original = np.fft.fft2(padded_image.astype(float))
    centered_dft = np.fft.fftshift(dft_original)
    print("Computed and centered DFT.")

    # D0 values as per textbook Fig 4.50: D0=50 for (b), D0=20 for (c)
    D0_values = [50, 20]
    filtered_images = {}

    for D0 in D0_values:
        print(f"Applying Gaussian Lowpass Filter with D0 = {D0}...")
        H = create_gaussian_lowpass_filter((P, Q), D0)
        filtered_dft = centered_dft * H
        idft_shifted = np.fft.ifftshift(filtered_dft)
        filtered_image_complex = np.fft.ifft2(idft_shifted)
        filtered_image = np.real(filtered_image_complex)[0:original_M, 0:original_N]
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        filtered_images[D0] = filtered_image
        print(f"Finished GLPF with D0={D0}.")

    # --- Visualization (Matching Fig 4.50(a) to (c)) ---
    plt.figure(figsize=(18, 6))
    plt.suptitle(f'GLPF for Satellite Image (Fig 4.50) - {image_filename}', fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Satellite Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(filtered_images[50], cmap='gray', vmin=0, vmax=255)
    plt.title(f'b) GLPF Filtered (D0=50)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(filtered_images[20], cmap='gray', vmin=0, vmax=255)
    plt.title(f'c) GLPF Filtered (D0=20)')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_output_path = os.path.join(output_dir, f"{base_name}_glpf_satellite_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()

    # Save individual filtered images
    for D0, img in filtered_images.items():
        filtered_image_path = os.path.join(output_dir, f"{base_name}_glpf_D0_{D0}.tif")
        imageio.imwrite(filtered_image_path, img)
        print(f"GLPF filtered image (D0={D0}) saved to: {filtered_image_path}")

    print("\nImage processing complete.")
