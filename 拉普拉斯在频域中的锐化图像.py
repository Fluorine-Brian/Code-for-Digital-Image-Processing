import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os


def pad_image_for_dft(image, padding_mode='reflect'):
    """Pads the image to a size suitable for DFT"""
    M, N = image.shape
    P, Q = 2 * M, 2 * N
    pad_h = P - M
    pad_w = Q - N
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode=padding_mode)
    return padded_image, (M, N)


def create_frequency_domain_laplacian(shape):
    """
    Creates a Laplacian filter in the frequency domain
    """
    P, Q = shape
    H = np.zeros((P, Q), dtype=float)

    center_u, center_v = P / 2, Q / 2
    u_coords = np.arange(P) - center_u
    v_coords = np.arange(Q) - center_v
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    D_uv_squared = U ** 2 + V ** 2

    H = -4 * (np.pi ** 2) * D_uv_squared
    return H


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)
    image_filename = "Fig0458(a)(blurry_moon).tif"
    image_path = os.path.join(input_dir, image_filename)
    base_name = os.path.splitext(image_filename)[0]
    original_image = imageio.imread(image_path)
    original_M, original_N = original_image.shape

    f_normalized = original_image.astype(float) / 255.0

    padded_image, _ = pad_image_for_dft(f_normalized, padding_mode='reflect')
    P, Q = padded_image.shape
    dft_original = np.fft.fft2(padded_image)
    centered_dft = np.fft.fftshift(dft_original)
    H_laplacian = create_frequency_domain_laplacian((P, Q))
    filtered_dft = centered_dft * H_laplacian
    idft_shifted = np.fft.ifftshift(filtered_dft)
    laplacian_response_complex = np.fft.ifft2(idft_shifted)
    laplacian_response_unscaled = np.real(laplacian_response_complex)[0:original_M, 0:original_N]
    max_abs_laplacian = np.max(np.abs(laplacian_response_unscaled))
    if max_abs_laplacian > 0:
        laplacian_normalized = laplacian_response_unscaled / max_abs_laplacian
    else:
        laplacian_normalized = np.zeros_like(laplacian_response_unscaled)
    g_normalized = f_normalized - laplacian_normalized
    sharpened_image = np.clip(g_normalized * 255, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Frequency Domain Laplacian Sharpening (Fig 4.56) - {image_filename}', fontsize=16)

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Blurry Moon Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_image, cmap='gray', vmin=0, vmax=255)
    plt.title('b) Frequency Domain Laplacian Sharpened')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_output_path = os.path.join(output_dir, f"{base_name}_freq_laplacian_sharpening_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_output_path}")
    plt.close()
