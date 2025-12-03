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


def create_gaussian_lowpass_filter(shape, D0):
    """Creates a Gaussian Lowpass Filter in the frequency domain"""
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


def create_gaussian_highpass_filter(shape, D0):
    """Creates a Gaussian Highpass Filter in the frequency domain"""
    H_lp = create_gaussian_lowpass_filter(shape, D0)
    return 1 - H_lp


def create_high_frequency_emphasis_filter(shape, D0, k1, k2):
    """
    Creates a high-frequency emphasis filter based on GHPF
    """
    H_ghpf = create_gaussian_highpass_filter(shape, D0)
    H_emphasis = k1 + k2 * H_ghpf
    return H_emphasis


def histogram_equalization(image):
    """Performs histogram equalization on an 8-bit grayscale image"""
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_min = cdf.min()
    cdf_max = cdf.max()
    if cdf_max == cdf_min:
        equalized_image = np.full_like(image, 127, dtype=np.uint8)
    else:
        transform_func = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
        transform_func = transform_func.astype('uint8')
        equalized_image = transform_func[image]
    return equalized_image


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)
    image_filename = "Fig0459(a)(orig_chest_xray).tif"
    image_path = os.path.join(input_dir, image_filename)
    base_name = os.path.splitext(image_filename)[0]
    original_image = imageio.imread(image_path)
    original_M, original_N = original_image.shape
    padded_image, _ = pad_image_for_dft(original_image, padding_mode='reflect')
    P, Q = padded_image.shape
    dft_original = np.fft.fft2(padded_image.astype(float))
    centered_dft = np.fft.fftshift(dft_original)

    D0_ghpf = 70
    D0_hfe = 70
    k1_hfe = 0.5
    k2_hfe = 0.75
    H_ghpf = create_gaussian_highpass_filter((P, Q), D0_ghpf)
    filtered_dft_ghpf = centered_dft * H_ghpf
    idft_shifted_ghpf = np.fft.ifftshift(filtered_dft_ghpf)
    ghpf_image_complex = np.fft.ifft2(idft_shifted_ghpf)
    ghpf_image = np.real(ghpf_image_complex)[0:original_M, 0:original_N]

    min_val = np.min(ghpf_image)
    max_val = np.max(ghpf_image)
    if max_val > min_val:
        ghpf_image_display = ((ghpf_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        ghpf_image_display = np.zeros_like(ghpf_image, dtype=np.uint8)

    H_hfe = create_high_frequency_emphasis_filter((P, Q), D0_hfe, k1_hfe, k2_hfe)
    filtered_dft_hfe = centered_dft * H_hfe
    idft_shifted_hfe = np.fft.ifftshift(filtered_dft_hfe)
    hfe_image_complex = np.fft.ifft2(idft_shifted_hfe)
    hfe_image = np.real(hfe_image_complex)[0:original_M, 0:original_N]
    hfe_image = np.clip(hfe_image, 0, 255).astype(np.uint8)
    hfe_equalized_image = histogram_equalization(hfe_image)

    plt.figure(figsize=(12, 12))
    plt.suptitle(f'High-Frequency Emphasis Filtering (Fig 4.57) - {image_filename}', fontsize=16)

    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Chest X-Ray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(ghpf_image_display, cmap='gray', vmin=0, vmax=255)
    plt.title(f'b) GHPF Filtered (D0={D0_ghpf})')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(hfe_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'c) High-Frequency Emphasis (D0={D0_hfe}, k1={k1_hfe}, k2={k2_hfe})')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(hfe_equalized_image, cmap='gray', vmin=0, vmax=255)
    plt.title('d) HFE Result Equalized')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_output_path = os.path.join(output_dir, f"{base_name}_hfe_enhancement_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    plt.close()
