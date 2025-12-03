import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os


def pad_image_for_dft(image, padding_mode='reflect'):
    """Pads the image to a size suitable for DFT (e.g., 2*M x 2*N)"""
    M, N = image.shape
    P, Q = 2 * M, 2 * N
    pad_h = P - M
    pad_w = Q - N
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode=padding_mode)
    return padded_image, (M, N)


def create_ideal_highpass_filter(shape, D0):
    """Creates an Ideal Highpass Filter (IHPF) in the frequency domain"""
    P, Q = shape
    H = np.ones((P, Q), dtype=float)
    center_u, center_v = P / 2, Q / 2
    u_coords = np.arange(P) - center_u
    v_coords = np.arange(Q) - center_v
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    D_uv = np.sqrt(U ** 2 + V ** 2)
    H[D_uv <= D0] = 0
    return H


def create_gaussian_highpass_filter(shape, D0):
    """Creates a Gaussian Highpass Filter (GHPF) in the frequency domain"""
    P, Q = shape
    H = np.zeros((P, Q), dtype=float)
    center_u, center_v = P / 2, Q / 2
    u_coords = np.arange(P) - center_u
    v_coords = np.arange(Q) - center_v
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    D_uv_squared = U ** 2 + V ** 2
    if D0 == 0:
        return np.ones(shape, dtype=float)
    else:
        H_lp = np.exp(-D_uv_squared / (2 * D0 ** 2))
        return 1 - H_lp


def create_butterworth_highpass_filter(shape, D0, n):
    """Creates a Butterworth Highpass Filter (BHPF) in the frequency domain"""
    P, Q = shape
    H = np.zeros((P, Q), dtype=float)
    center_u, center_v = P / 2, Q / 2
    u_coords = np.arange(P) - center_u
    v_coords = np.arange(Q) - center_v
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    D_uv = np.sqrt(U ** 2 + V ** 2)
    if D0 == 0:
        return np.ones(shape, dtype=float)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            H_lp = 1 / (1 + (D_uv / D0) ** (2 * n))
        H_lp[np.isinf(H_lp)] = 0
        H_lp[int(center_u), int(center_v)] = 1.0
        return 1 - H_lp


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)
    image_filename = "characterTestPattern688.tif"
    image_path = os.path.join(input_dir, image_filename)
    base_name = os.path.splitext(image_filename)[0]

    original_image = imageio.imread(image_path)
    original_M, original_N = original_image.shape
    padded_image, _ = pad_image_for_dft(original_image, padding_mode='reflect')
    P, Q = padded_image.shape
    dft_original = np.fft.fft2(padded_image.astype(float))
    centered_dft = np.fft.fftshift(dft_original)
    D0_values = [60, 160]
    n_order = 2
    filtered_images = []
    titles = []

    for D0 in D0_values:
        H_ihpf = create_ideal_highpass_filter((P, Q), D0)
        filtered_dft_ihpf = centered_dft * H_ihpf
        idft_shifted_ihpf = np.fft.ifftshift(filtered_dft_ihpf)
        filtered_image_complex_ihpf = np.fft.ifft2(idft_shifted_ihpf)
        filtered_image_ihpf = np.real(filtered_image_complex_ihpf)[0:original_M, 0:original_N]
        filtered_images.append(np.clip(filtered_image_ihpf, 0, 255).astype(np.uint8))
        titles.append(f'IHPF (D0={D0})')

        H_ghpf = create_gaussian_highpass_filter((P, Q), D0)
        filtered_dft_ghpf = centered_dft * H_ghpf
        idft_shifted_ghpf = np.fft.ifftshift(filtered_dft_ghpf)
        filtered_image_complex_ghpf = np.fft.ifft2(idft_shifted_ghpf)
        filtered_image_ghpf = np.real(filtered_image_complex_ghpf)[0:original_M, 0:original_N]
        filtered_images.append(np.clip(filtered_image_ghpf, 0, 255).astype(np.uint8))
        titles.append(f'GHPF (D0={D0})')

        H_bhpf = create_butterworth_highpass_filter((P, Q), D0, n_order)
        filtered_dft_bhpf = centered_dft * H_bhpf
        idft_shifted_bhpf = np.fft.ifftshift(filtered_dft_bhpf)
        filtered_image_complex_bhpf = np.fft.ifft2(idft_shifted_bhpf)
        filtered_image_bhpf = np.real(filtered_image_complex_bhpf)[0:original_M, 0:original_N]
        filtered_images.append(np.clip(filtered_image_bhpf, 0, 255).astype(np.uint8))
        titles.append(f'BHPF (D0={D0}, n={n_order})')

    plt.figure(figsize=(18, 12))
    plt.suptitle(f'Highpass Filtering Comparison (Fig 4.53) - {image_filename}', fontsize=16)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(filtered_images[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_output_path = os.path.join(output_dir, f"{base_name}_highpass_filter_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    plt.close()
