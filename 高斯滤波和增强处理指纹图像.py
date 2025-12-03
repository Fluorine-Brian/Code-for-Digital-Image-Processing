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


def create_butterworth_highpass_filter(shape, D0, n):
    """Creates a Butterworth Highpass Filter in the frequency domain"""
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
    image_filename = "Fig0457(a)(thumb_print).tif"
    image_path = os.path.join(input_dir, image_filename)
    base_name = os.path.splitext(image_filename)[0]
    original_image = imageio.imread(image_path)
    original_M, original_N = original_image.shape
    padded_image, _ = pad_image_for_dft(original_image, padding_mode='reflect')
    P, Q = padded_image.shape
    dft_original = np.fft.fft2(padded_image.astype(float))
    centered_dft = np.fft.fftshift(dft_original)

    D0_value = 50
    n_order = 4
    H_bhpf = create_butterworth_highpass_filter((P, Q), D0_value, n_order)
    filtered_dft = centered_dft * H_bhpf
    idft_shifted = np.fft.ifftshift(filtered_dft)
    filtered_image_complex = np.fft.ifft2(idft_shifted)
    highpass_filtered_float = np.real(filtered_image_complex)[0:original_M, 0:original_N]
    highpass_filtered_display = np.clip(highpass_filtered_float, 0, 255).astype(np.uint8)
    thresholded_image = np.zeros_like(highpass_filtered_float, dtype=np.uint8)
    thresholded_image[highpass_filtered_float > 0] = 255

    plt.figure(figsize=(18, 6))
    plt.suptitle(f'Thumb Print Enhancement (Fig 4.55) - {image_filename}', fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('a) Original Thumb Print')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(highpass_filtered_display, cmap='gray', vmin=0, vmax=255)
    plt.title(f'b) Highpass Filtered (D0={D0_value}, n={n_order})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(thresholded_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'c) Thresholded Result')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_output_path = os.path.join(output_dir, f"{base_name}_enhancement_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    plt.close()