import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image


# def apply_average_filter(img_arr, kernel_size=5):
#     img_float = img_arr.astype(float)
#     smoothed = ndimage.uniform_filter(img_float, size=kernel_size)
#     return smoothed


def compute_kirsch_responses(img_arr):
    # Normalize to 0-1 range
    img_float = img_arr.astype(float) / 255.0

    # NW Kernel (45 degree) from Fig 10.15
    k_nw = np.array([[-3, 5, 5],
                     [-3, 0, 5],
                     [-3, -3, -3]], dtype=float)

    # SW Kernel (-45 degree) from Fig 10.15
    k_sw = np.array([[5, 5, -3],
                     [5, 0, -3],
                     [-3, -3, -3]], dtype=float)

    # Convolve
    resp_nw = ndimage.convolve(img_float, k_nw)
    resp_sw = ndimage.convolve(img_float, k_sw)

    # Return absolute values
    return np.abs(resp_nw), np.abs(resp_sw)


def normalize_for_display(img_data):
    min_val = np.min(img_data)
    max_val = np.max(img_data)
    if max_val - min_val == 0:
        return np.zeros_like(img_data, dtype=np.uint8)

    norm_img = (img_data - min_val) / (max_val - min_val) * 255
    return norm_img.astype(np.uint8)


def process_diagonal_edge_detection(image_path, output_dir):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    pil_img = Image.open(image_path).convert('L')
    img = np.array(pil_img)

    # 1. Apply Smoothing (Input is Fig 10.18(a))
    img_smoothed = img

    # 2. Compute Kirsch Responses
    abs_nw, abs_sw = compute_kirsch_responses(img_smoothed)

    # 3. Normalize
    display_nw = normalize_for_display(abs_nw)
    display_sw = normalize_for_display(abs_sw)

    # 4. Visualization

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Diagonal Edge Detection (Fig 10.19)', fontsize=14)

    # (a) NW Kernel Response
    axes[0].imshow(display_nw, cmap='gray')
    axes[0].set_title('(a) NW Kernel (45 degree)')
    axes[0].axis('off')

    # (b) SW Kernel Response
    axes[1].imshow(display_sw, cmap='gray')
    axes[1].set_title('(b) SW Kernel (-45 degree)')
    axes[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig1019_diagonal_edges.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Done. Result saved to {save_path}")


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    filename = "Fig1016(a)(building_original).tif"
    path = os.path.join(input_dir, filename)

    process_diagonal_edge_detection(path, output_dir)