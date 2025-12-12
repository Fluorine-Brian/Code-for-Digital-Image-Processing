import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image


def apply_average_filter(img_arr, kernel_size=5):
    img_float = img_arr.astype(float)
    return ndimage.uniform_filter(img_float, size=kernel_size)


def compute_sobel_magnitude(img_arr):
    # Normalize to 0-1 range
    img_float = img_arr.astype(float) / 255.0

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=float)

    gx = ndimage.convolve(img_float, kx)
    gy = ndimage.convolve(img_float, ky)

    return np.abs(gx) + np.abs(gy)


def threshold_image(magnitude, percentage=0.33):
    max_val = np.max(magnitude)
    T = max_val * percentage

    # Values >= 33% of max are set to White (255), others Black (0)
    binary_edges = (magnitude >= T).astype(np.uint8) * 255
    return binary_edges


def process_thresholded_edges(image_path, output_dir):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    pil_img = Image.open(image_path).convert('L')
    img = np.array(pil_img)

    # --- Path A: Original -> Sobel -> Threshold ---
    mag_original = compute_sobel_magnitude(img)
    edges_original = threshold_image(mag_original, percentage=0.33)

    # --- Path B: Smoothed -> Sobel -> Threshold ---
    # Smooth first (5x5 averaging filter)
    img_smoothed = apply_average_filter(img, kernel_size=5)
    mag_smoothed = compute_sobel_magnitude(img_smoothed)
    edges_smoothed = threshold_image(mag_smoothed, percentage=0.33)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Thresholded Gradient (Fig 10.20)', fontsize=14)

    # (a) Edges from Original Gradient
    axes[0].imshow(edges_original, cmap='gray')
    axes[0].set_title('(a) Thresholded Gradient of Original Image')
    axes[0].axis('off')

    # (b) Edges from Smoothed Gradient
    axes[1].imshow(edges_smoothed, cmap='gray')
    axes[1].set_title('(b) Thresholded Gradient of Smoothed Image')
    axes[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig1020_thresholded_edges.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Done. Result saved to {save_path}")


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    filename = "Fig1016(a)(building_original).tif"
    path = os.path.join(input_dir, filename)

    process_thresholded_edges(path, output_dir)