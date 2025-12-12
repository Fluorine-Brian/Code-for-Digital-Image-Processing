import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image
import cv2


def apply_average_filter(img_arr, kernel_size=5):
    # Apply averaging filter (Box filter)
    # This blurs the image to remove fine details (like bricks)
    return cv2.blur(img_arr, (kernel_size, kernel_size))


def compute_sobel_gradients(img_arr):
    # Normalize to 0-1 range for consistent calculation
    img_float = img_arr.astype(float) / 255.0

    # Kernel for |gx| (Detects Vertical Edges)
    kx = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=float)

    # Kernel for |gy| (Detects Horizontal Edges)
    ky = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

    gx = ndimage.convolve(img_float, kx)
    gy = ndimage.convolve(img_float, ky)

    return np.abs(gx), np.abs(gy)


def normalize_for_display(img_data):
    # Robust normalization to match textbook contrast
    min_val = np.min(img_data)
    max_val = np.max(img_data)
    if max_val - min_val == 0:
        return np.zeros_like(img_data, dtype=np.uint8)

    norm_img = (img_data - min_val) / (max_val - min_val) * 255
    return norm_img.astype(np.uint8)


def process_smoothed_sobel(image_path, output_dir):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    pil_img = Image.open(image_path).convert('L')
    img = np.array(pil_img)

    # 1. Smoothing (Key difference from previous task)
    # Use a 5x5 averaging filter before edge detection
    img_smoothed = apply_average_filter(img, kernel_size=5)

    # 2. Compute Sobel gradients on the SMOOTHED image
    abs_gx, abs_gy = compute_sobel_gradients(img_smoothed)
    magnitude = abs_gx + abs_gy

    # 3. Normalize for display
    display_gx = normalize_for_display(abs_gx)
    display_gy = normalize_for_display(abs_gy)
    display_mag = normalize_for_display(magnitude)

    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 12))

    # (a) Smoothed Image
    axes[0, 0].imshow(img_smoothed, cmap='gray')
    axes[0, 0].set_title('(a) Smoothed Image (5x5 Avg)')
    axes[0, 0].axis('off')

    # (b) |gx| (Vertical Edges)
    axes[0, 1].imshow(display_gx, cmap='gray')
    axes[0, 1].set_title('(b) |gx| (Vertical)')
    axes[0, 1].axis('off')

    # (c) |gy| (Horizontal Edges)
    axes[1, 0].imshow(display_gy, cmap='gray')
    axes[1, 0].set_title('(c) |gy| (Horizontal)')
    axes[1, 0].axis('off')

    # (d) |gx| + |gy|
    axes[1, 1].imshow(display_mag, cmap='gray')
    axes[1, 1].set_title('(d) Gradient |gx| + |gy|')
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig1018_smoothed_sobel.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Done. Result saved to {save_path}")


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    filename = "Fig1016(a)(building_original).tif"
    path = os.path.join(input_dir, filename)

    process_smoothed_sobel(path, output_dir)