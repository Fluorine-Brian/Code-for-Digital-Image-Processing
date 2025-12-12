import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy import ndimage


def manual_threshold(img_arr, threshold=205):
    binary = np.zeros_like(img_arr)
    binary[img_arr > threshold] = 255
    return binary


def manual_erosion(binary_img, kernel_size=5):
    img_bool = (binary_img / 255).astype(float)
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    kernel_area = kernel_size * kernel_size

    neighbor_sum = ndimage.convolve(img_bool, kernel, mode='constant', cval=0.0)
    eroded_bool = neighbor_sum > (kernel_area - 0.5)

    return eroded_bool.astype(np.uint8) * 255


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)
    image_filename = "Fig0918(a)(Chickenfilet with bones).tif"
    image_path = os.path.join(input_dir, image_filename)
    pil_img = Image.open(image_path).convert('L')
    img = np.array(pil_img)

    binary = manual_threshold(img, threshold=205)
    eroded = manual_erosion(binary, kernel_size=5)
    display_b = 255 - binary
    display_c = eroded

    fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    fig.suptitle(f'Connected Components Extraction (Fig 9.20)', fontsize=14)

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('a) Original Image')
    axes[0].axis('off')

    axes[1].imshow(display_b, cmap='gray')
    axes[1].set_title('b) Thresholded (Negative)')
    axes[1].axis('off')

    axes[2].imshow(display_c, cmap='gray')
    axes[2].set_title('c) Eroded (Positive)')
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig0920_bone_analysis_compact.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()