import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy import ndimage
MANUAL_INVERT = False


def perform_erosion(binary_image, size):
    """
    执行形态学腐蚀
    """
    structure = np.ones((size, size), dtype=bool)
    eroded_mask = ndimage.binary_erosion(binary_image, structure=structure)
    return eroded_mask.astype(np.uint8) * 255


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)
    image_filename = "Fig0905(a)(wirebond-mask).tif"
    image_path = os.path.join(input_dir, image_filename)
    raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    binary_image = raw_image > 127
    if MANUAL_INVERT:
        binary_image = ~binary_image

    foreground_pixels = np.sum(binary_image)
    total_pixels = binary_image.size
    ratio = foreground_pixels / total_pixels

    # 11x11
    image_b = perform_erosion(binary_image, 11)
    # 15x15
    image_c = perform_erosion(binary_image, 15)
    # 45x45
    image_d = perform_erosion(binary_image, 45)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Morphological Erosion (Fig 9.5) - {image_filename}', fontsize=16)

    display_original = binary_image.astype(np.uint8) * 255
    axes[0, 0].imshow(display_original, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('a) Original Image (White=Object)')

    axes[0, 1].imshow(image_b, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('b) Erosion with 11x11 SE')

    axes[1, 0].imshow(image_c, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('c) Erosion with 15x15 SE')

    axes[1, 1].imshow(image_d, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('d) Erosion with 45x45 SE')

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_final_fix.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()