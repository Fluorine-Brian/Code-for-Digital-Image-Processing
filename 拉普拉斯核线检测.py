import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image


def process_laplacian_line_detection(image_path, output_dir):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    pil_img = Image.open(image_path).convert('L')
    img = np.array(pil_img)

    if np.mean(img) > 128:
        img = 255 - img

    img_float = img.astype(float)

    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], dtype=float)

    laplacian = ndimage.convolve(img_float, kernel)

    img_b = np.clip(laplacian + 128, 0, 255)
    img_c = np.abs(laplacian)
    img_d = np.clip(laplacian, 0, None)

    zoom_slice = (slice(125, 175), slice(220, 270))
    img_b_zoom = img_b[zoom_slice]

    fig = plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('a) Original Image')
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(img_b, cmap='gray')
    ax2.set_title('b) Laplacian Image')
    ax2.axis('off')

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(img_b_zoom, cmap='gray', interpolation='nearest')
    ax3.set_title('b) Zoomed Detail')
    ax3.axis('off')
    for spine in ax3.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)

    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(img_c, cmap='gray')
    ax4.set_title('c) Absolute Value')
    ax4.axis('off')

    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(img_d, cmap='gray')
    ax5.set_title('d) Positive Values')
    ax5.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig1005_line_detection.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Done. Result saved to {save_path}")


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    filename = "Fig0905(a)(wirebond-mask).tif"
    path = os.path.join(input_dir, filename)

    process_laplacian_line_detection(path, output_dir)