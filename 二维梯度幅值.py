import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image


def compute_sobel_gradients(img_arr):
    img_float = img_arr.astype(float)

    # Kernel for |gx| (Detects Vertical Edges)
    # Corresponds to Fig 10.14(g) in some editions, used for Fig 10.16(b)
    # [-1, 0, 1]
    # [-2, 0, 2]
    # [-1, 0, 1]
    kx = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=float)

    # Kernel for |gy| (Detects Horizontal Edges)
    # Corresponds to Fig 10.14(f), used for Fig 10.16(c)
    # [-1, -2, -1]
    # [ 0,  0,  0]
    # [ 1,  2,  1]
    ky = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

    gx = ndimage.convolve(img_float, kx)
    gy = ndimage.convolve(img_float, ky)

    return np.abs(gx), np.abs(gy)


def process_sobel_edge_detection(image_path, output_dir):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    pil_img = Image.open(image_path).convert('L')
    img = np.array(pil_img)

    abs_gx, abs_gy = compute_sobel_gradients(img)

    magnitude = abs_gx + abs_gy

    display_gx = np.clip(abs_gx, 0, 255).astype(np.uint8)
    display_gy = np.clip(abs_gy, 0, 255).astype(np.uint8)
    display_mag = np.clip(magnitude, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # (a) Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('(a) Original Image')
    axes[0, 0].axis('off')

    # (b) |gx| (Vertical Edges)
    axes[0, 1].imshow(display_gx, cmap='gray')
    axes[0, 1].set_title('(b) |gx| component (Vertical Edges)')
    axes[0, 1].axis('off')

    # (c) |gy| (Horizontal Edges)
    axes[1, 0].imshow(display_gy, cmap='gray')
    axes[1, 0].set_title('(c) |gy| component (Horizontal Edges)')
    axes[1, 0].axis('off')

    # (d) |gx| + |gy|
    axes[1, 1].imshow(display_mag, cmap='gray')
    axes[1, 1].set_title('(d) Gradient image |gx| + |gy|')
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig1016_sobel_gradients.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Done. Result saved to {save_path}")


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    filename = "Fig1016(a)(building_original).tif"
    path = os.path.join(input_dir, filename)

    process_sobel_edge_detection(path, output_dir)