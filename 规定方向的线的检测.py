import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image


def apply_45_degree_kernel(img_arr):
    img_float = img_arr.astype(float)
    kernel = np.array([[2, -1, -1],
                       [-1, 2, -1],
                       [-1, -1, 2]], dtype=float)

    response = ndimage.convolve(img_float, kernel)
    return response


def process_and_visualize_six_figures(image_path, output_dir):
    pil_img = Image.open(image_path).convert('L')
    img = np.array(pil_img)
    raw_response = apply_45_degree_kernel(img)

    max_abs = np.max(np.abs(raw_response))
    if max_abs == 0: max_abs = 1
    display_b = ((raw_response / max_abs) * 127.5 + 127.5).astype(np.uint8)

    slice_c = (slice(10, 110), slice(10, 110))
    img_c = display_b[slice_c]

    slice_d = (slice(370, 480), slice(370, 480))
    img_d = display_b[slice_d]


    response_e = raw_response.copy()
    response_e[response_e < 0] = 0
    display_e = np.clip(response_e, 0, 255).astype(np.uint8)

    max_val = np.max(response_e)
    if max_val > 0:
        T = max_val
        mask_f = response_e >= T
    else:
        mask_f = np.zeros_like(response_e, dtype=bool)

    img_f_binary = np.zeros_like(img)
    img_f_binary[mask_f] = 255

    structure = np.ones((5, 5), dtype=bool)
    img_f_dilated = ndimage.binary_dilation(img_f_binary > 0, structure=structure)
    img_f = img_f_dilated.astype(np.uint8) * 255

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('(a) Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(display_b, cmap='gray')
    axes[0, 1].set_title('(b) +45 Line Response')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(img_c, cmap='gray')
    axes[0, 2].set_title('(c) Zoomed Top-Left')
    axes[0, 2].axis('off')
    for spine in axes[0, 2].spines.values():
        spine.set_edgecolor('white')

    axes[1, 0].imshow(img_d, cmap='gray')
    axes[1, 0].set_title('(d) Zoomed Bottom-Right')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(display_e, cmap='gray')
    axes[1, 1].set_title('(e) Positive Response')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(img_f, cmap='gray')
    axes[1, 2].set_title('(f) Strongest Points (>T)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig1007_full_process.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    filename = "Fig1007(a)(wirebond_mask).tif"
    path = os.path.join(input_dir, filename)
    process_and_visualize_six_figures(path, output_dir)