import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
from scipy import ndimage


def vector_gradient(image):
    img_float = image.astype(np.float64)
    g_xx = np.zeros_like(img_float[:, :, 0])
    g_yy = np.zeros_like(img_float[:, :, 0])

    for i in range(3):
        channel = img_float[:, :, i]
        g_x = ndimage.sobel(channel, axis=1)
        g_y = ndimage.sobel(channel, axis=0)
        g_xx += g_x ** 2
        g_yy += g_y ** 2

    M = np.sqrt(g_xx + g_yy)
    return M


def sum_of_gradients(image):
    img_float = image.astype(np.float64)
    M_sum = np.zeros_like(img_float[:, :, 0])

    for i in range(3):
        channel = img_float[:, :, i]
        g_x = ndimage.sobel(channel, axis=1)
        g_y = ndimage.sobel(channel, axis=0)
        M_sum += np.sqrt(g_x ** 2 + g_y ** 2)

    return M_sum


def scale_to_uint8(image_float):
    max_val = np.max(image_float)
    if max_val > 0:
        scaled_image = (image_float / max_val) * 255
    else:
        scaled_image = image_float
    return scaled_image.astype(np.uint8)


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    image_filename = "Fig0646(a)(lenna_original_RGB).tif"
    image_path = os.path.join(input_dir, image_filename)
    original_image = imageio.imread(image_path)
    base_name = os.path.splitext(image_filename)[0]

    grad_vector = vector_gradient(original_image)
    grad_sum = sum_of_gradients(original_image)
    grad_diff = np.abs(grad_vector - grad_sum)

    display_vector = scale_to_uint8(grad_vector)
    display_sum = scale_to_uint8(grad_sum)
    display_diff = scale_to_uint8(grad_diff)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'RGB Edge Detection (Fig 6.44) - {image_filename}', fontsize=16)

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('a) Original RGB Image')

    axes[0, 1].imshow(display_vector, cmap='gray')
    axes[0, 1].set_title('b) Vector Gradient')

    axes[1, 0].imshow(display_sum, cmap='gray')
    axes[1, 0].set_title('c) Sum of Individual Gradients')

    axes[1, 1].imshow(display_diff, cmap='gray')
    axes[1, 1].set_title('d) Difference (b) - (c)')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    combined_output_path = os.path.join(output_dir, f"{base_name}_rgb_edge_detection_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    plt.close()
