import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os


def color_slicing_cube(image, center_color, width, neutral_color=(0.5, 0.5, 0.5)):
    img_float = image.astype(np.float64) / 255.0
    center = np.array(center_color)

    diff = np.abs(img_float - center)
    mask = np.any(diff > width / 2.0, axis=2)

    result_image = np.copy(image)
    neutral_pixel = (np.array(neutral_color) * 255).astype(np.uint8)
    result_image[mask] = neutral_pixel

    return result_image


def color_slicing_sphere(image, center_color, radius, neutral_color=(0.5, 0.5, 0.5)):
    img_float = image.astype(np.float64) / 255.0
    center = np.array(center_color)

    distances_sq = np.sum((img_float - center) ** 2, axis=2)
    mask = distances_sq > radius ** 2

    result_image = np.copy(image)
    neutral_pixel = (np.array(neutral_color) * 255).astype(np.uint8)
    result_image[mask] = neutral_pixel

    return result_image


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    image_filename = "Fig0630(01)(strawberries_fullcolor).tif"
    image_path = os.path.join(input_dir, image_filename)
    original_image = imageio.imread(image_path)
    base_name = os.path.splitext(image_filename)[0]

    center_color_a = (0.6863, 0.1608, 0.1922)
    width_W = 0.2549
    radius_R0 = 0.1765

    cube_sliced_image = color_slicing_cube(original_image, center_color_a, width_W)
    sphere_sliced_image = color_slicing_sphere(original_image, center_color_a, radius_R0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Color Slicing (Fig 6.32) - {image_filename}', fontsize=16)

    axes[0].imshow(cube_sliced_image)
    axes[0].set_title(f'a) Cube transform, $W={width_W}$')
    axes[0].axis('off')

    axes[1].imshow(sphere_sliced_image)
    axes[1].set_title(f'b) Sphere transform, $R_0={radius_R0}$')
    axes[1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    combined_output_path = os.path.join(output_dir, f"{base_name}_color_slicing_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    plt.close()
