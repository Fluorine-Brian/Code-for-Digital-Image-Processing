import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image

MANUAL_INVERT = False


def fill_all_holes_robust(binary_image):
    background_mask = ~binary_image

    labels, num_features = ndimage.label(background_mask)

    if num_features == 0:
        return binary_image

    sizes = np.bincount(labels.ravel())

    main_background_label = np.argmax(sizes[1:]) + 1

    holes_mask = (labels != 0) & (labels != main_background_label)

    return binary_image | holes_mask


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    image_filename = "Fig0916(a)(region-filling-reflections).tif"
    image_path = os.path.join(input_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        exit()

    raw_image_pil = Image.open(image_path).convert('L')
    raw_image = np.array(raw_image_pil)

    binary_image = raw_image > 128

    if MANUAL_INVERT:
        binary_image = ~binary_image

    filled_image = fill_all_holes_robust(binary_image)

    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Morphological Hole Filling (Fig 9.18) - {image_filename}', fontsize=14)

    axes[0].imshow(binary_image.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('a) Original Image')
    axes[0].axis('off')

    axes[1].imshow(filled_image.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('b) Hole Filling Result')
    axes[1].axis('off')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_hole_filling_final.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Processing complete. Result saved to: {save_path}")