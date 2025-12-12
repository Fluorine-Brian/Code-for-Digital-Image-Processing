import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image  # 使用 Pillow 替代 imageio

# ==========================================
# Config
# ==========================================
MANUAL_INVERT = False


def extract_boundary(binary_image, structure):
    # Formula: Beta(A) = A - (A eroded by B)
    eroded = ndimage.binary_erosion(binary_image, structure=structure)
    # Use logical XOR for set difference (A - Subset_of_A)
    # A ^ B is equivalent to A - B when B is a subset of A
    return binary_image ^ eroded


if __name__ == "__main__":
    # --- 1. Path Setup ---
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    # Fig 9.14 in 3rd Ed / Fig 9.16 in 4th Ed
    image_filename = "Fig0914(a)(licoln from penny).tif"
    image_path = os.path.join(input_dir, image_filename)

    # --- 2. Load Image (Using PIL to fix decompression error) ---
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        exit()

    try:
        # .convert('L') ensures grayscale
        raw_image_pil = Image.open(image_path).convert('L')
        raw_image = np.array(raw_image_pil)
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    # --- 3. Binarization ---
    # Assume Object=White(255), Background=Black(0)
    binary_image = raw_image > 128

    if MANUAL_INVERT:
        binary_image = ~binary_image

    # --- 4. Define Structuring Element ---
    # 3x3 Square of ones
    structure = np.ones((3, 3), dtype=bool)

    # --- 5. Perform Boundary Extraction ---
    boundary_image = extract_boundary(binary_image, structure)

    # --- 6. Visualization ---
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Boundary Extraction (Fig 9.16) - {image_filename}', fontsize=14)

    # Original
    axes[0].imshow(binary_image.astype(np.uint8) * 255, cmap='gray')
    axes[0].set_title('a) Original Image')
    axes[0].axis('off')

    # Result
    axes[1].imshow(boundary_image.astype(np.uint8) * 255, cmap='gray')
    axes[1].set_title('b) Boundary Extraction')
    axes[1].axis('off')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_boundary.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Processing complete. Result saved to: {save_path}")

