import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image

# ==========================================
# Configuration
# ==========================================
# Use standard sans-serif font (Arial/Helvetica/DejaVu Sans)
# This avoids the need for Chinese font installation
plt.rcParams['font.family'] = 'sans-serif'


def perform_erosion(binary_image, structure):
    """ Performs Morphological Erosion """
    return ndimage.binary_erosion(binary_image, structure=structure)


def perform_dilation(binary_image, structure):
    """ Performs Morphological Dilation """
    return ndimage.binary_dilation(binary_image, structure=structure)


if __name__ == "__main__":
    # --- 1. Path Setup ---
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    image_filename = "Fig0911(a)(noisy_fingerprint).tif"
    image_path = os.path.join(input_dir, image_filename)

    # --- 2. Load Image ---
    try:
        # Load as grayscale ('L')
        raw_image_pil = Image.open(image_path).convert('L')
        raw_image = np.array(raw_image_pil)
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        exit()

    # --- 3. Binarization ---
    # Convert to Boolean (True/False).
    # Assuming standard fingerprint: Ridges=White(255), Background=Black(0)
    binary_image = raw_image > 128

    # --- 4. Define Structure Element ---
    # Standard 3x3 square of ones
    structure_size = 3
    structure = np.ones((structure_size, structure_size), dtype=bool)

    # Prepare original for display
    display_original = binary_image.astype(np.uint8) * 255

    # --- 5. Perform Operations (Fig 9.11 sequence) ---

    # c) Erosion of A
    # Formula: A ⊖ B
    image_c_eroded_bool = perform_erosion(binary_image, structure)
    image_c = image_c_eroded_bool.astype(np.uint8) * 255

    # d) Opening of A
    # Formula: A ○ B = (A ⊖ B) ⊕ B
    # Implementation: Dilate the result of step (c)
    image_d_opened_bool = perform_dilation(image_c_eroded_bool, structure)
    image_d = image_d_opened_bool.astype(np.uint8) * 255

    # e) Dilation of the Opening
    # Formula: (A ○ B) ⊕ B
    # Implementation: Dilate the result of step (d)
    image_e_dilated_again_bool = perform_dilation(image_d_opened_bool, structure)
    image_e = image_e_dilated_again_bool.astype(np.uint8) * 255

    # f) Closing of the Opening
    # Formula: (A ○ B) ● B = [ (A ○ B) ⊕ B ] ⊖ B
    # Implementation: Erode the result of step (e)
    image_f_closed_bool = perform_erosion(image_e_dilated_again_bool, structure)
    image_f = image_f_closed_bool.astype(np.uint8) * 255

    # --- 6. Visualization (English Labels) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Morphological Filtering (Fig 9.11) - {image_filename}', fontsize=16)

    # a) Noisy Image
    axes[0, 0].imshow(display_original, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(r'a) Noisy Image ($A$)')

    # c) Eroded Image
    axes[0, 1].imshow(image_c, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(r'c) Eroded Image ($A \ominus B$)')

    # d) Opening
    axes[0, 2].imshow(image_d, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(r'd) Opening ($A \circ B$)')

    # e) Dilation of the Opening
    axes[1, 0].imshow(image_e, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(r'e) Dilation of the Opening ($(A \circ B) \oplus B$)')

    # f) Closing of the Opening
    axes[1, 1].imshow(image_f, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(r'f) Closing of the Opening ($(A \circ B) \bullet B$)')

    # Hide unused subplot (bottom right)
    axes[1, 2].axis('off')

    # Hide axis ticks for cleaner look
    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    combined_output_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_morph_filter.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    plt.close()

    print(f"Processing complete. Result saved to: {combined_output_path}")