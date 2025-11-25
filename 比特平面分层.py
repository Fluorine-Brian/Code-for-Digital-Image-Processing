import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for saving plots without displaying a window
import matplotlib.pyplot as plt
import os


def to_grayscale(image):
    """
    Converts an image to grayscale if it's not already grayscale.
    Handles 2D (grayscale), 3D (RGB), and 4D (RGBA) images.
    """
    if len(image.shape) == 2:  # Already 2D grayscale (H, W)
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 1:  # 3D grayscale (H, W, 1)
            return image.squeeze(axis=2)  # Remove the channel dimension
        elif image.shape[2] == 3:  # RGB image (H, W, 3)
            # Convert RGB to grayscale using luminance method
            # L = 0.2989 * R + 0.5870 * G + 0.1140 * B
            return (0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]).astype(image.dtype)
        elif image.shape[2] == 4:  # RGBA image (H, W, 4)
            # Discard alpha channel and convert RGB part to grayscale
            return (0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]).astype(image.dtype)
        else:
            raise ValueError(f"Unsupported 3D image format with {image.shape[2]} channels.")
    else:
        raise ValueError(f"Unsupported image format with {len(image.shape)} dimensions. Expected 2D or 3D.")


def bit_plane_slice(input_image, bit_position):
    """
    Extracts a specific bit plane from an 8-bit grayscale image.
    The output is a binary image (0 or 255) representing the specified bit plane.

    Args:
        input_image (np.array): An 8-bit grayscale image (pixel values 0-255).
        bit_position (int): The bit position to extract (0 for LSB, 7 for MSB).

    Returns:
        np.array: A binary image (uint8, 0 or 255) representing the bit plane.
    """
    bit_plane = ((input_image >> bit_position) & 1) * 255
    return bit_plane.astype('uint8')


if __name__ == "__main__":
    # Define image path and output directory
    image_path = "./original_image/Fig0314.png"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read the image
    try:
        original_image_raw = imageio.imread(image_path)
        original_image = to_grayscale(original_image_raw)
        if original_image.dtype != np.uint8:
            if np.max(original_image) > 255:
                original_image = (original_image / np.max(original_image) * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)
        print(f"Successfully loaded and converted image to 8-bit grayscale: {image_path}")
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        print("Please ensure the image file is in the same directory or provide the full path.")
        print("Creating a random 8-bit grayscale image for demonstration.")
        original_image = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    except ValueError as e:
        print(f"Error processing image: {e}")
        print("Creating a random 8-bit grayscale image for demonstration.")
        original_image = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)

    # 2. Perform bit-plane slicing for all 8 planes (from LSB 0 to MSB 7)
    bit_planes = []
    for i in range(8):
        plane = bit_plane_slice(original_image, i)
        bit_planes.append(plane)
        print(f"Generated bit-plane {i}")

    # --- Visualization (Adjusted to match textbook order: MSB first) ---
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Image')
    plt.axis('off')

    # Display bit planes from MSB (7) down to LSB (0)
    # The textbook shows 8 bit planes (b to i).
    # We will map:
    # subplot 2 -> Bit-plane 7 (MSB)
    # subplot 3 -> Bit-plane 6
    # subplot 4 -> Bit-plane 5
    # subplot 5 -> Bit-plane 4
    # subplot 6 -> Bit-plane 3
    # subplot 7 -> Bit-plane 2
    # subplot 8 -> Bit-plane 1
    # subplot 9 -> Bit-plane 0 (LSB)

    for i in range(8):
        # Calculate the bit plane index for display (7-i for descending order)
        bit_plane_idx_for_display = 7 - i
        # Subplot index starts from 2 (after the original image)
        plt.subplot(3, 3, i + 2)
        plt.imshow(bit_planes[bit_plane_idx_for_display], cmap='gray', vmin=0, vmax=255)
        plt.title(f'Bit-plane {bit_plane_idx_for_display}')
        plt.axis('off')

    plt.tight_layout()

    # --- Save Visualization and Individual Images ---
    combined_output_path = os.path.join(output_dir, "Fig0314_bit_planes_combined_MSB_first.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined visualization (MSB first) saved to: {combined_output_path}")

    # Saving individual planes can still be in ascending order, or you can reverse this loop too.
    # For consistency with the combined plot, let's save them with MSB first naming.
    for i in range(8):
        bit_plane_idx_for_save = 7 - i
        individual_output_path = os.path.join(output_dir, f"Fig0314_bit_plane_{bit_plane_idx_for_save}_MSB_first.png")
        imageio.imwrite(individual_output_path, bit_planes[bit_plane_idx_for_save])
        print(f"Bit-plane {bit_plane_idx_for_save} saved to: {individual_output_path}")

