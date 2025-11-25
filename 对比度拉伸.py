import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


def contrast_stretch(input_image):
    """
    Performs contrast stretching on the input image
    """
    r_min = np.min(input_image)
    r_max = np.max(input_image)

    if r_max == r_min:
        return input_image.astype('uint8')

    input_image_float = input_image.astype(float)
    output_image = (input_image_float - r_min) * (255.0 / (r_max - r_min))
    return output_image.astype('uint8')


def threshold_processing(input_image, threshold_value):
    """
    Performs grayscale thresholding on the image
    """
    output_image = np.zeros_like(input_image)
    output_image[input_image >= threshold_value] = 255
    return output_image.astype('uint8')


if __name__ == "__main__":
    image_path = "Fig0310(b)(washed_out_pollen_image).tif"
    original_image = imageio.imread(image_path)

    stretched_image = contrast_stretch(original_image)
    m = int(np.mean(original_image))
    thresholded_image = threshold_processing(original_image, m)

    # Visualization
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'a) Original Image\n(Gray Range: [{np.min(original_image)}, {np.max(original_image)}])')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(stretched_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'c) Contrast Stretch Result\n(Gray Range: [{np.min(stretched_image)}, {np.max(stretched_image)}])')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(thresholded_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'd) Threshold Processing Result\n(Threshold m = {m})')
    plt.axis('off')

    plt.tight_layout()
    output_path = "Fig0310_combined_results.png"
    plt.savefig(output_path, bbox_inches='tight')

    imageio.imwrite("Fig0310_stretched.tif", stretched_image)
    imageio.imwrite("Fig0310_thresholded.tif", thresholded_image)
