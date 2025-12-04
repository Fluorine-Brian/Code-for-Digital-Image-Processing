import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os


def tone_correction(image, contrast, midpoint):
    img_float = image.astype(np.float64) / 255.0
    epsilon = 1e-6
    corrected_float = 1 / (1 + (midpoint / (img_float + epsilon)) ** contrast)
    corrected_image = np.clip(corrected_float * 255, 0, 255).astype(np.uint8)
    return corrected_image


def plot_tone_curve(ax, contrast, midpoint):
    r = np.linspace(0, 1, 256)
    epsilon = 1e-6
    s = 1 / (1 + (midpoint / (r + epsilon)) ** contrast)

    ax.plot(r, s, 'k')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ticks = np.linspace(0, 1, 5)  # For 4 squares
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(['0', '', '', '', '1'])
    ax.set_yticklabels(['0', '', '', '', '1'])

    ax.grid(True, color='black', linewidth=0.75)  # Added grid and adjusted linewidth
    ax.tick_params(length=0)  # Remove tick marks

    ax.text(0.5, 0.2, 'R,G,B', ha='center', va='center',
            bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 4})


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    image_filename = "Fig0635(top_ left_flower).tif"
    image_path = os.path.join(input_dir, image_filename)
    original_image = imageio.imread(image_path)
    base_name = os.path.splitext(image_filename)[0]

    contrast_E = 10
    midpoint_m = 0.5

    corrected_image = tone_correction(original_image, contrast_E, midpoint_m)

    fig = plt.figure(figsize=(18, 7))  # Increased figure size
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 4, 1.5], wspace=0.05)  # Adjusted width_ratios and wspace
    fig.suptitle(f'Tone Correction (Fig 6.33) - {image_filename}', fontsize=16)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(original_image)
    ax0.set_title('a) Flat Image')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(corrected_image)
    ax1.set_title('b) Corrected Image')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[2])
    plot_tone_curve(ax2, contrast_E, midpoint_m)
    ax2.set_title('c) Transformation')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    combined_output_path = os.path.join(output_dir, f"{base_name}_tone_correction_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    plt.close()
