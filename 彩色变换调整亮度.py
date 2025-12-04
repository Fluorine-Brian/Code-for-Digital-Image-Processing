import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec


def adjust_intensity_rgb(image, k):
    img_float = image.astype(float) / 255.0
    processed = img_float * k
    processed = np.clip(processed, 0, 1)
    return (processed * 255).astype(np.uint8)


def plot_mapping_function(ax, slope=None, intercept=None, label_text="", k_val=0.7, subplot_label=""):
    x = np.linspace(0, 1, 100)

    if slope is not None and intercept is not None:
        y = slope * x + intercept
    elif slope is not None:
        y = slope * x
    else:
        y = x

    y = np.clip(y, 0, 1)

    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))

    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.grid(True, linestyle='-', alpha=0.8, color='gray', which='major')
    ax.set_aspect('equal')

    if k_val is not None:
        if "R,G,B" in label_text or label_text == "I":
            ax.plot([0, 1], [k_val, k_val], 'k--', alpha=0.3)
            ax.text(-0.1, k_val, r'$k$', va='center', ha='right', fontsize=12)
        elif "C,M,Y" in label_text and intercept is not None and intercept > 0:
            val = 1 - k_val
            ax.plot([0, 1], [val, val], 'k--', alpha=0.3)
            ax.text(-0.1, val, r'$1-k$', va='center', ha='right', fontsize=12)

    ax.set_xticklabels([0, '', '', '', 1])
    ax.set_yticklabels([0, '', '', '', 1])

    ax.tick_params(axis='y', pad=5)

    ax.text(0.95, 0.05, label_text, transform=ax.transAxes,
            ha='right', va='bottom', color='white', backgroundcolor='black', fontsize=9, fontweight='bold')

    ax.text(-0.25, 1.1, subplot_label, transform=ax.transAxes,
            ha='left', va='top', fontsize=12, fontweight='bold')


def plot_image_subplot(ax, image_data, subplot_label=""):
    ax.imshow(image_data)
    ax.axis('off')
    ax.text(-0.05, 1.05, subplot_label, transform=ax.transAxes,
            ha='left', va='top', fontsize=12, fontweight='bold')


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    image_filename = "Fig0630(01)(strawberries_fullcolor).tif"

    if not os.path.exists(os.path.join(input_dir, image_filename)):
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if files:
            image_filename = files[0]
        else:
            exit()

    image_path = os.path.join(input_dir, image_filename)
    base_name = os.path.splitext(image_filename)[0]

    original_image = imageio.imread(image_path)
    if len(original_image.shape) == 2:
        original_image = np.stack((original_image,) * 3, axis=-1)
    elif original_image.shape[2] == 4:
        original_image = original_image[:, :, :3]

    k = 0.7
    result_image = adjust_intensity_rgb(original_image, k)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[2.5, 1])

    ax_img1 = fig.add_subplot(gs[0, 0:3])
    plot_image_subplot(ax_img1, original_image, subplot_label="a")

    ax_img2 = fig.add_subplot(gs[0, 3:6])
    plot_image_subplot(ax_img2, result_image, subplot_label="b")

    ax_f1 = fig.add_subplot(gs[1, 0])
    plot_mapping_function(ax_f1, slope=k, intercept=0, label_text="R,G,B", k_val=k, subplot_label="c")

    ax_f2 = fig.add_subplot(gs[1, 1])
    plot_mapping_function(ax_f2, slope=k, intercept=1 - k, label_text="C,M,Y", k_val=k, subplot_label="d")

    ax_f3 = fig.add_subplot(gs[1, 2])
    plot_mapping_function(ax_f3, slope=1, intercept=0, label_text="K", k_val=k, subplot_label="e")

    ax_f4 = fig.add_subplot(gs[1, 3])
    plot_mapping_function(ax_f4, slope=k, intercept=1 - k, label_text="C,M,Y", k_val=k, subplot_label="f")

    ax_f5 = fig.add_subplot(gs[1, 4])
    plot_mapping_function(ax_f5, slope=k, intercept=0, label_text="I", k_val=k, subplot_label="g")

    ax_f6 = fig.add_subplot(gs[1, 5])
    plot_mapping_function(ax_f6, slope=1, intercept=0, label_text="H,S", k_val=k, subplot_label="h")

    save_path = os.path.join(output_dir, f"{base_name}_fig629_final.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)