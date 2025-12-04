import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import warnings
import matplotlib.gridspec as gridspec


def rgb_complement(image):
    """
    Computes the complement of an RGB image
    """
    return 255 - image


def rgb_to_hsi(rgb_image):
    """
    Converts an RGB image to HSI color space
    """
    img_float = rgb_image.astype(float) / 255.0
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    H = np.zeros_like(R)
    S = np.zeros_like(R)
    I = (R + G + B) / 3.0
    min_rgb = np.minimum(np.minimum(R, G), B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = 1 - (3.0 / (R + G + B + 1e-6)) * min_rgb
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        theta = np.arccos(np.clip(num / (den + 1e-6), -1, 1))
    theta = np.degrees(theta)
    H[B <= G] = theta[B <= G]
    H[B > G] = 360 - theta[B > G]
    H[S == 0] = 0
    hsi_image = np.stack([H, S, I], axis=2)
    return hsi_image


def hsi_to_rgb(hsi_image):
    """
    Converts an HSI image back to RGB color space
    """
    H, S, I = hsi_image[:, :, 0], hsi_image[:, :, 1], hsi_image[:, :, 2]
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

    idx = (H >= 0) & (H < 120)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(np.radians(H[idx]))) / np.cos(np.radians(60 - H[idx])))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])
    idx = (H >= 120) & (H < 240)
    H_prime = H[idx] - 120
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(np.radians(H_prime))) / np.cos(np.radians(60 - H_prime)))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    idx = (H >= 240) & (H < 360)
    H_prime = H[idx] - 240
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(np.radians(H_prime))) / np.cos(np.radians(60 - H_prime)))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    rgb_image = np.stack([R, G, B], axis=2)
    rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

    return rgb_image


def hsi_transform_from_book(image):
    """
    Computes the complement based on the transformation functions
    """
    hsi_img = rgb_to_hsi(image)
    H, S, I = hsi_img[:, :, 0], hsi_img[:, :, 1], hsi_img[:, :, 2]

    H_transformed = (H + 180) % 360
    S_transformed = S
    I_transformed = 1.0 - I
    hsi_transformed = np.stack([H_transformed, S_transformed, I_transformed], axis=2)
    return hsi_to_rgb(hsi_transformed)


def plot_transform_functions(fig, spec):
    """
    Plots the four transformation functions
    """
    gs_nested = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, wspace=0.1, hspace=0.1)
    x = np.linspace(0, 1, 100)
    plots_def = {
        '00': {'type': 'rgb', 'label': 'R,G,B'},
        '01': {'type': 'h', 'label': 'H'},
        '10': {'type': 's', 'label': 'S'},
        '11': {'type': 'i', 'label': 'I'}
    }
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs_nested[i, j])
            key = f'{i}{j}'
            p_def = plots_def[key]
            if p_def['type'] == 'rgb' or p_def['type'] == 'i':
                ax.plot([0, 1], [1, 0], 'k')
            elif p_def['type'] == 'h':
                ax.plot([0, 0.5], [0.5, 1], 'k')
                ax.plot([0.5, 1], [0, 0.5], 'k')
            elif p_def['type'] == 's':
                ax.plot([0, 1], [0, 1], 'k')

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ticks = np.linspace(0, 1, 6)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(['0', '', '', '', '', '1'])
            ax.set_yticklabels(['0', '', '', '', '', '1'])
            ax.grid(True, color='black', linewidth=0.75)
            ax.tick_params(length=0)

            ax.text(0.25, 0.15, p_def['label'], ha='center', va='center',
                    bbox={'facecolor': 'darkgray', 'edgecolor': 'black', 'pad': 4})


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)
    image_filename = "Fig0630(01)(strawberries_fullcolor).tif"
    image_path = os.path.join(input_dir, image_filename)
    original_image = imageio.imread(image_path)
    base_name = os.path.splitext(image_filename)[0]
    rgb_comp_image = rgb_complement(original_image)
    hsi_comp_image = hsi_transform_from_book(original_image)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f'Color Complement Transformations (Fig 6.31) - {image_filename}', fontsize=16)
    gs_main = gridspec.GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.15)

    ax_a = fig.add_subplot(gs_main[0, 0])
    ax_a.imshow(original_image)
    ax_a.set_title('a) Original Image')
    ax_a.axis('off')

    plot_transform_functions(fig, gs_main[0, 1])
    ax_b_title = fig.add_subplot(gs_main[0, 1])
    ax_b_title.set_title('b) Transformation Functions')
    ax_b_title.axis('off')

    ax_c = fig.add_subplot(gs_main[1, 0])
    ax_c.imshow(rgb_comp_image)
    ax_c.set_title('c) RGB Complement')
    ax_c.axis('off')

    ax_d = fig.add_subplot(gs_main[1, 1])
    ax_d.imshow(hsi_comp_image)
    ax_d.set_title('d) HSI-based Complement')
    ax_d.axis('off')

    combined_output_path = os.path.join(output_dir, f"{base_name}_complement_full_figure.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    plt.close()

