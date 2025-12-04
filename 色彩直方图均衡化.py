import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import warnings
import matplotlib.gridspec as gridspec


def rgb_to_hsi(rgb_image):
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


def histogram_equalization(intensity_channel):
    I_uint8 = np.clip(intensity_channel * 255, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(I_uint8.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    equalized_I_uint8 = cdf[I_uint8]
    equalized_I_float = equalized_I_uint8.astype(float) / 255.0
    equalized_hist, _ = np.histogram(equalized_I_uint8.flatten(), 256, [0, 256])
    transform_func = cdf / 255.0
    return equalized_I_float, hist, equalized_hist, transform_func


def plot_panel_b(fig, spec, hist_orig, hist_eq, transform_func, median_orig, median_eq):
    gs_nested = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, wspace=0.2, hspace=0.2)

    def style_ax(ax):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ticks = np.linspace(0, 1, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(['0', '', '', '', '1'])  # Added tick labels for grid
        ax.set_yticklabels(['0', '', '', '', '1'])  # Added tick labels for grid
        ax.grid(True, color='black', linewidth=0.75)
        ax.tick_params(length=0)

    ax_h = fig.add_subplot(gs_nested[0, 0])
    ax_h.plot([0, 1], [0, 1], 'k')
    ax_h.text(0.5, 0.5, 'H', ha='center', va='center', bbox={'facecolor': 'white', 'edgecolor': 'black'})
    style_ax(ax_h)

    ax_s = fig.add_subplot(gs_nested[0, 1])
    ax_s.plot([0, 1], [0, 1], 'k')
    ax_s.text(0.5, 0.5, 'S', ha='center', va='center', bbox={'facecolor': 'white', 'edgecolor': 'black'})
    style_ax(ax_s)

    ax_i = fig.add_subplot(gs_nested[1, 0])
    r = np.linspace(0, 1, 256)
    ax_i.plot(r, transform_func, 'k')
    ax_i.plot([median_orig, median_orig], [0, median_eq], 'k--')
    ax_i.plot([0, median_orig], [median_eq, median_eq], 'k--')
    ax_i.text(0.5, 0.5, 'I', ha='center', va='center', bbox={'facecolor': 'white', 'edgecolor': 'black'})
    style_ax(ax_i)
    # Specific labels for I plot
    ax_i.set_xticklabels(['0', '', '', '', '1'])
    ax_i.set_yticklabels(['0', '', '', '', '1'])
    ax_i.set_xlabel(f'{median_orig:.2f}', labelpad=-10)
    ax_i.set_ylabel(f'{median_eq:.2f}', labelpad=-10, rotation=0, ha='right')

    gs_hist = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_nested[1, 1], hspace=0.1)
    ax_hist1 = fig.add_subplot(gs_hist[0])
    ax_hist2 = fig.add_subplot(gs_hist[1])
    ax_hist1.bar(range(256), hist_orig, color='black', width=1.0)
    ax_hist1.set_title(f'Before (median={median_orig:.2f})', fontsize=8)
    ax_hist2.bar(range(256), hist_eq, color='black', width=1.0)
    ax_hist2.set_title(f'After (median={median_eq:.2f})', fontsize=8)
    for ax in [ax_hist1, ax_hist2]:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, 255)


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    image_filename = "Fig0637(a)(caster_stand_original).tif"
    image_path = os.path.join(input_dir, image_filename)
    original_image = imageio.imread(image_path)
    base_name = os.path.splitext(image_filename)[0]

    hsi_orig = rgb_to_hsi(original_image)
    H_orig, S_orig, I_orig = hsi_orig[:, :, 0], hsi_orig[:, :, 1], hsi_orig[:, :, 2]

    I_eq, hist_orig, hist_eq, transform_func = histogram_equalization(I_orig)
    hsi_c = np.stack([H_orig, S_orig, I_eq], axis=2)
    image_c = hsi_to_rgb(hsi_c)

    S_d_pre = S_orig ** 0.75
    I_d_pre = I_orig ** 0.75
    I_d_eq, _, _, _ = histogram_equalization(I_d_pre)
    hsi_d = np.stack([H_orig, S_d_pre, I_d_eq], axis=2)
    image_d = hsi_to_rgb(hsi_d)

    fig = plt.figure(figsize=(10, 10))
    # Increased hspace to create more vertical separation between rows
    gs_main = gridspec.GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.3)
    fig.suptitle(f'HSI Histogram Equalization (Fig 6.35) - {image_filename}', fontsize=16)

    ax_a = fig.add_subplot(gs_main[0, 0])
    ax_a.imshow(original_image)
    ax_a.set_title('a) Original Image')
    ax_a.axis('off')

    median_orig = np.median(I_orig)
    median_eq = np.median(I_eq)
    plot_panel_b(fig, gs_main[0, 1], hist_orig, hist_eq, transform_func, median_orig, median_eq)
    ax_b_title = fig.add_subplot(gs_main[0, 1])
    ax_b_title.set_title('b) Transformations and Histograms')
    ax_b_title.axis('off')

    ax_c = fig.add_subplot(gs_main[1, 0])
    ax_c.imshow(image_c)
    ax_c.set_title('c) Intensity Equalized')
    ax_c.axis('off')

    ax_d = fig.add_subplot(gs_main[1, 1])
    ax_d.imshow(image_d)
    ax_d.set_title('d) Sat. & Int. Adjusted, then Int. Equalized')
    ax_d.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    combined_output_path = os.path.join(output_dir, f"{base_name}_hsi_equalization_results.png")
    plt.savefig(combined_output_path, bbox_inches='tight')
    plt.close()
