import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image


# ==========================================
# 1. 基础工具函数 (手搓核心)
# ==========================================

def gaussian_kernel(size, sigma=1):
    """ 生成高斯核 """
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def sobel_filters_manual(img):
    """ 计算梯度幅值和方向 """
    # 标准 Sobel 算子
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)
    return G, theta


def non_max_suppression_manual(img, D):
    """ 非极大值抑制 """
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                # 0度 (垂直边缘)
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # 45度
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # 90度 (水平边缘)
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # 135度
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0
            except IndexError:
                pass
    return Z


def hysteresis_manual(img, weak, strong=255):
    """ 滞后边界跟踪 (栈实现DFS) """
    M, N = img.shape
    strong_rows, strong_cols = np.where(img == strong)
    stack = list(zip(strong_rows, strong_cols))

    while stack:
        i, j = stack.pop()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0: continue
                ni, nj = i + di, j + dj
                if 0 <= ni < M and 0 <= nj < N:
                    if img[ni, nj] == weak:
                        img[ni, nj] = strong
                        stack.append((ni, nj))

    # 清理弱边缘
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                img[i, j] = 0
    return img


# ==========================================
# 2. 具体任务实现函数
# ==========================================

def get_image_b_thresholded_gradient(img_norm):
    """
    图 10.26(b): 平滑 -> 梯度 -> 阈值 (15% Max)
    """
    # 1. 均值平滑 5x5
    img_smoothed = ndimage.uniform_filter(img_norm, size=5)

    # 2. Sobel 梯度
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    gx = ndimage.convolve(img_smoothed, Kx)
    gy = ndimage.convolve(img_smoothed, Ky)
    magnitude = np.hypot(gx, gy)

    # 3. 阈值处理 (15% of Max)
    # 教材说明: "阈值是该梯度图像极大值的 15%"
    thresh_val = 0.15 * np.max(magnitude)
    edges = (magnitude >= thresh_val).astype(np.uint8) * 255
    return edges


def get_image_c_marr_hildreth(img_norm):
    """
    图 10.26(c): LoG (sigma=3, n=19) -> 零交叉 (Thresh=0.002)
    """
    # 1. 计算 LoG
    # sigma=3, size=19 -> radius=9 -> truncate = 9/3 = 3.0
    log_resp = ndimage.gaussian_laplace(img_norm, sigma=3, truncate=3.0)

    # 2. 过零点检测
    # 教材说明: "所用的阈值为 0.002"
    # 这里的 0.002 是绝对值阈值 (针对 [0,1] 图像)
    threshold = 0.002

    edges = np.zeros_like(img_norm, dtype=np.uint8)

    # 简单的过零点检测逻辑
    # 水平
    curr_h = log_resp[:, :-1]
    right_h = log_resp[:, 1:]
    sign_diff_h = (curr_h * right_h) < 0
    mag_check_h = np.abs(curr_h - right_h) > threshold
    edges[:, :-1][sign_diff_h & mag_check_h] = 255

    # 垂直
    curr_v = log_resp[:-1, :]
    down_v = log_resp[1:, :]
    sign_diff_v = (curr_v * down_v) < 0
    mag_check_v = np.abs(curr_v - down_v) > threshold
    edges[:-1, :][sign_diff_v & mag_check_v] = 255

    return edges


def get_image_d_manual_canny(img_norm):
    """
    图 10.26(d): 手搓 Canny
    参数: sigma=2, size=13, TL=0.05, TH=0.15
    """
    # 1. 高斯平滑 (sigma=2, size=13)
    # 这里的 13x13 核对应 sigma=2 -> radius=6 -> truncate=3.0
    kernel = gaussian_kernel(size=13, sigma=2)
    img_smoothed = ndimage.convolve(img_norm, kernel)

    # 2. 计算梯度
    grad_mag, grad_dir = sobel_filters_manual(img_smoothed)

    # 关键：为了使用绝对阈值 0.05 和 0.15，我们需要将梯度幅值归一化到 [0, 1]
    # 因为教材给出的阈值不依赖于图像的具体灰度范围，通常是基于归一化梯度的
    grad_mag_norm = grad_mag / np.max(grad_mag) if np.max(grad_mag) > 0 else grad_mag

    # 3. 非极大值抑制
    # 注意：传入原始幅值或归一化幅值皆可，只要统一
    img_nms = non_max_suppression_manual(grad_mag_norm, grad_dir)

    # 4. 双阈值 (TL=0.05, TH=0.15)
    t_low = 0.05
    t_high = 0.15

    weak_val = np.int32(50)
    strong_val = np.int32(255)

    res = np.zeros_like(img_nms, dtype=np.int32)
    strong_i, strong_j = np.where(img_nms >= t_high)
    weak_i, weak_j = np.where((img_nms <= t_high) & (img_nms >= t_low))

    res[strong_i, strong_j] = strong_val
    res[weak_i, weak_j] = weak_val

    # 5. 滞后边界跟踪
    img_final = hysteresis_manual(res, weak_val, strong_val)

    return img_final


# ==========================================
# 主流程
# ==========================================

def process_head_ct(image_path, output_dir):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # 0. 加载并归一化图像 [0, 1]
    pil_img = Image.open(image_path).convert('L')
    img_norm = np.array(pil_img).astype(float) / 255.0

    # 1. 生成 (b) 阈值梯度
    print("Generating Image (b)...")
    img_b = get_image_b_thresholded_gradient(img_norm)

    # 2. 生成 (c) Marr-Hildreth
    print("Generating Image (c)...")
    img_c = get_image_c_marr_hildreth(img_norm)

    # 3. 生成 (d) 手搓 Canny
    print("Generating Image (d)...")
    img_d = get_image_d_manual_canny(img_norm)

    # 4. 可视化
    plt.rcParams['font.family'] = 'sans-serif'

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # fig.suptitle('Edge Detection on Head CT (Fig 10.26)', fontsize=14)

    # (a) Original
    axes[0, 0].imshow(img_norm, cmap='gray')
    axes[0, 0].set_title('(a) Original Image (Head CT)')
    axes[0, 0].axis('off')

    # (b) Thresholded Gradient
    axes[0, 1].imshow(img_b, cmap='gray')
    axes[0, 1].set_title('(b) Thresholded Gradient (15% Max)')
    axes[0, 1].axis('off')

    # (c) Marr-Hildreth
    axes[1, 0].imshow(img_c, cmap='gray')
    axes[1, 0].set_title('(c) Marr-Hildreth (LoG)')
    axes[1, 0].axis('off')

    # (d) Canny
    axes[1, 1].imshow(img_d, cmap='gray')
    axes[1, 1].set_title('(d) Canny Algorithm (Manual)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig1026_HeadCT_EdgeDetection.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Done. Result saved to {save_path}")


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    filename = "Fig1026(a)(headCT-Vandy).tif"
    path = os.path.join(input_dir, filename)

    # 容错机制：如果没有找到指定文件，尝试备用名称
    if not os.path.exists(path):
        # 尝试常见的命名变体
        alt = "Fig1026(a).tif"
        alt_path = os.path.join(input_dir, alt)
        if os.path.exists(alt_path):
            path = alt_path

    process_head_ct(path, output_dir)