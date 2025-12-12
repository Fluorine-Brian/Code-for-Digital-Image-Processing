import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image


# ==========================================
# 1. 通用辅助函数 (高斯核, Sobel等)
# ==========================================

def gaussian_kernel(size, sigma=1):
    """ 生成高斯核 (用于手搓 Canny) """
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def apply_average_filter(img_arr, kernel_size=5):
    """ 均值滤波 (用于方法 b) """
    img_float = img_arr.astype(float)
    return ndimage.uniform_filter(img_float, size=kernel_size)


# ==========================================
# 2. 方法 (b) 实现: 平滑 + Sobel + 阈值
# ==========================================
def get_image_b(img):
    # 1. 平滑 (5x5 均值)
    img_smoothed = apply_average_filter(img, kernel_size=5)

    # 2. Sobel 梯度
    img_float = img_smoothed.astype(float) / 255.0
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    gx = ndimage.convolve(img_float, kx)
    gy = ndimage.convolve(img_float, ky)
    magnitude = np.abs(gx) + np.abs(gy)

    # 3. 阈值处理 (33% of Max)
    max_val = np.max(magnitude)
    threshold = 0.33 * max_val
    edges = (magnitude >= threshold).astype(np.uint8) * 255
    return edges


# ==========================================
# 3. 方法 (c) 实现: Marr-Hildreth (LoG)
# ==========================================
def get_image_c(img):
    img_float = img.astype(float)
    # 1. LoG (sigma=4, n=25)
    # truncate=3.0 -> radius=12 -> size=25
    log_resp = ndimage.gaussian_laplace(img_float, sigma=4, truncate=3.0)

    # 2. 过零点检测 (阈值 = 4% Max)
    max_log = np.max(np.abs(log_resp))
    threshold = 0.04 * max_log

    edges = np.zeros_like(img, dtype=np.uint8)

    # 水平检查
    curr_h = log_resp[:, :-1]
    right_h = log_resp[:, 1:]
    sign_diff_h = (curr_h * right_h) < 0
    mag_check_h = np.abs(curr_h - right_h) > threshold
    edges[:, :-1][sign_diff_h & mag_check_h] = 255

    # 垂直检查
    curr_v = log_resp[:-1, :]
    down_v = log_resp[1:, :]
    sign_diff_v = (curr_v * down_v) < 0
    mag_check_v = np.abs(curr_v - down_v) > threshold
    edges[:-1, :][sign_diff_v & mag_check_v] = 255

    return edges


# ==========================================
# 4. 方法 (d) 实现: 手搓 Canny 算法
# ==========================================
def sobel_filters_manual(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta


def non_max_suppression_manual(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                # 0度
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # 45度
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # 90度
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
    M, N = img.shape
    strong_rows, strong_cols = np.where(img == strong)
    # 使用栈进行非递归 DFS
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

    # 清理未连接的弱边缘
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                img[i, j] = 0
    return img


def get_image_d_manual_canny(img):
    img_float = img.astype(float)

    # 1. 高斯平滑 (Sigma=4, Size=25)
    # 手动生成核并卷积
    kernel = gaussian_kernel(size=25, sigma=4)
    img_smoothed = ndimage.convolve(img_float, kernel)

    # 2. 计算梯度
    grad_mag, grad_dir = sobel_filters_manual(img_smoothed)

    # 3. 非极大值抑制
    img_nms = non_max_suppression_manual(grad_mag, grad_dir)

    # 4. 双阈值 (TL=0.04*Max, TH=0.10*Max)
    # 直接计算绝对值，避免之前的 ZeroDivisionError
    max_val = img_nms.max()
    lowThreshold = max_val * 0.04
    highThreshold = max_val * 0.10

    res = np.zeros_like(img_nms, dtype=np.int32)
    weak_val = np.int32(50)
    strong_val = np.int32(255)

    strong_i, strong_j = np.where(img_nms >= highThreshold)
    weak_i, weak_j = np.where((img_nms <= highThreshold) & (img_nms >= lowThreshold))

    res[strong_i, strong_j] = strong_val
    res[weak_i, weak_j] = weak_val

    # 5. 滞后边界跟踪
    img_final = hysteresis_manual(res, weak_val, strong_val)

    return img_final


# ==========================================
# 主程序
# ==========================================
def process_comparison(image_path, output_dir):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # Load Image
    pil_img = Image.open(image_path).convert('L')
    img = np.array(pil_img)

    # Generate all 4 images
    print("Generating Image (b)...")
    img_b = get_image_b(img)

    print("Generating Image (c)...")
    img_c = get_image_c(img)

    print("Generating Image (d) using Manual Canny...")
    img_d = get_image_d_manual_canny(img)

    # Visualization
    plt.rcParams['font.family'] = 'sans-serif'  # Standard font

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a)
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('(a) Original Image')
    axes[0, 0].axis('off')

    # (b)
    axes[0, 1].imshow(img_b, cmap='gray')
    axes[0, 1].set_title('(b) Thresholded Gradient (Smoothed)')
    axes[0, 1].axis('off')

    # (c)
    axes[1, 0].imshow(img_c, cmap='gray')
    axes[1, 0].set_title('(c) Marr-Hildreth (LoG)')
    axes[1, 0].axis('off')

    # (d)
    axes[1, 1].imshow(img_d, cmap='gray')
    axes[1, 1].set_title('(d) Canny Algorithm (Manual)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig1025_Full_Comparison_Manual.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Done. Result saved to {save_path}")


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)

    filename = "Fig1016(a)(building_original).tif"
    path = os.path.join(input_dir, filename)

    process_comparison(path, output_dir)