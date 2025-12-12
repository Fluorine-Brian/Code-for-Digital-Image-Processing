import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
MANUAL_INVERT = False


def perform_dilation(binary_image, kernel):
    """
    执行形态学膨胀操作。
    """
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    return dilated_image


if __name__ == "__main__":
    input_dir = "original_image"
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)
    image_filename = "Fig0907(a)(text_gaps_1_and_2_pixels).tif"
    image_path = os.path.join(input_dir, image_filename)
    raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if raw_image is None:
        raw_image = np.zeros((300, 300), dtype=np.uint8)
        cv2.putText(raw_image, "broken text", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1)
        raw_image[140:160, 100:102] = 0

        # 强制归一化到 0-255
    if raw_image.max() <= 1:
        raw_image = (raw_image * 255).astype(np.uint8)

    # --- 3. 智能二值化与极性检测 ---
    # 我们需要：前景(文字) = 255/True, 背景 = 0/False

    # 计算平均亮度
    mean_val = np.mean(raw_image)

    # 自动判断：如果平均亮度很高(>127)，说明是白底黑字，需要反转
    if mean_val > 127:
        print("检测到白底黑字，自动反转为黑底白字(前景=白)...")
        _, binary_image = cv2.threshold(raw_image, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        print("检测到黑底白字，保持原样...")
        _, binary_image = cv2.threshold(raw_image, 127, 255, cv2.THRESH_BINARY)

    # 手动反转覆盖
    if MANUAL_INVERT:
        print(">>> 应用手动反转 <<<")
        binary_image = cv2.bitwise_not(binary_image)

    # --- 4. 定义结构元 (Fig 9.7 b) ---
    # 十字形结构元
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    # 或者使用 OpenCV 内置函数:
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # --- 5. 执行膨胀 ---
    dilated_image = perform_dilation(binary_image, kernel)

    # --- 6. 截取局部细节 ('e' 和 'a') ---
    # 使用顶部定义的切片进行截取
    slice_e = (slice(342, 367), slice(433, 467))
    e_orig = binary_image[slice_e]
    e_dilated = dilated_image[slice_e]

    # --- 7. 可视化布局 ---
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f'Morphological Dilation (Fig 9.7) - {image_filename}', fontsize=16)

    # 使用 GridSpec 进行复杂布局
    # 2行, 4列
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], hspace=0.3)

    # 第一行：显示完整的大图
    ax1 = fig.add_subplot(gs[0, :2])  # 占左边两列
    ax1.imshow(binary_image, cmap='gray')
    ax1.set_title('a) Original Image with Broken Text')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2:])  # 占右边两列
    ax2.imshow(dilated_image, cmap='gray')
    ax2.set_title('c) Dilated Image (Repaired)')
    ax2.axis('off')

    # 第二行：显示放大的细节
    # 细节 1: 'e' 原图
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(e_orig, cmap='gray', interpolation='nearest')  # nearest 用于显示像素格
    ax3.set_title("Zoom 'e' (Original)")
    ax3.axis('off')

    # 细节 2: 'e' 膨胀后
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(e_dilated, cmap='gray', interpolation='nearest')
    ax4.set_title("Zoom 'e' (Dilated)")
    ax4.axis('off')

    # 保存结果
    save_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_dilation_result.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"处理完成。结果已保存至: {save_path}")
    print("提示：如果底部的小图没有准确显示 'e' 或 'a'，请调整代码顶部的 slice_e 和 slice_a 坐标。")