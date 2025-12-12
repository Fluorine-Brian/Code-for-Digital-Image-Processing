import cv2
import os

input_dir = "original_image"
output_dir = "output_image"
os.makedirs(output_dir, exist_ok=True)

image_filename = "Fig0905(a)(wirebond-mask).tif"
image_path = os.path.join(input_dir, image_filename)

# 2. 加载图像
binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# cv2.resizeWindow('res', 600, 800)
cv2.imshow('res', binary_image)
cv2.waitKey(0)
