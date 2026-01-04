import cv2
import numpy as np
import os

# ایجاد فولدر خروجی اگر وجود نداشته باشد
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# بارگذاری تصویر
image_path = './input/Noise.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# اعمال فیلتر Median
median_filtered = cv2.medianBlur(image, 3)

# اعمال فیلتر Min با استفاده از عملیات Erosion
min_filtered = cv2.erode(image, np.ones((3, 3), np.uint8))

# اعمال فیلتر Max با استفاده از عملیات Dilation
max_filtered = cv2.dilate(image, np.ones((3, 3), np.uint8))

# ذخیره‌سازی تصاویر فیلتر شده
cv2.imwrite(os.path.join(output_dir, 'min_filtered.png'), min_filtered)
cv2.imwrite(os.path.join(output_dir, 'max_filtered.png'), max_filtered)
cv2.imwrite(os.path.join(output_dir, 'median_filtered.png'), median_filtered)

# نمایش تصاویر فیلتر شده
cv2.imshow('Noisy Image', image)
cv2.imshow('Min Filtered', min_filtered)
cv2.imshow('Max Filtered', max_filtered)
cv2.imshow('Median Filtered', median_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
