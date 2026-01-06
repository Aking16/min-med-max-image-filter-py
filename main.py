import cv2
import numpy as np
import os

# ایجاد دایرکتوری خروجی اگر وجود نداشته باشد
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
src_output_dir = './output/source'
if not os.path.exists(src_output_dir):
    os.makedirs(src_output_dir)

# بارگذاری تصویر
image_path = './input/Noise.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# اعمال فیلتر Median
median_filtered = cv2.medianBlur(image, 3)

# اعمال فیلتر Min با استفاده از عملیات Erosion
min_filtered = cv2.erode(image, np.ones((3, 3), np.uint8))

# اعمال فیلتر Max با استفاده از عملیات Dilation
max_filtered = cv2.dilate(image, np.ones((3, 3), np.uint8))

# اضافه کردن برچسب به هر تصویر
font = cv2.FONT_HERSHEY_SIMPLEX

color = (20, 0, 255) 

# تبدیل تصاویر به رنگی برای نمایش رنگ متن
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
min_filtered_color = cv2.cvtColor(min_filtered, cv2.COLOR_GRAY2BGR)
median_filtered_color = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)
max_filtered_color = cv2.cvtColor(max_filtered, cv2.COLOR_GRAY2BGR)

# اضافه کردن متن با رنگ مشخص شده
cv2.putText(image_color, 'Noisy Image', (10, 30), font, 1, color, 2, cv2.LINE_AA)
cv2.putText(min_filtered_color, 'Min Filter', (10, 30), font, 1, color, 2, cv2.LINE_AA)
cv2.putText(median_filtered_color, 'Median Filter', (10, 30), font, 1, color, 2, cv2.LINE_AA)
cv2.putText(max_filtered_color, 'Max Filter', (10, 30), font, 1, color, 2, cv2.LINE_AA)

# چسباندن تصاویر به صورت عمودی
top_row = np.hstack((image_color, median_filtered_color))  # چسباندن تصویر noisy و median
bottom_row = np.hstack((min_filtered_color, max_filtered_color))  # چسباندن min و max

# چسباندن ردیف‌ها به صورت عمودی
final_image = np.vstack((top_row, bottom_row))

# ذخیره‌سازی تصویر نهایی
cv2.imwrite(os.path.join(output_dir, 'final_filtered_image.jpg'), final_image)
cv2.imwrite(os.path.join(src_output_dir, 'min_filtered_image.jpg'), min_filtered_color)
cv2.imwrite(os.path.join(src_output_dir, 'max_filtered_image.jpg'), max_filtered_color)
cv2.imwrite(os.path.join(src_output_dir, 'med_filtered_image.jpg'), median_filtered_color)

# نمایش تصویر نهایی
cv2.imshow('Final Merged Image', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
