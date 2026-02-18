# 检查OpenCV版本和编码器支持
import os
import numpy as np
import cv2
print(f"OpenCV版本: {cv2.__version__}")

# 测试简单图像保存
tar
test_img = np.zeros((10, 10, 3), dtype=np.uint8)
test_path = os.path.join(target_dir, "opencv_test.png")
test_success = cv2.imwrite(test_path, test_img)
print(f"OpenCV测试保存: {test_success}")