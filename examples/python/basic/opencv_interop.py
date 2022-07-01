import cupoch as cph
import numpy as np
import cv2

img_src = cv2.imread("../../testdata/lena_color.jpg")

# uint8, 3channels
img_cpu = np.ascontiguousarray(img_src[:, :, ::-1])
print(img_cpu.dtype)
img_gpu = cph.geometry.Image(img_cpu)
cph.visualization.draw_geometries([img_gpu])

# flat32, 1channels
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_gray = img_gray.astype(np.float32) / 255.0
print(img_gray.dtype)
img_gpu = cph.geometry.Image(img_gray)
cph.visualization.draw_geometries([img_gpu])
