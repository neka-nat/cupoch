import cupoch as cph
import numpy as np

img = cph.io.read_image("../../testdata/lena_gray.jpg")
cph.visualization.draw_geometries([img])

g_img = img.filter(cph.geometry.ImageFilterType.Gaussian3)
cph.visualization.draw_geometries([g_img])

b_img = img.bilateral_filter(3, 0.1, 10.0)
cph.visualization.draw_geometries([b_img])