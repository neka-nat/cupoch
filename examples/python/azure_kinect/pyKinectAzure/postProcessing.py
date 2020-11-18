import cv2 
import numpy as np


def smooth_depth_image(depth_image, max_hole_size=10):
	"""Smoothes depth image by filling the holes using inpainting method

		Parameters:
		depth_image(Image): Original depth image
		max_hole_size(int): Maximum size of hole to fill
			
		Returns:
		Image: Smoothed depth image
		
		Remarks:
		Bigger maximum hole size will try to fill bigger holes but requires longer time
		"""
	mask = np.zeros(depth_image.shape,dtype=np.uint8)
	mask[depth_image==0] = 1

	# Do not include in the mask the holes bigger than the maximum hole size
	kernel = np.ones((max_hole_size,max_hole_size),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 1)
	mask = mask - erosion

	smoothed_depth_image = cv2.inpaint(depth_image.astype(np.uint16),mask,max_hole_size,cv2.INPAINT_NS)

	return smoothed_depth_image