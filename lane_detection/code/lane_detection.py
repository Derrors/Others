# encoding:utf-8

import matplotlib.pyplot as plt
import numpy as np 
import cv2

# 解决中文显示
plt.rcParams['font.sans-serif']=['SimHei']


class lane(object):

	def __init__(self, image):
		self.img = image
		self.width = self.img.shape[1]
		self.height = self.img.shape[0]

		self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

		# 对要识别的道路区域进行切片
		self.roi_img = self.gray_img[int(0.6 * self.height): ,]
		self.roi_img_rgb = self.img[int(0.6 * self.height): ,]

		# 根据切片区域进行计算阈值，这里取均值的2倍，最大阈值取均值的3倍
		self.threshold = int(2 * np.mean(self.roi_img))
		self.threshold_max = int(3 * np.mean(self.roi_img))

	# 高斯滤波
	def gaussian_blur(self, kernel_size):
		self.gaus_img = cv2.GaussianBlur(self.roi_img, (kernel_size, kernel_size), 0)

	# 阈值分割
	def threshold_img(self, threshold):
		ret, binary =  cv2.threshold(self.gaus_img, threshold, 255, cv2.THRESH_BINARY)
		self.bin_img = binary

	# 边缘检测
	def canny(self, low_threshold, high_threshold):
		self.edges = cv2.Canny(self.bin_img, low_threshold, high_threshold)

	# 将检测出的边缘进行 “膨胀” 操作
	def dilate(self):
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
		self.dilated_edges = cv2.dilate(self.edges, kernel)

	# 霍夫变换检测直线
	def hough_lines(self, rho, theta, threshold, min_line_len, max_line_gap):
		self.lines = cv2.HoughLinesP(self.dilated_edges, rho, theta, threshold, min_line_len, max_line_gap)

	# 在图像中画出检测的道路
	def draw_lines(self, color=[255, 0, 0], thickness=1):
		left_lines = []
		right_lines = []
		top_y = 1e6

		# 根据直线的斜率分为左边直线和右边直线
		for line in self.lines:
			for x1, y1, x2, y2 in line:
				if x1 != x2:
					k = (y2 - y1) / (x2 - x1)
					if k > 0:
						left_lines.append([x1, y1, x2, y2])
					else:
						right_lines.append([x1, y1, x2, y2])
				if top_y > y1:
					top_y = y1
				if top_y > y2:
					top_y = y2

		# 检测左侧道路直线
		if len(left_lines) > 0:
			left_line = [0, 0, 0, 0]
			for line in left_lines:
				for i in range(4):
					left_line[i] += (line[i] / len(left_lines))
			k = (left_line[3] - left_line[1]) / (left_line[2] - left_line[0])
			top_x = left_line[0] + (top_y - left_line[1]) / k
			bottom_x  = left_line[0] + (self.roi_img_rgb.shape[0] - left_line[1]) / k

			cv2.line(self.roi_img_rgb, (int(bottom_x), self.roi_img_rgb.shape[0]), (int(top_x), int(top_y)), color, thickness * 10)

		# 检测右侧道路直线
		if len(right_lines) > 0:
			right_line = [0, 0, 0, 0]
			for line in right_lines:
				for i in range(4):
					right_line[i] += (line[i] / len(right_lines))
			k = (right_line[3] - right_line[1]) / (right_line[2] - right_line[0])
			top_x = right_line[0] + (top_y - right_line[1]) / k
			bottom_x  = right_line[0] + (self.roi_img_rgb.shape[0] - right_line[1]) / k

			cv2.line(self.roi_img_rgb, (int(bottom_x), self.roi_img_rgb.shape[0]), (int(top_x), int(top_y)), color, thickness * 10)


if __name__ == '__main__':
	# 配置测试图片路径
	path = '../test_images/'

	for i in range(5):
		image = cv2.imread(path + str(i) + '.jpg')

		plt.figure(i, figsize=(10, 6))
		plt.subplot(2, 2, 1)
		plt.imshow(image)
		plt.title("读取图像")

		img = lane(image)

		img.gaussian_blur(3)
		img.threshold_img(img.threshold)
		plt.subplot(2, 2, 2)
		plt.imshow(img.bin_img, cmap="gray")
		plt.title("高斯滤波并二值化图像")

		img.canny(img.threshold, img.threshold_max)
		img.dilate()
		plt.subplot(2, 2, 3)
		plt.imshow(img.dilated_edges, cmap="gray")
		plt.title("边缘检测并腐蚀图像")

		img.hough_lines(2, np.pi/180, img.threshold, 150, 50)
		img.draw_lines()
		plt.subplot(2, 2, 4)
		plt.imshow(img.roi_img_rgb, cmap="gray")
		plt.title("检测道路结果")

		plt.savefig('../result/reutlt_image' + str(i))
	plt.show()
