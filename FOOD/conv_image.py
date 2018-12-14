import cv2
from PIL import Image
import numpy as np

file_list = np.load('file_list.npy')
for i, img_path in enumerate(file_list):
	img = Image.open(img_path)
	img = img.resize((256, 256))
	path = img_path.replace('food-101', 'food-101_256')
	img.save(path, 'JPEG', quality=100, optimize=True)
	print(i)

print('FINISH')
