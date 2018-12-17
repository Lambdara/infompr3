from PIL import Image
import os
import sys

from resizeimage import resizeimage

FOLDER = sys.argv[1]
RESIZED_FOLDER = FOLDER.rsplit('/', 1)[0] + '/resized_images'

for file_name in os.listdir(FOLDER):
	with open(FOLDER + '/' + file_name, 'rb') as f:
		with Image.open(f) as image:
			if not os.path.exists(RESIZED_FOLDER):
				os.makedirs(RESIZED_FOLDER)
			cover = resizeimage.resize_cover(image, [64, 64])
			cover.save(RESIZED_FOLDER + '/' + file_name, image.format)