from PIL import Image
import os
import sys

from resizeimage import resizeimage

directory = sys.argv[1]
resized_directory = directory.rsplit('/', 1)[0] + '/resized_images'

for file_name in os.listdir(directory):
    with open(directory + '/' + file_name, 'rb') as f:
        with Image.open(f) as image:
            if not os.path.exists(resized_directory):
                os.makedirs(resized_directory)
            cover = resizeimage.resize_cover(image, [64, 64])
            cover.save(resized_directory + '/' + file_name, image.format)
