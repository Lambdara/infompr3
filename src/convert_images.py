from PIL import Image
import os
import numpy as np
import xmltodict
import scipy
import scipy.misc
from resizeimage import resizeimage

resize_images = True
resize_trimaps = True
black_background = True
new_image_size = [64,64]

directory_path = os.path.dirname(os.getcwd())
images_directory_path = directory_path + '\\data\\images'
new_images_directory_path = directory_path + '\\data\\new_images'
trimaps_directory = directory_path + '\\data\\trimaps'
new_trimaps_directory = directory_path + '\\data\\new_trimaps'

if not os.path.exists(new_images_directory_path):
    os.makedirs(new_images_directory_path)


def resize_and_crop_images():
    names = [path.rsplit('.',1)[0] for path in os.listdir('../data/xmls')]
    cats = []
    dogs = []

    for name in names:
        img = np.array(Image.open('../data/images/' + name + '.jpg'))

        with open('../data/xmls/' + name + '.xml') as xml_file:
            xml = xmltodict.parse(xml_file.read())

        def process_object(obj):
            species = obj['name']

            box = obj['bndbox']

            xmin = int(box['xmin'])
            xmax = int(box['xmax'])
            ymin = int(box['ymin'])
            ymax = int(box['ymax'])
            subimg = scipy.misc.imresize(img[ymin:ymax,xmin:xmax],new_image_size)

            if np.shape(subimg) == (new_image_size[0],new_image_size[1],3):
                Image.fromarray(subimg).save(new_images_directory_path + '/' + name + '.jpg')
                if species == 'cat':
                    cats.append(subimg)
                elif species == 'dog':
                    dogs.append(subimg)
                else:
                    print('Unrecognized species')

        if type(xml['annotation']['object']) == list:
            for obj in xml['annotation']['object']:
                process_object(obj)
        else:
            process_object(xml['annotation']['object'])
    
def resize_and_crop_trimaps():
    if not os.path.exists(new_trimaps_directory):
        os.makedirs(new_trimaps_directory)
    names = [path.rsplit('.',1)[0] for path in os.listdir('../data/xmls')]
    for name in names:
        img = np.array(Image.open('../data/trimaps/' + name + '.png'))
        with open('../data/xmls/' + name + '.xml') as xml_file:
            xml = xmltodict.parse(xml_file.read())

        def process_object(obj):
            species = obj['name']

            box = obj['bndbox']

            xmin = int(box['xmin'])
            xmax = int(box['xmax'])
            ymin = int(box['ymin'])
            ymax = int(box['ymax'])
            subimg = scipy.misc.imresize(img[ymin:ymax,xmin:xmax],new_image_size)
            if np.shape(subimg) == (new_image_size[0],new_image_size[1]):
                Image.fromarray(subimg).save(new_trimaps_directory + '\\' + name + '.png')
            

        if type(xml['annotation']['object']) == list:
            for obj in xml['annotation']['object']:
                process_object(obj)
        else:
            process_object(xml['annotation']['object'])


#There are pixels between the edge and the foreground that are annotated as background, but which you actually want to include in the picture. This determines whether or not it is such a pixel
def in_between_pixel(x,y, trimap, trimapPixels):
    no_edge = x != 0 and y != 0 and x != trimap.size[0] - 1 and y != trimap.size[1] - 1
    if no_edge:
        edge = trimapPixels[x+1,y] == 3 or trimapPixels[x-1,y] == 3 or trimapPixels[x,y-1] == 3 or trimapPixels[x,y+1] == 3 or trimapPixels[x-1,y-1] == 3 or trimapPixels[x-1,y+1] == 3 or trimapPixels[x+1,y-1] == 3 or trimapPixels[x+1,y+1] == 3
        face = trimapPixels[x+1,y] == 1 or trimapPixels[x-1,y] == 1 or trimapPixels[x,y-1] == 1 or trimapPixels[x,y+1] == 1 or trimapPixels[x-1,y-1] == 1 or trimapPixels[x-1,y+1] == 1 or trimapPixels[x+1,y-1] == 1 or trimapPixels[x+1,y+1] == 1
        return edge and face
    else:
        return False

def make_backgrounds_black():
    names = [path.rsplit('.',1)[0] for path in os.listdir('../data/new_images')]
    for name in names:
        with Image.open(new_images_directory_path + '\\' + name + '.jpg') as image:
            with Image.open(new_trimaps_directory + '\\' + name + '.png') as trimap:
                trimapPixels = trimap.load()
                imagePixels = image.load()
                for x in range(trimap.size[0]):
                    for y in range(trimap.size[1]):
                        if trimapPixels[x,y] == 2 and not in_between_pixel(x,y, trimap, trimapPixels):
                            imagePixels[x,y] = 0,0,0
                image.save(new_images_directory_path + '\\' + name + '.jpg')            

if resize_images:
    resize_and_crop_images()

if resize_trimaps:
    resize_and_crop_trimaps()

if black_background:
    make_backgrounds_black()