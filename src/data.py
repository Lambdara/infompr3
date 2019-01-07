import os
from PIL import Image  # pillow package
import numpy as np
import xmltodict
import scipy
import scipy.misc

imgsize = (32, 32)


def get_cats_and_dogs():
    names = [path.rsplit('.', 1)[0] for path in os.listdir('../data/xmls')]
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

            subimg = scipy.misc.imresize(img[ymin:ymax,xmin:xmax],imgsize)
            if np.shape(subimg) == (32,32):
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
    return cats, dogs

