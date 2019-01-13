from PIL import Image
import os
import numpy as np
import xmltodict
import scipy
import scipy.misc

black_background = False
new_image_size = [64,64]

def resize_and_crop_images():
    names = [path.rsplit('.',1)[0] for path in os.listdir('../data/xmls')]
    results = []

    size = len(names)

    for i,name in enumerate(names):
        print(str(i+1) + '/' + str(size))
        img = np.array(Image.open('../data/images/' + name + '.jpg'))
        tri = np.array(Image.open('../data/trimaps/' + name + '.png'))
        
        with open('../data/xmls/' + name + '.xml') as xml_file:
            xml = xmltodict.parse(xml_file.read())

        def process_object(obj):
            species = obj['name']

            box = obj['bndbox']

            xmin = int(box['xmin'])
            xmax = int(box['xmax'])
            ymin = int(box['ymin'])
            ymax = int(box['ymax'])
            subimg = img[ymin:ymax,xmin:xmax]
            subtri = tri[ymin:ymax,xmin:xmax]

            if len(np.shape(subimg)) == 3:
                if (black_background):
                    for x in range(xmax - xmin):
                        for y in range(ymax - ymin):
                            if subtri[y,x] == 2:
                                subimg[y,x] = 0,0,0

                subimg = scipy.misc.imresize(subimg, new_image_size)

                Image.fromarray(subimg).save('result'+str(i)+'.png')
                
        if type(xml['annotation']['object']) == list:
            for obj in xml['annotation']['object']:
                process_object(obj)
        else:
            process_object(xml['annotation']['object'])


if __name__ == "__main__":
    resize_and_crop_images()
