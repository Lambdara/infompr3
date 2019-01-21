from __future__ import division
import numpy as np
import scipy.misc


def get_batch(data, labels, batch_size, idx, image_height, image_width, grayscale=False):
    batch_files = data[idx * batch_size:(idx + 1) * batch_size]
    batch = [
        get_image(batch_file, image_height, image_width, resize_height=image_height, resize_width=image_width,
                  grayscale=grayscale) for batch_file in batch_files]
    if grayscale:
        x = np.array(batch).astype(np.float32)[:, :, :, None]
    else:
        x = np.array(batch).astype(np.float32)

    y = labels[idx * batch_size:(idx + 1) * batch_size]
    return x, y


def imread(path, grayscale=False):
    if grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)


def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width,
                                    resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

